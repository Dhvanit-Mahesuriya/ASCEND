"""
ssl_pretrain.py
================

Self-supervised pretraining of the SAMoE-C `SharedBackbone` (models.py),
adapting the logic of CIG-MAE -- "Cross-Modal Information-Guided Masked
Autoencoder for Self-Supervised WiFi Sensing" (Liu, Hao & Zou, IEEE IoT-J,
2026) -- to SAMoE-C ("Scene-Adaptive Continual Learning for CSI-based HAR
with Mixture of Experts", Zheng, Mao & Ho, arXiv:2605.06447v1).

This REPLACES the labeled part of Phase 1 (Sec. IV-A, Eq. 9) of the
Decoupled Continual Training Protocol: instead of training the shared
backbone jointly and supervised with specialist-0 on D1, we self-supervise
the backbone on UNLABELED CSI (activity labels are discarded), freeze it
exactly as before, and then fine-tune specialist-0 (and every later
specialist) with plain supervised cross-entropy via `train.train_specialist`,
unchanged. Phase 2 (incremental specialists) and Phase 3 (router update via
the replay buffer) are also completely unchanged.

--------------------------------------------------------------------------
What is adopted from CIG-MAE, and what is deliberately dropped
--------------------------------------------------------------------------
ADOPTED:
  - Adaptive Information-Guided Masking (AIM): a lightweight policy network
    (`AIMPolicyNetwork`, CIG-MAE Eq. 2-3) scores subcarrier patches, and a
    Gumbel-Top-K sampler decides which patches stay VISIBLE, spending the
    "visibility budget" on hard-to-reconstruct (likely activity-related)
    regions instead of masking uniformly at random.
  - High-mask-ratio masked reconstruction (CIG-MAE found rho=0.95 optimal;
    we default lower here since SAMoE-C's CSI windows are much shorter,
    T=10 vs. CIG-MAE's T=200 -- see `Config.ssl_mask_ratio`).
  - MAE (L1) loss on an UN-normalized reconstruction target rather than
    MSE, for robustness to CSI hardware noise spikes (CIG-MAE Sec. III-C
    and Table IV ablation).
  - Decoupled backbone/policy optimization (CIG-MAE Algorithm 1, Phase 4):
    the backbone+decoder minimize reconstruction loss; the policy is
    updated independently via a REINFORCE-style estimator (Eq. 4) using the
    (stop-gradient) reconstruction error as the reward signal.

DROPPED / NOT DIRECTLY APPLICABLE HERE:
  - CIG-MAE is a DUAL-STREAM (amplitude + phase) framework with a Barlow
    Twins (BT) cross-modal alignment loss L_BT (Sec. III-D). This repo's
    data pipeline (`fast_preprocess.py`) only extracts `CSIamp` from the
    .mat files -- no phase is stored -- so there is no second modality to
    align against here. `SharedBackbone.C = 3` in this codebase means
    N_tx * N_rx antenna pairs, NOT "amplitude vs. phase" streams.
    If you re-run `preprocess_mmfi.py` and additionally export unwrapped
    CSI phase per antenna pair, you can instantiate a second
    `SharedBackbone` copy and re-enable BT alignment -- see the
    `enable_bt` / `pretrain_backbone_cigmae_dualstream` stub at the bottom
    of this file for exactly where that would plug in.
  - Masking here only operates along the subcarrier axis S, independently
    per timestep t, mirroring the fact that `SharedBackbone` itself does
    not mix information across timesteps (Table I: the shared backbone is
    applied per-timestep; temporal mixing only happens later, inside each
    `Specialist`'s Bi-GRU). This matches the "simple version" discussed
    for this codebase; a temporal-masking variant would need a small
    transformer/GRU mixer inserted between backbone and decoder purely for
    pretraining (not included here to keep the change minimal).

Usage
-----
    from ssl_pretrain_samoec import pretrain_backbone_cigmae
    from models_samoec import SAMoEC

    model = SAMoEC(cfg).to(device)
    pretrain_backbone_cigmae(model, cfg, device, unlabeled_dataset)
    # model.backbone is now frozen, exactly as after train_initial_domain()

    # Phase 1's specialist-0 must still be trained supervised (labels are
    # only needed for classification, not for backbone pretraining):
    spec_idx = model.add_specialist()
    assert spec_idx == 0
    train_specialist(model, cfg, device, train_ds_D1, spec_idx=0)

    # Phases 2/3 proceed exactly as in the original protocol.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Patch utilities: split the S=114 subcarrier axis into patches, independently
# per timestep (consistent with SharedBackbone's per-timestep processing).
# --------------------------------------------------------------------------- #
def patchify_subcarrier(X, patch_size):
    """
    X: (B, C, T, S) raw CSI amplitude tensor.
    Returns:
      patches:  (B, T, n_patches, C * patch_size)  -- flattened patch content
      n_patches: int
      S_trim:    S rounded DOWN to a multiple of patch_size
    """
    B, C, T, S = X.shape
    n_patches = S // patch_size
    S_trim = n_patches * patch_size
    x = X[..., :S_trim]                                    # (B,C,T,S_trim)
    x = x.view(B, C, T, n_patches, patch_size)
    x = x.permute(0, 2, 3, 1, 4).contiguous()               # (B,T,n_patches,C,patch_size)
    patches = x.view(B, T, n_patches, C * patch_size)
    return patches, n_patches, S_trim


def expand_patch_mask(vis_mask, B, T, n_patches, patch_size, C, S, S_trim, device):
    """
    vis_mask: (B*T, n_patches) bool, True = VISIBLE.
    Returns a float (B, C, T, S) mask with 1 = visible, 0 = hidden/masked,
    broadcast identically across the C (antenna-pair) axis and expanded back
    out to per-subcarrier resolution.
    """
    m = vis_mask.view(B, T, n_patches, 1, 1).expand(-1, -1, -1, C, patch_size)
    m = m.permute(0, 3, 1, 2, 4).reshape(B, C, T, n_patches * patch_size)
    if S_trim < S:
        pad = torch.ones(B, C, T, S - S_trim, dtype=m.dtype, device=device)  # leave the
        m = torch.cat([m, pad], dim=-1)                                     # remainder visible
    return m.float()


# --------------------------------------------------------------------------- #
# AIM: Adaptive Information-Guided Masking policy network (CIG-MAE Sec. III-B)
# --------------------------------------------------------------------------- #
class AIMPolicyNetwork(nn.Module):
    """
    Single-layer Transformer-encoder policy that scores the contextual
    importance of each subcarrier patch (Eq. 2-3 of CIG-MAE):
        Z = {x_i} + MHA(LN({x_i}))
        U = Z + MLP(LN(Z))
        p_i = softmax(w^T U_i)     (probability patch i should be VISIBLE)
    """
    def __init__(self, patch_dim, hidden=128, n_heads=4):
        super().__init__()
        self.embed = nn.Linear(patch_dim, hidden)
        self.ln_in = nn.LayerNorm(hidden)
        self.mha = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.ln_mlp = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.score = nn.Linear(hidden, 1)

    def forward(self, patches):
        # patches: (N, n_patches, patch_dim), N = B*T (each timestep scored independently)
        x = self.embed(patches)
        xn = self.ln_in(x)
        attn_out, _ = self.mha(xn, xn, xn)
        z = x + attn_out
        u = z + self.mlp(self.ln_mlp(z))
        logits = self.score(u).squeeze(-1)          # (N, n_patches)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


# --------------------------------------------------------------------------- #
# Domain-Conditioned AIM: masking policy that adapts to each domain's unique
# channel characteristics (NOVEL — CIG-MAE has no concept of domains)
# --------------------------------------------------------------------------- #
class DomainConditionedAIMPolicy(nn.Module):
    """
    Extends AIMPolicyNetwork with domain conditioning. Different WiFi
    environments have different multipath propagation profiles, so the
    informative subcarrier patterns differ per domain. This module injects
    a learnable domain embedding into the patch scoring pipeline.

    The domain embedding is added to the patch representations BEFORE the
    Transformer-based scoring, allowing the MHA to attend differently
    depending on the domain context.

    This is a NOVEL component: CIG-MAE's AIM is domain-agnostic, and
    SAMoE-C doesn't use adaptive masking at all.
    """
    def __init__(self, patch_dim, hidden=128, n_heads=4, n_domains=4):
        super().__init__()
        self.domain_embeddings = nn.Embedding(n_domains, hidden)
        self.embed = nn.Linear(patch_dim, hidden)
        self.ln_in = nn.LayerNorm(hidden)
        self.mha = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.ln_mlp = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.score = nn.Linear(hidden, 1)

    def forward(self, patches):
        """Unconditioned forward (domain-agnostic, for backward compat)."""
        x = self.embed(patches)
        xn = self.ln_in(x)
        attn_out, _ = self.mha(xn, xn, xn)
        z = x + attn_out
        u = z + self.mlp(self.ln_mlp(z))
        logits = self.score(u).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        return probs, logits

    def forward_conditioned(self, patches, domain_idx):
        """
        Domain-conditioned forward pass.

        Args:
            patches: (N, n_patches, patch_dim) — N = B*T
            domain_idx: (N,) long tensor — domain index per sample

        Returns:
            probs: (N, n_patches) — visibility probabilities
            logits: (N, n_patches) — raw scores
        """
        x = self.embed(patches)                    # (N, n_patches, hidden)
        d_emb = self.domain_embeddings(domain_idx)  # (N, hidden)
        # Inject domain context: broadcast across patches
        x = x + d_emb.unsqueeze(1)
        xn = self.ln_in(x)
        attn_out, _ = self.mha(xn, xn, xn)
        z = x + attn_out
        u = z + self.mlp(self.ln_mlp(z))
        logits = self.score(u).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


def gumbel_topk_visible_mask(probs, n_visible, eps=1e-9):
    """
    Gumbel-Top-K sampling (CIG-MAE Sec. III-B): partitions the L patches into
    a visible set V of fixed size `n_visible` and a masked set M = L \\ V,
    sampled from the policy's categorical distribution `probs`.
    Returns a bool mask (N, L) with True = VISIBLE.
    """
    N, L = probs.shape
    g = -torch.log(-torch.log(torch.rand_like(probs) + eps) + eps)
    scores = torch.log(probs + eps) + g
    topk_idx = scores.topk(n_visible, dim=-1).indices
    mask = torch.zeros(N, L, dtype=torch.bool, device=probs.device)
    mask.scatter_(1, topk_idx, True)
    return mask


# --------------------------------------------------------------------------- #
# Lightweight reconstruction decoder, pretraining-only (discarded afterward)
# --------------------------------------------------------------------------- #
class CSIReconstructionDecoder(nn.Module):
    """Z (B*T, 256, S') -> reconstructed CSI amplitude (B*T, C, S)."""
    def __init__(self, in_ch, out_ch, out_len, mid_channels=(144, 96, 64)):
        super().__init__()
        layers, ch = [], in_ch
        for m in mid_channels:
            layers += [
                nn.ConvTranspose1d(ch, m, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(m),
                nn.ReLU(inplace=True),
            ]
            ch = m
        self.up = nn.Sequential(*layers)
        self.final = nn.Conv1d(ch, out_ch, kernel_size=3, padding=1)
        self.out_len = out_len

    def forward(self, z):
        x = self.up(z)
        x = F.interpolate(x, size=self.out_len, mode="linear", align_corners=False)
        return self.final(x)


# --------------------------------------------------------------------------- #
# Barlow Twins projector & helpers for intra-stream consistency
# (single-stream substitute for CIG-MAE's dual-stream cross-modal BT loss,
# Sec. III-D / Eq. 9-11).  Two independently-masked views of the same CSI
# window are encoded, projected, BATCH-NORMALIZED per feature dimension
# (the actual collapse-prevention mechanism -- NOT optional), and pushed
# toward identical cross-correlation.
# --------------------------------------------------------------------------- #
def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BTProjector(nn.Module):
    """2-layer MLP projector with batch normalization on its output, matching
    the Barlow Twins projector design (Zbontar et al., 2021)."""
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        # Final BN: z-scores each feature dimension across the batch
        # BEFORE the cross-correlation matrix.  This is the mechanism
        # that prevents representation collapse without negative samples
        # (Zbontar et al. Sec. 3).
        self.bn_out = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, x):
        return self.bn_out(self.net(x))


# --------------------------------------------------------------------------- #
# Main entry point: CIG-MAE-style pretraining of SAMoE-C's SharedBackbone
# --------------------------------------------------------------------------- #
def pretrain_backbone_cigmae(model, cfg, device, unlabeled_dataset, policy=None, verbose=True):
    """
    Self-supervised, single-stream (amplitude-only) adaptation of CIG-MAE
    Algorithm 1. Trains `model.backbone` + a throwaway decoder + a Barlow
    Twins projector + an AIM policy network to reconstruct masked CSI
    amplitude while enforcing representation consistency across two
    independently-masked views of each sample (intra-stream BT loss),
    then freezes the backbone (mirrors `model.freeze_backbone()` at the
    end of the original supervised `train_initial_domain`).

    `unlabeled_dataset` may be D1 alone (matching the original protocol's
    "Phase 1 uses D1 only") or a concatenation of several/all domains --
    since no labels or domain ids are used, pooling data across domains is
    valid and typically improves representation quality (see CIG-MAE's
    Sec. IV-F data-efficiency study: more unlabeled pretraining data helps
    most in noisy / high-variance settings, which extra CSI domains are).

    Returns (model, policy, decoder, bt_projector). `policy`, `decoder`,
    and `bt_projector` are only needed for pretraining and can be discarded
    afterward.
    """
    C, S, T = cfg.C, cfg.S, cfg.T
    patch_size = cfg.ssl_patch_size
    n_patches = S // patch_size
    n_visible = max(1, int(round((1.0 - cfg.ssl_mask_ratio) * n_patches)))
    if verbose:
        print(f"  [CIG-MAE pretrain] S={S}, patch_size={patch_size} -> "
              f"{n_patches} patches/timestep, mask_ratio={cfg.ssl_mask_ratio:.2f} "
              f"({n_visible} visible / {n_patches})")

    patch_dim = C * patch_size
    if policy is None:
        policy = AIMPolicyNetwork(patch_dim, hidden=cfg.ssl_aim_hidden).to(device)
    else:
        policy = policy.to(device)
    decoder = CSIReconstructionDecoder(
        in_ch=cfg.backbone_channels[-1], out_ch=C, out_len=S,
        mid_channels=cfg.ssl_decoder_channels,
    ).to(device)
    bt_projector = BTProjector(
        in_dim=cfg.backbone_channels[-1], proj_dim=cfg.ssl_bt_dim,
    ).to(device)

    # Backbone, decoder, AND projector share a single optimizer so that
    # rec_loss + wbt*bt_loss are combined into ONE backward / ONE step,
    # matching CIG-MAE Algorithm 1, line 21:
    #   Theta <- Optimizer(grad_Theta(L_rec + w_bt * L_BT))
    backbone_params = (
        list(model.backbone.parameters())
        + list(decoder.parameters())
        + list(bt_projector.parameters())
    )
    opt_backbone = torch.optim.AdamW(backbone_params, lr=cfg.ssl_lr, weight_decay=cfg.weight_decay)
    opt_policy = torch.optim.AdamW(policy.parameters(), lr=cfg.ssl_lr)

    loader = torch.utils.data.DataLoader(
        unlabeled_dataset, batch_size=cfg.ssl_batch_size, shuffle=True, drop_last=True
    )

    model.backbone.train()
    policy.train()
    decoder.train()
    bt_projector.train()

    for ep in range(cfg.ssl_epochs):
        tot_rec, tot_bt, tot_aim, n = 0.0, 0.0, 0.0, 0
        for X, _y in loader:                      # activity label discarded -- self-supervised
            X = X.to(device)                       # (B, C, T, S)
            B = X.shape[0]

            patches, n_p, S_trim = patchify_subcarrier(X, patch_size)     # (B,T,n_p,C*ps)
            patches_flat = patches.view(B * T, n_p, -1)

            # ---- 1) sample TWO independent visible/masked partitions ----
            # View 1 (used for reconstruction AND consistency)
            with torch.no_grad():
                probs, _ = policy(patches_flat)
            vis_mask_v1 = gumbel_topk_visible_mask(probs, n_visible)      # (B*T, n_p)
            visible_full_v1 = expand_patch_mask(vis_mask_v1, B, T, n_p, patch_size, C, S, S_trim, device)
            X_masked_v1 = X * visible_full_v1

            # View 2 (independent Gumbel sample, same policy probs —
            # used ONLY for the BT consistency term)
            vis_mask_v2 = gumbel_topk_visible_mask(probs, n_visible)      # (B*T, n_p)
            visible_full_v2 = expand_patch_mask(vis_mask_v2, B, T, n_p, patch_size, C, S, S_trim, device)
            X_masked_v2 = X * visible_full_v2

            # ---- 2a) backbone + decoder: masked-region MAE (L1) on view 1 ----
            Z_v1 = model.backbone(X_masked_v1)                            # (B,T,256,S')
            Z_flat = Z_v1.reshape(B * T, Z_v1.shape[2], Z_v1.shape[3])
            recon = decoder(Z_flat).view(B, T, C, S).permute(0, 2, 1, 3)  # (B,C,T,S)

            hidden_mask = (1.0 - visible_full_v1).bool()
            if hidden_mask.any():
                rec_loss = F.l1_loss(recon[hidden_mask], X[hidden_mask])  # un-normalized MAE
            else:
                rec_loss = F.l1_loss(recon, X)

            # ---- 2b) Barlow Twins intra-stream consistency (view 1 vs view 2) ----
            # Encode view 2 through backbone, GAP over (T, S') -> (B, 256)
            Z_v2 = model.backbone(X_masked_v2)                            # (B,T,256,S')
            z1_pool = Z_v1.mean(dim=(1, 3))                               # (B, 256)
            z2_pool = Z_v2.mean(dim=(1, 3))                               # (B, 256)

            # Project + EXPLICIT feature-wise batch normalization.  The
            # BTProjector's final bn_out z-scores each feature dimension
            # (mean=0, std=1) across the batch BEFORE the cross-correlation.
            # This is the actual mechanism that prevents representation
            # collapse without negative samples (Zbontar et al. Sec. 3).
            p1 = bt_projector(z1_pool)                                    # (B, bt_dim), z-scored
            p2 = bt_projector(z2_pool)                                    # (B, bt_dim), z-scored

            # Cross-correlation matrix (bt_dim x bt_dim)
            corr = (p1.T @ p2) / B                                       # (bt_dim, bt_dim)

            # BT loss: invariance (diagonal -> 1) + redundancy reduction
            on_diag = (1.0 - corr.diagonal()).pow(2).sum()
            off_diag = off_diagonal(corr).pow(2).sum()
            bt_loss = on_diag + cfg.ssl_bt_lambda * off_diag

            # ---- 2c) SINGLE combined backward + step (CIG-MAE Alg. 1, line 21) ----
            total_backbone_loss = rec_loss + cfg.ssl_wbt * bt_loss
            opt_backbone.zero_grad()
            total_backbone_loss.backward()
            opt_backbone.step()

            # ---- 3) policy update: REINFORCE-style, Eq. 4 of CIG-MAE ----
            with torch.no_grad():
                per_pixel_err = (recon.detach() - X).abs()                # (B,C,T,S)
                per_pixel_err = per_pixel_err[..., :S_trim]
                per_pixel_err = per_pixel_err.view(B, C, T, n_p, patch_size)
                per_patch_err = per_pixel_err.mean(dim=(1, 4))            # (B,T,n_p)
                per_patch_err = per_patch_err.reshape(B * T, n_p)

            probs2, _ = policy(patches_flat)
            hidden_flat = ~vis_mask_v1                                    # (B*T, n_p) True=masked
            if hidden_flat.any():
                # NOTE: probs2[hidden_flat] is p_visible evaluated at the
                # patches that landed in the MASKED set.  The REINFORCE
                # gradient -log(p_visible) * error pushes p_visible UP for
                # high-error patches, making them more likely to be kept
                # VISIBLE next round — which is the intended AIM behaviour
                # (CIG-MAE Eq. 4).
                log_p_vis_of_masked = torch.log(probs2[hidden_flat] + 1e-9)
                reward = per_patch_err[hidden_flat]                       # stop-gradient reward
                aim_loss = -(log_p_vis_of_masked * reward).mean()
            else:
                aim_loss = torch.zeros((), device=device)

            opt_policy.zero_grad()
            (cfg.ssl_waim * aim_loss).backward()
            opt_policy.step()

            tot_rec += rec_loss.item() * B
            tot_bt += bt_loss.item() * B
            tot_aim += aim_loss.item() * B
            n += B

        if verbose and (ep + 1) % max(1, cfg.ssl_epochs // 10) == 0:
            print(f"  [CIG-MAE pretrain] epoch {ep+1}/{cfg.ssl_epochs}  "
                  f"rec_loss(MAE)={tot_rec/n:.4f}  "
                  f"bt_loss={tot_bt/n:.4f}  aim_loss={tot_aim/n:.4f}")

    model.freeze_backbone()   # identical effect to the end of train_initial_domain()
    return model, policy, decoder, bt_projector


# --------------------------------------------------------------------------- #
# Stub: where dual-stream (amplitude + phase) + Barlow Twins would plug in,
# IF you extend preprocess_mmfi.py / fast_preprocess.py to also export phase.
# --------------------------------------------------------------------------- #
def pretrain_backbone_cigmae_dualstream(model_amp, model_phase, cfg, device,
                                         unlabeled_amp_ds, unlabeled_phase_ds,
                                         bt_dim=1024, bt_lambda=0.005, wbt=0.2):
    """
    NOT WIRED INTO main.py -- reference sketch only.

    To restore CIG-MAE's full dual-stream design once phase data is
    available, instantiate a SECOND `SharedBackbone` (e.g. `model_phase =
    SAMoEC(cfg).to(device)` used only for its `.backbone`), run
    `pretrain_backbone_cigmae` independently on each stream to get the two
    masked-reconstruction losses, and additionally add a Barlow Twins term
    computed on UNMASKED inputs (CIG-MAE Eq. 9-11):

        p_A = bt_projector_A(model_amp.backbone(X_A).mean(dim=(1,3)))   # GAP over T,S'
        p_P = bt_projector_P(model_phase.backbone(X_P).mean(dim=(1,3)))
        p_A, p_P = batch_norm(p_A), batch_norm(p_P)
        Corr = (p_A.T @ p_P) / B
        L_BT = (1 - diag(Corr)).pow(2).sum() + bt_lambda * off_diag(Corr).pow(2).sum()

    and add `wbt * L_BT` to the backbone optimizer step in both streams
    (paralleling `pretrain_backbone_cigmae`'s `opt_backbone.step()`).
    This is left unimplemented here since `fast_preprocess.py` currently
    discards phase entirely.
    """
    raise NotImplementedError(
        "Dual-stream pretraining requires phase-preserving preprocessing; "
        "see the docstring above for how to wire it in once available."
    )
