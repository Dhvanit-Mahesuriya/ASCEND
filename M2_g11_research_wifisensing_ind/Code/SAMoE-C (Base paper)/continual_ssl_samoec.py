"""
continual_ssl.py
================

Domain-Adaptive Continual Self-Supervised Learning with EWC Regularization.

This is the CORE NOVELTY of the proposed framework. Neither SAMoE-C nor
CIG-MAE addresses the intersection of SSL pretraining and domain-incremental
continual learning:

  - SAMoE-C freezes the backbone after D1 — missing domain-specific features
    for D2, D3, D4.
  - CIG-MAE provides powerful SSL but has no continual learning mechanism.

This module introduces:
  1. Fisher Information Matrix (FIM) computation on SSL objectives
  2. EWC-style regularization to prevent backbone representational drift
  3. Lightweight SSL adaptation step when each new domain arrives

The backbone is NO LONGER permanently frozen after D1. Instead, it undergoes
a short, regularized SSL fine-tuning on each new domain's unlabeled data,
producing domain-adapted representations while preserving knowledge from
all previously seen domains.

References:
  - Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
    networks", PNAS 2017 (EWC)
  - Schwarz et al., "Progress & Compress", ICML 2018 (online EWC variant)
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinualSSLRegularizer:
    """
    Elastic Weight Consolidation (EWC) regularizer operating on SSL
    objectives (reconstruction loss), NOT on supervised classification loss.

    After each domain's SSL adaptation, we:
      1. Compute the diagonal Fisher Information Matrix (FIM) of the backbone
         parameters w.r.t. the SSL reconstruction loss.
      2. Store a snapshot of the backbone parameters.

    During subsequent SSL adaptations, the EWC penalty:
        L_ewc = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
    prevents the backbone from drifting too far from its consolidated state.

    We use the ONLINE EWC variant (Schwarz et al., 2018): the Fisher from
    each domain is accumulated into a running diagonal, avoiding storage of
    per-domain Fishers.
    """
    def __init__(self, lambda_ewc: float = 5000.0):
        self.lambda_ewc = lambda_ewc
        self.fisher_running = {}    # param_name -> running diagonal Fisher
        self.consolidated_params = {}  # param_name -> theta*
        self.n_domains_seen = 0

    @torch.no_grad()
    def _zero_fisher(self, model_backbone):
        """Initialize a zero Fisher dict matching the backbone's parameters."""
        fisher = {}
        for name, p in model_backbone.named_parameters():
            if p.requires_grad:
                fisher[name] = torch.zeros_like(p)
        return fisher

    def compute_fisher(self, model_backbone, decoder, dataloader, device,
                       n_samples=None, patch_fn=None, mask_fn=None,
                       policy=None, cfg=None):
        """
        Compute the diagonal Fisher Information Matrix for the backbone
        parameters w.r.t. the SSL reconstruction loss (NOT supervised loss).

        This uses the empirical Fisher: F_i = E[(d log p / d theta_i)^2],
        approximated by averaging squared gradients of the reconstruction
        loss over the dataset.

        Args:
            model_backbone: The shared backbone module.
            decoder: The SSL reconstruction decoder.
            dataloader: DataLoader for the current domain (unlabeled).
            device: torch device.
            n_samples: Max samples for Fisher estimation (None = all).
            patch_fn: Function to patchify CSI (from ssl_pretrain_samoec).
            mask_fn: Function to generate masks (from ssl_pretrain_samoec).
            policy: AIM policy network (for generating masks).
            cfg: Config object.
        """
        from ssl_pretrain_samoec import patchify_subcarrier, gumbel_topk_visible_mask, expand_patch_mask

        fisher = self._zero_fisher(model_backbone)
        model_backbone.train()
        decoder.train()

        count = 0
        C, S, T = cfg.C, cfg.S, cfg.T
        patch_size = cfg.ssl_patch_size
        n_patches = S // patch_size
        n_visible = max(1, int(round((1.0 - cfg.ssl_mask_ratio) * n_patches)))

        for X, _ in dataloader:
            X = X.to(device)
            B = X.shape[0]

            patches, n_p, S_trim = patchify_subcarrier(X, patch_size)
            patches_flat = patches.view(B * T, n_p, -1)

            with torch.no_grad():
                probs, _ = policy(patches_flat)
            vis_mask = gumbel_topk_visible_mask(probs, n_visible)
            visible_full = expand_patch_mask(vis_mask, B, T, n_p, patch_size, C, S, S_trim, device)
            X_masked = X * visible_full

            # Forward through backbone + decoder
            Z = model_backbone(X_masked)
            Z_flat = Z.reshape(B * T, Z.shape[2], Z.shape[3])
            recon = decoder(Z_flat).view(B, T, C, S).permute(0, 2, 1, 3)

            hidden_mask = (1.0 - visible_full).bool()
            if hidden_mask.any():
                rec_loss = F.l1_loss(recon[hidden_mask], X[hidden_mask])
            else:
                rec_loss = F.l1_loss(recon, X)

            model_backbone.zero_grad()
            rec_loss.backward()

            for name, p in model_backbone.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += (p.grad.data ** 2) * B

            count += B
            if n_samples is not None and count >= n_samples:
                break

        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= max(count, 1)

        return fisher

    def consolidate(self, model_backbone, fisher_new, gamma=0.95):
        """
        Online EWC consolidation (Schwarz et al., 2018):
            F_running = gamma * F_running + (1-gamma) * F_new
            theta* = current backbone parameters

        gamma controls how much we weight old vs. new Fisher information.
        """
        for name, p in model_backbone.named_parameters():
            if name in fisher_new:
                if name in self.fisher_running:
                    self.fisher_running[name] = (
                        gamma * self.fisher_running[name].to(p.device)
                        + (1 - gamma) * fisher_new[name].to(p.device)
                    )
                else:
                    self.fisher_running[name] = fisher_new[name].to(p.device)
                self.consolidated_params[name] = p.data.clone()

        self.n_domains_seen += 1

    def penalty(self, model_backbone):
        """
        Compute the EWC penalty:
            L_ewc = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

        Returns 0 if no domains have been consolidated yet (first domain).
        """
        if self.n_domains_seen == 0:
            return torch.tensor(0.0, device=next(model_backbone.parameters()).device)

        loss = torch.tensor(0.0, device=next(model_backbone.parameters()).device)
        for name, p in model_backbone.named_parameters():
            if name in self.fisher_running and name in self.consolidated_params:
                fisher = self.fisher_running[name].to(p.device)
                theta_star = self.consolidated_params[name].to(p.device)
                loss += (fisher * (p - theta_star) ** 2).sum()

        return (self.lambda_ewc / 2.0) * loss


def continual_ssl_adapt(model, cfg, device, domain_dataset, domain_idx,
                        policy, decoder, bt_projector, ewc_reg,
                        verbose=True):
    """
    Perform a lightweight SSL fine-tuning of the backbone on a NEW domain,
    regularized by the accumulated EWC penalty from all previous domains.

    This is the key difference from the original SAMoE-C protocol:
      - Original: backbone frozen forever after D1
      - Ours: backbone undergoes short, regularized SSL adaptation per domain

    The adaptation uses fewer epochs than initial pretraining (controlled by
    cfg.ssl_adapt_epochs) and a lower learning rate (cfg.ssl_adapt_lr).

    Args:
        model: SAMoEC model (backbone will be temporarily unfrozen).
        cfg: Config object.
        device: torch device.
        domain_dataset: Dataset for the new domain (labels discarded).
        domain_idx: Index of the current domain (for domain-conditioned masking).
        policy: AIM policy network.
        decoder: SSL reconstruction decoder.
        bt_projector: Barlow Twins projector.
        ewc_reg: ContinualSSLRegularizer instance.
        verbose: Print progress.

    Returns:
        Updated (model, policy, decoder, bt_projector, ewc_reg).
    """
    from ssl_pretrain_samoec import (patchify_subcarrier, gumbel_topk_visible_mask,
                               expand_patch_mask, off_diagonal)

    C, S, T = cfg.C, cfg.S, cfg.T
    patch_size = cfg.ssl_patch_size
    n_patches = S // patch_size
    n_visible = max(1, int(round((1.0 - cfg.ssl_mask_ratio) * n_patches)))

    if verbose:
        print(f"  [Continual SSL] Adapting backbone to domain {domain_idx} "
              f"({cfg.ssl_adapt_epochs} epochs, lr={cfg.ssl_adapt_lr:.1e}, "
              f"lambda_ewc={cfg.lambda_ewc:.0f})")

    # Temporarily unfreeze backbone for adaptation
    for p in model.backbone.parameters():
        p.requires_grad = True

    backbone_params = (
        list(model.backbone.parameters())
        + list(decoder.parameters())
        + list(bt_projector.parameters())
    )
    opt_backbone = torch.optim.AdamW(
        backbone_params, lr=cfg.ssl_adapt_lr, weight_decay=cfg.weight_decay
    )
    opt_policy = torch.optim.AdamW(policy.parameters(), lr=cfg.ssl_adapt_lr)

    loader = torch.utils.data.DataLoader(
        domain_dataset, batch_size=cfg.ssl_batch_size, shuffle=True, drop_last=True
    )

    model.backbone.train()
    policy.train()
    decoder.train()
    bt_projector.train()

    # Build domain index tensor for domain-conditioned masking
    domain_tensor = torch.full((cfg.ssl_batch_size,), domain_idx,
                                dtype=torch.long, device=device)

    for ep in range(cfg.ssl_adapt_epochs):
        tot_rec, tot_bt, tot_ewc, tot_aim, n = 0.0, 0.0, 0.0, 0.0, 0
        for X, _y in loader:
            X = X.to(device)
            B = X.shape[0]

            patches, n_p, S_trim = patchify_subcarrier(X, patch_size)
            patches_flat = patches.view(B * T, n_p, -1)

            # Use domain-conditioned masking if policy supports it
            with torch.no_grad():
                if hasattr(policy, 'forward_conditioned'):
                    d_tensor = domain_tensor[:B].unsqueeze(1).expand(-1, T).reshape(B * T)
                    probs, _ = policy.forward_conditioned(patches_flat, d_tensor)
                else:
                    probs, _ = policy(patches_flat)

            # View 1 for reconstruction
            vis_mask_v1 = gumbel_topk_visible_mask(probs, n_visible)
            visible_full_v1 = expand_patch_mask(vis_mask_v1, B, T, n_p, patch_size, C, S, S_trim, device)
            X_masked_v1 = X * visible_full_v1

            # View 2 for BT consistency
            vis_mask_v2 = gumbel_topk_visible_mask(probs, n_visible)
            visible_full_v2 = expand_patch_mask(vis_mask_v2, B, T, n_p, patch_size, C, S, S_trim, device)
            X_masked_v2 = X * visible_full_v2

            # MAE reconstruction on view 1
            Z_v1 = model.backbone(X_masked_v1)
            Z_flat = Z_v1.reshape(B * T, Z_v1.shape[2], Z_v1.shape[3])
            recon = decoder(Z_flat).view(B, T, C, S).permute(0, 2, 1, 3)

            hidden_mask = (1.0 - visible_full_v1).bool()
            if hidden_mask.any():
                rec_loss = F.l1_loss(recon[hidden_mask], X[hidden_mask])
            else:
                rec_loss = F.l1_loss(recon, X)

            # BT intra-stream consistency
            Z_v2 = model.backbone(X_masked_v2)
            z1_pool = Z_v1.mean(dim=(1, 3))
            z2_pool = Z_v2.mean(dim=(1, 3))
            p1 = bt_projector(z1_pool)
            p2 = bt_projector(z2_pool)
            corr = (p1.T @ p2) / B
            on_diag = (1.0 - corr.diagonal()).pow(2).sum()
            off_diag = off_diagonal(corr).pow(2).sum()
            bt_loss = on_diag + cfg.ssl_bt_lambda * off_diag

            # EWC penalty (0 for first domain)
            ewc_loss = ewc_reg.penalty(model.backbone)

            # Combined loss
            total_loss = rec_loss + cfg.ssl_wbt * bt_loss + ewc_loss
            opt_backbone.zero_grad()
            total_loss.backward()
            opt_backbone.step()

            # Policy update (REINFORCE)
            with torch.no_grad():
                per_pixel_err = (recon.detach() - X).abs()
                per_pixel_err = per_pixel_err[..., :S_trim]
                per_pixel_err = per_pixel_err.view(B, C, T, n_p, patch_size)
                per_patch_err = per_pixel_err.mean(dim=(1, 4))
                per_patch_err = per_patch_err.reshape(B * T, n_p)

            if hasattr(policy, 'forward_conditioned'):
                d_tensor = domain_tensor[:B].unsqueeze(1).expand(-1, T).reshape(B * T)
                probs2, _ = policy.forward_conditioned(patches_flat, d_tensor)
            else:
                probs2, _ = policy(patches_flat)

            hidden_flat = ~vis_mask_v1
            if hidden_flat.any():
                log_p = torch.log(probs2[hidden_flat] + 1e-9)
                reward = per_patch_err[hidden_flat]
                aim_loss = -(log_p * reward).mean()
            else:
                aim_loss = torch.zeros((), device=device)

            opt_policy.zero_grad()
            (cfg.ssl_waim * aim_loss).backward()
            opt_policy.step()

            tot_rec += rec_loss.item() * B
            tot_bt += bt_loss.item() * B
            tot_ewc += ewc_loss.item() * B
            tot_aim += aim_loss.item() * B
            n += B

        if verbose and (ep + 1) % max(1, cfg.ssl_adapt_epochs // 5) == 0:
            print(f"    epoch {ep+1}/{cfg.ssl_adapt_epochs}  "
                  f"rec={tot_rec/n:.4f}  bt={tot_bt/n:.4f}  "
                  f"ewc={tot_ewc/n:.4f}  aim={tot_aim/n:.4f}")

    # Compute Fisher on this domain's data and consolidate
    fisher_new = ewc_reg.compute_fisher(
        model.backbone, decoder, loader, device,
        policy=policy, cfg=cfg
    )
    ewc_reg.consolidate(model.backbone, fisher_new, gamma=cfg.ewc_gamma)

    # Re-freeze backbone
    model.freeze_backbone()

    return model, policy, decoder, bt_projector, ewc_reg


def adapter_ssl_adapt(model, cfg, device, domain_dataset, adapter_idx,
                      policy, decoder, bt_projector, verbose=True):
    """
    SSL-train a domain-specific ADAPTER using CIG-MAE (NOVEL).

    Unlike continual_ssl_adapt which modifies the backbone (causing drift),
    this function keeps the backbone PERMANENTLY FROZEN and trains only the
    lightweight adapter module. The adapter learns to refine frozen backbone
    features for the new domain's data distribution.

    This preserves Change 1 (Continual SSL) while eliminating feature space
    drift entirely. No EWC needed since the backbone never changes.

    Args:
        model: SAMoEC model (backbone frozen, adapter at adapter_idx trainable).
        cfg: Config object.
        device: torch device.
        domain_dataset: Dataset for the new domain (labels discarded).
        adapter_idx: Index of the adapter to train.
        policy: AIM policy network.
        decoder: SSL reconstruction decoder.
        bt_projector: Barlow Twins projector.
        verbose: Print progress.

    Returns:
        Updated (model, policy, decoder, bt_projector).
    """
    from ssl_pretrain_samoec import (patchify_subcarrier, gumbel_topk_visible_mask,
                               expand_patch_mask, off_diagonal)

    C, S, T = cfg.C, cfg.S, cfg.T
    patch_size = cfg.ssl_patch_size
    n_patches = S // patch_size
    n_visible = max(1, int(round((1.0 - cfg.ssl_mask_ratio) * n_patches)))

    if verbose:
        print(f"  [Adapter SSL] Training adapter {adapter_idx} via CIG-MAE "
              f"({cfg.ssl_adapt_epochs} epochs, lr={cfg.ssl_adapt_lr:.1e})")

    adapter = model.adapters[adapter_idx]

    # Backbone stays FROZEN throughout — only adapter + decoder + projector train
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in adapter.parameters():
        p.requires_grad = True

    adapt_params = (
        list(adapter.parameters())
        + list(decoder.parameters())
        + list(bt_projector.parameters())
    )
    opt_adapt = torch.optim.AdamW(adapt_params, lr=cfg.ssl_adapt_lr, weight_decay=cfg.weight_decay)
    opt_policy = torch.optim.AdamW(policy.parameters(), lr=cfg.ssl_adapt_lr)

    loader = torch.utils.data.DataLoader(
        domain_dataset, batch_size=cfg.ssl_batch_size, shuffle=True, drop_last=True
    )

    adapter.train()
    policy.train()
    decoder.train()
    bt_projector.train()

    domain_tensor = torch.full((cfg.ssl_batch_size,), adapter_idx,
                                dtype=torch.long, device=device)

    for ep in range(cfg.ssl_adapt_epochs):
        tot_rec, tot_bt, tot_aim, n = 0.0, 0.0, 0.0, 0
        for X, _y in loader:
            X = X.to(device)
            B = X.shape[0]

            patches, n_p, S_trim = patchify_subcarrier(X, patch_size)
            patches_flat = patches.view(B * T, n_p, -1)

            # Domain-conditioned masking
            with torch.no_grad():
                if hasattr(policy, 'forward_conditioned'):
                    d_tensor = domain_tensor[:B].unsqueeze(1).expand(-1, T).reshape(B * T)
                    probs, _ = policy.forward_conditioned(patches_flat, d_tensor)
                else:
                    probs, _ = policy(patches_flat)

            # View 1 for reconstruction
            vis_mask_v1 = gumbel_topk_visible_mask(probs, n_visible)
            visible_full_v1 = expand_patch_mask(vis_mask_v1, B, T, n_p, patch_size, C, S, S_trim, device)
            X_masked_v1 = X * visible_full_v1

            # View 2 for BT consistency
            vis_mask_v2 = gumbel_topk_visible_mask(probs, n_visible)
            visible_full_v2 = expand_patch_mask(vis_mask_v2, B, T, n_p, patch_size, C, S, S_trim, device)
            X_masked_v2 = X * visible_full_v2

            # Backbone (FROZEN) → Adapter (TRAINABLE) → Decoder
            with torch.no_grad():
                Z_v1_raw = model.backbone(X_masked_v1)  # (B,T,256,S') frozen
                Z_v2_raw = model.backbone(X_masked_v2)  # frozen

            Z_v1 = adapter(Z_v1_raw)  # adapter is trainable
            Z_flat = Z_v1.reshape(B * T, Z_v1.shape[2], Z_v1.shape[3])
            recon = decoder(Z_flat).view(B, T, C, S).permute(0, 2, 1, 3)

            hidden_mask = (1.0 - visible_full_v1).bool()
            if hidden_mask.any():
                rec_loss = F.l1_loss(recon[hidden_mask], X[hidden_mask])
            else:
                rec_loss = F.l1_loss(recon, X)

            # BT intra-stream consistency through adapter
            Z_v2 = adapter(Z_v2_raw)
            z1_pool = Z_v1.mean(dim=(1, 3))
            z2_pool = Z_v2.mean(dim=(1, 3))
            p1 = bt_projector(z1_pool)
            p2 = bt_projector(z2_pool)
            corr = (p1.T @ p2) / B
            on_diag = (1.0 - corr.diagonal()).pow(2).sum()
            off_diag = off_diagonal(corr).pow(2).sum()
            bt_loss = on_diag + cfg.ssl_bt_lambda * off_diag

            total_loss = rec_loss + cfg.ssl_wbt * bt_loss
            opt_adapt.zero_grad()
            total_loss.backward()
            opt_adapt.step()

            # Policy update (REINFORCE)
            with torch.no_grad():
                per_pixel_err = (recon.detach() - X).abs()
                per_pixel_err = per_pixel_err[..., :S_trim]
                per_pixel_err = per_pixel_err.view(B, C, T, n_p, patch_size)
                per_patch_err = per_pixel_err.mean(dim=(1, 4))
                per_patch_err = per_patch_err.reshape(B * T, n_p)

            if hasattr(policy, 'forward_conditioned'):
                d_tensor = domain_tensor[:B].unsqueeze(1).expand(-1, T).reshape(B * T)
                probs2, _ = policy.forward_conditioned(patches_flat, d_tensor)
            else:
                probs2, _ = policy(patches_flat)

            hidden_flat = ~vis_mask_v1
            if hidden_flat.any():
                log_p = torch.log(probs2[hidden_flat] + 1e-9)
                reward = per_patch_err[hidden_flat]
                aim_loss = -(log_p * reward).mean()
            else:
                aim_loss = torch.zeros((), device=device)

            opt_policy.zero_grad()
            (cfg.ssl_waim * aim_loss).backward()
            opt_policy.step()

            tot_rec += rec_loss.item() * B
            tot_bt += bt_loss.item() * B
            tot_aim += aim_loss.item() * B
            n += B

        if verbose and (ep + 1) % max(1, cfg.ssl_adapt_epochs // 5) == 0:
            print(f"    epoch {ep+1}/{cfg.ssl_adapt_epochs}  "
                  f"rec={tot_rec/n:.4f}  bt={tot_bt/n:.4f}  aim={tot_aim/n:.4f}")

    return model, policy, decoder, bt_projector
