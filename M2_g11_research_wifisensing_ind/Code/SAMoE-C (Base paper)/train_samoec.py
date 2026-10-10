"""
Decoupled Continual Training Protocol (Sec. IV of the paper).

ENHANCED with:
  - SSL-anchored knowledge distillation during specialist training (Change 3)
  - SSL prototype computation for SSLInformedRouter (Change 2)

Phase 1 - Initial Domain Training (Sec. IV-A, Eq. 9):
    Train backbone + specialist-1 jointly on D1, then freeze the backbone.

Phase 2 - Incremental Learning, for each new domain s (Sec. IV-B/C):
    (a) Specialist Training (Eq. 10): train a fresh specialist E_s only,
        backbone and older specialists frozen. NOW with SSL-anchored
        distillation loss to keep specialist representations aligned with
        the SSL backbone's cross-domain view.
    (b) Router Update (Eq. 11): freeze E_s, then train ONLY the router on
        D_s (labelled with domain id s) mixed with the tiny replay buffer.
    (c) Refresh the buffer with a rho-fraction balanced sample of D_s.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

class SubjectDatasetWrapper(torch.utils.data.Dataset):
    """Wraps a dataset to also return subject IDs (if available) to avoid breaking standard loaders."""
    def __init__(self, subset):
        self.subset = subset
        if hasattr(subset, 'dataset') and hasattr(subset.dataset, 'subj_y'):
            self.subj_y = subset.dataset.subj_y[subset.indices]
        elif hasattr(subset, 'subj_y'):
            self.subj_y = subset.subj_y
        else:
            self.subj_y = torch.zeros(len(subset), dtype=torch.long)
            
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        X, y = self.subset[idx]
        return X, y, self.subj_y[idx]


from utils_samoec import accuracy


def train_initial_domain(model, cfg, device, dataset, epochs=None, verbose=True):
    """Phase 1: joint backbone + specialist-0 training on D1 (Eq. 9) with DANN."""
    epochs = epochs or cfg.epochs_initial
    idx0 = model.add_specialist()  # specialist 0
    assert idx0 == 0

    loader = DataLoader(SubjectDatasetWrapper(dataset), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    # Enable subj_discriminator to be trained
    params = model.trainable_params(["backbone", "specialist_0", "subj_discriminator"])
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()

    model.train()
    total_steps = epochs * len(loader)
    for ep in range(epochs):
        tot_loss, tot_acc, n = 0.0, 0.0, 0
        for batch_idx, (X, y, subj_y) in enumerate(loader):
            X, y, subj_y = X.to(device), y.to(device), subj_y.to(device)
            
            # GRL lambda scheduling
            p = float(ep * len(loader) + batch_idx) / total_steps
            lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
            
            # Main classification pass
            logits, _, _ = model(X, mode="force", specialist_idx=0)
            ce_loss = ce(logits, y)
            
            # Adversarial pass to scrub subject identity (Optional)
            if getattr(cfg, 'use_dann', True):
                subj_logits = model.forward_adversarial(X, lambda_=lambda_)
                adv_loss = ce(subj_logits, subj_y)
                loss = ce_loss + adv_loss
            else:
                loss = ce_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            tot_loss += loss.item() * X.size(0)
            tot_acc += accuracy(logits, y) * X.size(0)
            n += X.size(0)
        if verbose and (ep + 1) % 5 == 0:
            print(f"  [Init D1] epoch {ep+1}/{epochs}  loss={tot_loss/n:.4f}  acc={tot_acc/n:.2f}%")

    model.freeze_backbone()   # backbone (+ attention) frozen forever after this
    return model


def train_specialist(model, cfg, device, dataset, specialist_idx,
                     epochs=None, verbose=True):
    """Phase 2a: train only the new specialist E_s (Eq. 10)."""
    epochs = epochs or cfg.epochs_specialist
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    params = model.trainable_params([f"specialist_{specialist_idx}"])
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()

    model.train()
    # make sure gradients are enabled ONLY for the new specialist
    for p in model.parameters():
        p.requires_grad = False
    for p in model.specialists[specialist_idx].parameters():
        p.requires_grad = True

    for ep in range(epochs):
        tot_loss, tot_acc, n = 0.0, 0.0, 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits, _, _ = model(X, mode="force", specialist_idx=specialist_idx)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * X.size(0)
            tot_acc += accuracy(logits, y) * X.size(0)
            n += X.size(0)
        if verbose and (ep + 1) % 5 == 0:
            print(f"  [Specialist {specialist_idx}] epoch {ep+1}/{epochs}  "
                  f"loss={tot_loss/n:.4f}  acc={tot_acc/n:.2f}%")

    model.freeze_specialist(specialist_idx)
    return model


def train_specialist_with_ssl_anchor(model, cfg, device, dataset, specialist_idx,
                                      epochs=None, verbose=True):
    """
    Phase 2a ENHANCED: jointly train ADAPTER + SPECIALIST with SSL-anchored
    knowledge distillation (NOVEL — not in SAMoE-C or CIG-MAE).

    The adapter provides domain-specific feature refinement while the
    backbone stays permanently frozen. In addition to the standard
    cross-entropy loss, we add a distillation loss that encourages the
    specialist's intermediate features to stay aligned with the frozen
    backbone's cross-domain representations.

    L_total = L_CE + alpha_kd * L_KD
    where L_KD = MSE(specialist_features, stop_grad(backbone_features))
    """
    epochs = epochs or cfg.epochs_specialist
    alpha_kd = cfg.alpha_kd
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    # Train BOTH adapter and specialist jointly
    adapter_params = list(model.adapters[specialist_idx].parameters())
    specialist_params = list(model.specialists[specialist_idx].parameters())
    params = adapter_params + specialist_params
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()

    # Projection layer to align specialist feature dim -> backbone feature dim
    spec_feat_dim = cfg.gru_hidden * 2
    bb_feat_dim = cfg.backbone_channels[-1]
    kd_projector = nn.Linear(spec_feat_dim, bb_feat_dim).to(device)
    opt_proj = torch.optim.AdamW(kd_projector.parameters(), lr=cfg.lr)

    model.train()
    kd_projector.train()

    # Enable gradients ONLY for the adapter + specialist of this domain
    for p in model.parameters():
        p.requires_grad = False
    for p in model.adapters[specialist_idx].parameters():
        p.requires_grad = True
    for p in model.specialists[specialist_idx].parameters():
        p.requires_grad = True

    for ep in range(epochs):
        tot_loss, tot_ce, tot_kd, tot_acc, n = 0.0, 0.0, 0.0, 0.0, 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            # Forward: backbone(frozen) -> adapter(trainable) -> specialist(trainable)
            logits, _, _ = model(X, mode="force", specialist_idx=specialist_idx)
            ce_loss = ce(logits, y)

            # SSL-anchored distillation: specialist features vs RAW backbone features
            # Raw backbone provides cross-domain anchor (no adapter applied)
            Z = model.backbone(X)  # (B, T, 256, S') — frozen raw features
            Z_adapted = model.adapters[specialist_idx](Z)  # domain-specific
            spec_features = model.specialists[specialist_idx].get_intermediate_features(Z_adapted)
            with torch.no_grad():
                bb_features = Z.mean(dim=(1, 3))  # (B, 256) — raw backbone anchor

            spec_proj = kd_projector(spec_features)
            kd_loss = F.mse_loss(spec_proj, bb_features.detach())

            total_loss = ce_loss + alpha_kd * kd_loss

            opt.zero_grad()
            opt_proj.zero_grad()
            total_loss.backward()
            opt.step()
            opt_proj.step()

            tot_loss += total_loss.item() * X.size(0)
            tot_ce += ce_loss.item() * X.size(0)
            tot_kd += kd_loss.item() * X.size(0)
            tot_acc += accuracy(logits, y) * X.size(0)
            n += X.size(0)

        if verbose and (ep + 1) % 5 == 0:
            print(f"  [Specialist {specialist_idx} + SSL-KD] epoch {ep+1}/{epochs}  "
                  f"ce={tot_ce/n:.4f}  kd={tot_kd/n:.4f}  acc={tot_acc/n:.2f}%")

    model.freeze_specialist(specialist_idx)
    return model


def compute_ssl_prototype(model, device, dataset, cfg):
    """
    Compute the mean backbone representation for a domain's data.
    Used to register SSL prototypes in the SSLInformedRouter (Change 2).

    Returns: (backbone_channels[-1],) tensor — the domain prototype.
    """
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    model.eval()
    total_feat = None
    n = 0
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            Z = model.backbone(X)              # (B, T, 256, S')
            feat = Z.mean(dim=(1, 3))          # (B, 256) — GAP over T, S'
            if total_feat is None:
                total_feat = feat.sum(dim=0)
            else:
                total_feat += feat.sum(dim=0)
            n += X.size(0)
    return total_feat / n


def update_router(model, cfg, device, X_mix, y_mix, epochs=None, verbose=True):
    """Phase 2b: calibrate the semantic router on D_s + replay buffer (Eq. 11)."""
    epochs = epochs or cfg.epochs_router
    for p in model.parameters():
        p.requires_grad = False
    for p in model.router.parameters():
        p.requires_grad = True
    for p in model.attn.parameters():
        p.requires_grad = True

    params = list(model.router.parameters()) + list(model.attn.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(X_mix, y_mix)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model.train()
    for ep in range(epochs):
        tot_loss, tot_acc, n = 0.0, 0.0, 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            _, c, _ = model.shared_forward(X)
            # For SSLInformedRouter, also pass SSL features
            Z = model.backbone(X)
            ssl_features = Z.mean(dim=(1, 3))
            logits = model.route_logits(c, ssl_features)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * X.size(0)
            tot_acc += accuracy(logits, y) * X.size(0)
            n += X.size(0)
        if verbose and (ep + 1) % 5 == 0:
            print(f"  [Router update] epoch {ep+1}/{epochs}  "
                  f"loss={tot_loss/n:.4f}  routing_acc={tot_acc/n:.2f}%")

    return model


@torch.no_grad()
@torch.no_grad()
def evaluate_specialist(model, device, dataset, specialist_idx, cfg, print_subject_breakdown=False):
    """Per-domain accuracy using the ground-truth specialist (Table II columns)."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    model.eval()
    tot_acc, n = 0.0, 0
    
    # Optional subject breakdown tracking
    do_subj_breakdown = print_subject_breakdown and hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'subj_y')
    if do_subj_breakdown:
        subj_y = dataset.dataset.subj_y[dataset.indices]
        subj_correct = {}
        subj_total = {}
        idx_offset = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits, _, _ = model(X, mode="force", specialist_idx=specialist_idx)
        preds = logits.argmax(dim=-1)
        
        # Aggregate global
        correct = (preds == y).sum().item()
        tot_acc += correct
        n += X.size(0)
        
        if do_subj_breakdown:
            batch_sz = X.size(0)
            batch_subjs = subj_y[idx_offset : idx_offset + batch_sz]
            for p, ty, s in zip(preds, y, batch_subjs):
                s = s.item()
                subj_total[s] = subj_total.get(s, 0) + 1
                if p == ty:
                    subj_correct[s] = subj_correct.get(s, 0) + 1
            idx_offset += batch_sz

    if do_subj_breakdown:
        print("\n    --- Per-Subject Val Acc ---")
        with open("diagnostic_results.md", "a") as f:
            f.write("\n### Per-Subject Breakdown\n")
            f.write("| Subject_ID | Samples | Accuracy(%) |\n")
            f.write("|---|---|---|\n")
            for sid in sorted(subj_total.keys()):
                sacc = 100. * subj_correct.get(sid, 0) / subj_total[sid]
                print(f"    S{sid+1:02d}: {sacc:.2f}% ({subj_total[sid]} samples)")
                f.write(f"| S{sid+1:02d} | {subj_total[sid]} | {sacc:.2f}% |\n")
        print("    ---------------------------\n")

    return 100.0 * tot_acc / n if n > 0 else 0


@torch.no_grad()
def evaluate_full_pipeline(model, device, dataset, cfg, mode="route"):
    """
    Full end-to-end accuracy (router decides specialist), used for Table III's
    "Final Avg. HAR Acc." over a balanced multi-domain test set.
    `mode="route"` -> SAMoE-C behaviour; `mode="all"` -> MEMO behaviour.
    """
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    model.eval()
    tot_acc, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits, _, _ = model(X, mode=mode)
        tot_acc += accuracy(logits, y) * X.size(0)
        n += X.size(0)
    return tot_acc / n


@torch.no_grad()
def evaluate_router_accuracy(model, device, dataset, domain_idx, cfg):
    """Domain-discrimination accuracy of the router alone (Fig. 3)."""
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    model.eval()
    correct, n = 0, 0
    for X, y in loader:
        X = X.to(device)
        _, c, _ = model.shared_forward(X)
        Z = model.backbone(X)
        ssl_features = Z.mean(dim=(1, 3))
        logits = model.route_logits(c, ssl_features)
        pred = logits.argmax(dim=-1)
        correct += (pred == domain_idx).sum().item()
        n += X.size(0)
    return 100.0 * correct / n
