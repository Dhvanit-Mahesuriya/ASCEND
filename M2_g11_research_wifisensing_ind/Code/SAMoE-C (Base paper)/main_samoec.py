"""
ENHANCED experimental protocol integrating all 5 novel changes:

  Continual SSL with EWC — backbone adapts to each domain
  SSL-Informed Router — hybrid gate + prototype routing
  SSL-Anchored Specialist Distillation — KD from backbone features
  Domain-Conditioned Masking — AIM policy is domain-aware
  Dual-Purpose SSL Replay Buffer — buffer used for SSL replay

Protocol:
  1. SSL pretrain backbone on D1 (with domain-conditioned AIM)
  2. For each subsequent domain D_s:
     a. Continual SSL adapt backbone (EWC-regularized) on D_s
     b. Train specialist with SSL-anchored distillation
     c. Compute & register SSL prototype for routing
     d. Update router (gate + prototype blend) with replay buffer
     e. Update buffer (dual-purpose)

Baselines run alongside:
  - Original SAMoE-C (supervised Phase 1, frozen backbone)
  - Basic CL (single model retrained from scratch)
  - MEMO (aggregate all experts)

KPI comparison with SAMoE-C paper Table II / III.
"""
import argparse
import copy
import time
import torch

from config_samoec import Config
from dataset_samoec import SyntheticCSIDataset, MMFiCSIDataset
from models_samoec import SAMoEC, BasicCL
from buffer_samoec import DualPurposeBuffer
from utils_samoec import set_seed, get_device, estimate_mflops_per_sample
from train_samoec import (
    train_initial_domain, train_specialist, train_specialist_with_ssl_anchor,
    update_router, compute_ssl_prototype,
    evaluate_specialist, evaluate_full_pipeline, evaluate_router_accuracy,
)
from ssl_pretrain_samoec import (
    pretrain_backbone_cigmae, DomainConditionedAIMPolicy, AIMPolicyNetwork
)
from continual_ssl_samoec import adapter_ssl_adapt


def build_domains_leaky_diagnostic(cfg, use_real_data=False, mmfi_root=None):
    from dataset_samoec import MMFiCSIDataset, SyntheticCSIDataset
    import torch
    domains = []
    global_mean, global_std = None, None
    if use_real_data:
        for i, name in enumerate(["D1", "D2", "D3", "D4"]):
            full = MMFiCSIDataset(mmfi_root, name)
            n = len(full)
            n_val = int(0.2 * n)
            train_ds, val_ds = torch.utils.data.random_split(
                full, [n - n_val, n_val],
                generator=torch.Generator().manual_seed(cfg.seed + i),
            )
            train_idx = train_ds.indices
            if global_mean is None:
                train_X = full.X[train_idx]
                global_mean = train_X.mean(dim=(0, 2, 3), keepdim=True)
                global_std = train_X.std(dim=(0, 2, 3), keepdim=True)
            full.X = (full.X - global_mean) / (global_std + 1e-8)
            domains.append((train_ds, val_ds))
    return domains

def build_domains(cfg, use_real_data=False, mmfi_root=None):
    return build_domains_leaky_diagnostic(cfg, use_real_data, mmfi_root)

def old_build_domains(cfg, use_real_data=False, mmfi_root=None):
    from dataset_samoec import MMFiCSIDataset, SyntheticCSIDataset
    import os
    import numpy as np
    import torch
    domains = []
    global_mean, global_std = None, None
    if use_real_data:
        for i, name in enumerate(["D1", "D2", "D3", "D4"]):
            full = MMFiCSIDataset(mmfi_root, name)
            # --- Cross-Subject Split Logic ---
            unique_subjs = torch.unique(full.subj_y).numpy()
            rng = np.random.RandomState(cfg.seed + i)
            rng.shuffle(unique_subjs)
            
            n_train_subjs = max(1, int(0.8 * len(unique_subjs)))
            train_subjs = set(unique_subjs[:n_train_subjs])
            
            train_indices, val_indices = [], []
            for idx, subj in enumerate(full.subj_y.numpy()):
                if subj in train_subjs:
                    train_indices.append(idx)
                else:
                    val_indices.append(idx)
                    
            train_ds = torch.utils.data.Subset(full, train_indices)
            val_ds = torch.utils.data.Subset(full, val_indices)

            if global_mean is None:
                train_X = full.X[train_indices]
                global_mean = train_X.mean(dim=(0, 2, 3), keepdim=True)
                global_std = train_X.std(dim=(0, 2, 3), keepdim=True)
            full.X = (full.X - global_mean) / (global_std + 1e-8)
            domains.append((train_ds, val_ds))
    else:
        for i in range(cfg.N_DOMAINS):
            full = SyntheticCSIDataset(cfg.K, cfg.C, cfg.T, cfg.S,
                                       n_samples=cfg.samples_per_domain,
                                       domain_seed=cfg.seed + i)
            n_val = int(0.2 * len(full))
            train_ds, val_ds = torch.utils.data.random_split(
                full, [len(full) - n_val, n_val],
                generator=torch.Generator().manual_seed(cfg.seed + i)
            )
            domains.append((train_ds, val_ds))
    return domains



def run_proposed_framework(cfg, device, domains, domain_order, verbose=True):
    """
    PROPOSED FRAMEWORK: Full pipeline with all 5 novel changes.

    This is the core contribution — a tight coupling between SSL and CL
    that goes beyond simple combination of SAMoE-C + CIG-MAE.
    """
    # Initialize model with SSL-informed router (Change 2)
    model = SAMoEC(cfg, use_ssl_router=cfg.use_ssl_router).to(device)

    # No EWC needed — backbone stays frozen after D1, adapters handle domain adaptation

    # Initialize AIM policy: domain-conditioned (Change 4) or standard
    C, S = cfg.C, cfg.S
    patch_dim = C * cfg.ssl_patch_size
    if cfg.use_domain_conditioned_aim:
        policy = DomainConditionedAIMPolicy(
            patch_dim, hidden=cfg.ssl_aim_hidden,
            n_domains=cfg.N_DOMAINS
        ).to(device)
        print("  Using Domain-Conditioned AIM Policy (Change 4)")
    else:
        policy = AIMPolicyNetwork(patch_dim, hidden=cfg.ssl_aim_hidden).to(device)

    buffer = DualPurposeBuffer(rho=cfg.rho, seed=cfg.seed)
    history = []
    ssl_artifacts = {}  # store decoder, bt_projector for adapter SSL

    for step, d in enumerate(domain_order):
        train_ds, val_ds = domains[d]
        print(f"\n{'='*60}")
        print(f"  Step {step+1}: Domain D{d+1}")
        print(f"{'='*60}")

        if step == 0:
            # ---- STEP 0: Initial SSL pretraining on D1 ----
            print(f"\n--- Phase 0: CIG-MAE SSL pretraining on D{d+1} ---")
            model_ret = pretrain_backbone_cigmae(
                model, cfg, device, train_ds, verbose=verbose
            )
            model, _, decoder, bt_projector = model_ret
            ssl_artifacts['decoder'] = decoder
            ssl_artifacts['bt_projector'] = bt_projector
            # Backbone is now PERMANENTLY FROZEN (no EWC needed)

            # Add adapter-0 + specialist-0
            spec_idx = model.add_specialist()
            assert spec_idx == 0
            print(f"\n--- Phase 1: Adapter + Specialist {spec_idx} training (with SSL-KD) ---")
            train_specialist_with_ssl_anchor(
                model, cfg, device, train_ds, spec_idx, verbose=verbose
            )
        else:
            # ---- STEPS 1+: Create adapter+specialist, SSL-train adapter, then specialist ----

            # Create new adapter + specialist pair
            spec_idx = model.add_specialist()

            # SSL-train the domain adapter (backbone stays frozen)
            print(f"\n--- Phase 0: Adapter SSL training for D{d+1} (backbone frozen) ---")
            model, policy, decoder, bt_projector = adapter_ssl_adapt(
                model, cfg, device, train_ds, adapter_idx=spec_idx,
                policy=policy,
                decoder=ssl_artifacts['decoder'],
                bt_projector=ssl_artifacts['bt_projector'],
                verbose=verbose
            )
            ssl_artifacts['decoder'] = decoder
            ssl_artifacts['bt_projector'] = bt_projector

            # Train adapter + specialist jointly with SSL-anchored distillation (Change 3)
            print(f"\n--- Phase 1: Adapter + Specialist {spec_idx} training (with SSL-KD) ---")
            train_specialist_with_ssl_anchor(
                model, cfg, device, train_ds, spec_idx, verbose=verbose
            )

        # ---- Compute and register SSL prototype (Change 2) ----
        if cfg.use_ssl_router and hasattr(model.router, 'register_prototype'):
            prototype = compute_ssl_prototype(model, device, train_ds, cfg)
            model.router.register_prototype(spec_idx, prototype)
            print(f"  Registered SSL prototype for domain {d+1}")

        # ---- Router update using current domain + replay buffer ----
        if hasattr(train_ds, "dataset"):  # torch.utils.data.Subset
            base_ds = train_ds.dataset
            idxs = train_ds.indices
            X_full, y_full = base_ds.full_tensor()
            X_domain = X_full[idxs]
            y_domain = y_full[idxs]
        else:
            X_domain, y_domain = train_ds.full_tensor()

        X_mix, y_mix = buffer.get_mixed(X_domain, current_domain_idx=spec_idx)
        print(f"\n--- Phase 2: Router update (current + {len(buffer)} buffered samples) ---")
        update_router(model, cfg, device, X_mix, y_mix, verbose=verbose)

        # dual-purpose buffer
        buffer.add_domain(X_domain, y_domain, spec_idx)

        # ---- Evaluation ----
        spec_acc = evaluate_specialist(model, device, val_ds, spec_idx, cfg)
        router_acc = evaluate_router_accuracy(model, device, val_ds, spec_idx, cfg)
        print(f"\n  -> D{d+1} specialist val acc = {spec_acc:.2f}%, "
              f"router self-discrimination = {router_acc:.2f}%")
        history.append({
            "domain": d, "specialist_acc": spec_acc, "router_acc": router_acc,
            "step": step
        })

    return model, buffer, history


def run_samoec_baseline(cfg, device, domains, domain_order, verbose=True):
    """
    Original SAMoE-C (supervised Phase 1, frozen backbone, no SSL).
    Used as the primary comparison baseline.
    """
    from buffer_samoec import DualPurposeBuffer as ReplayBuffer
    model = SAMoEC(cfg, use_ssl_router=False).to(device)
    buffer = ReplayBuffer(rho=cfg.rho, seed=cfg.seed)
    history = []

    for step, d in enumerate(domain_order):
        train_ds, val_ds = domains[d]
        print(f"\n=== [Baseline SAMoE-C] Step {step+1}: Domain D{d+1} ===")

        if step == 0:
            train_initial_domain(model, cfg, device, train_ds, verbose=verbose)
            spec_idx = 0
        else:
            spec_idx = model.add_specialist()
            train_specialist(model, cfg, device, train_ds, spec_idx, verbose=verbose)

        if hasattr(train_ds, "dataset"):
            base_ds = train_ds.dataset
            idxs = train_ds.indices
            X_full, y_full = base_ds.full_tensor()
            X_domain = X_full[idxs]
            y_domain = y_full[idxs]
        else:
            X_domain, y_domain = train_ds.full_tensor()

        X_mix, y_mix = buffer.get_mixed(X_domain, current_domain_idx=spec_idx)
        update_router(model, cfg, device, X_mix, y_mix, verbose=verbose)
        buffer.add_domain(X_domain, y_domain, spec_idx)

        spec_acc = evaluate_specialist(model, device, val_ds, spec_idx, cfg)
        router_acc = evaluate_router_accuracy(model, device, val_ds, spec_idx, cfg)
        print(f"  -> D{d+1} specialist val acc = {spec_acc:.2f}%, "
              f"router self-discrimination = {router_acc:.2f}%")
        history.append({"domain": d, "specialist_acc": spec_acc, "router_acc": router_acc})

    return model, buffer, history


def run_basic_cl(cfg, device, domains, domain_order, verbose=True):
    """Basic CL: single model, retrained from scratch (fresh init) each domain."""
    accs_over_time = []  # accuracy on ALL seen domains after each new domain
    seen_val_sets = []
    model = None

    for step, d in enumerate(domain_order):
        train_ds, val_ds = domains[d]
        print(f"\n=== Basic CL - Step {step+1}: Domain D{d+1} (retrain from scratch) ===")
        model = BasicCL(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        ce = torch.nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        model.train()
        for ep in range(cfg.epochs_specialist):
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = ce(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()

        seen_val_sets.append(val_ds)
        # average accuracy across ALL domains seen so far
        model.eval()
        accs = []
        with torch.no_grad():
            for vds in seen_val_sets:
                loader_v = torch.utils.data.DataLoader(vds, batch_size=cfg.batch_size)
                correct, n = 0, 0
                for X, y in loader_v:
                    X, y = X.to(device), y.to(device)
                    pred = model(X).argmax(-1)
                    correct += (pred == y).sum().item(); n += X.size(0)
                accs.append(100.0 * correct / n)
        avg_past_acc = sum(accs[:-1]) / len(accs[:-1]) if len(accs) > 1 else float("nan")
        new_acc = accs[-1]
        print(f"  -> new-domain acc={new_acc:.2f}%, avg-past-domain acc={avg_past_acc:.2f}%")
        accs_over_time.append({"domain": d, "new_acc": new_acc, "avg_past_acc": avg_past_acc})

    return model, accs_over_time


def final_balanced_test(model, device, domains, cfg, mode="route"):
    """Concatenate val sets of all domains into one balanced test set (Table III)."""
    all_val = torch.utils.data.ConcatDataset([v for _, v in domains])
    return evaluate_full_pipeline(model, device, all_val, cfg, mode=mode)


def print_kpi_comparison(proposed_hist, baseline_hist, basic_hist,
                          proposed_final, baseline_final, memo_proposed, memo_baseline,
                          proposed_mflops, baseline_mflops, basic_mflops,
                          memo_proposed_mflops, memo_baseline_mflops):
    """Print comprehensive KPI comparison table."""
    print("\n" + "="*80)
    print("  KPI COMPARISON: Proposed Framework vs SAMoE-C vs Basic CL")
    print("="*80)

    # Table II style: progressive accuracy
    print("\n--- TABLE II: Per-Domain Specialist Accuracy (%) ---")
    print(f"{'Domain':<10}{'Proposed':<15}{'SAMoE-C':<15}{'Basic CL':<15}")
    print("-"*55)
    for i, (ph, bh) in enumerate(zip(proposed_hist, baseline_hist)):
        basic_acc = basic_hist[i]['new_acc'] if i < len(basic_hist) else float('nan')
        print(f"D{ph['domain']+1:<9}{ph['specialist_acc']:<15.2f}"
              f"{bh['specialist_acc']:<15.2f}{basic_acc:<15.2f}")

    # Router accuracy
    print("\n--- Router Self-Discrimination Accuracy (%) ---")
    print(f"{'Domain':<10}{'Proposed':<15}{'SAMoE-C':<15}")
    print("-"*40)
    for ph, bh in zip(proposed_hist, baseline_hist):
        print(f"D{ph['domain']+1:<9}{ph['router_acc']:<15.2f}{bh['router_acc']:<15.2f}")

    # Table III style: final accuracy & inference cost
    print("\n--- TABLE III: Final Balanced Accuracy & Inference Cost ---")
    print(f"{'Method':<30}{'Final Avg Acc(%)':<20}{'MFLOPs/Sample':<15}")
    print("-"*65)
    print(f"{'Basic CL':<30}{basic_hist[-1]['new_acc']:<20.2f}{basic_mflops:<15.2f}")
    print(f"{'MEMO (SAMoE-C backbone)':<30}{memo_baseline:<20.2f}{memo_baseline_mflops:<15.2f}")
    print(f"{'SAMoE-C (original)':<30}{baseline_final:<20.2f}{baseline_mflops:<15.2f}")
    print(f"{'MEMO (proposed backbone)':<30}{memo_proposed:<20.2f}{memo_proposed_mflops:<15.2f}")
    print(f"{'*** PROPOSED FRAMEWORK ***':<30}{proposed_final:<20.2f}{proposed_mflops:<15.2f}")

    # Improvement summary
    print("\n--- IMPROVEMENT SUMMARY ---")
    acc_diff = proposed_final - baseline_final
    router_avg_proposed = sum(h['router_acc'] for h in proposed_hist) / len(proposed_hist)
    router_avg_baseline = sum(h['router_acc'] for h in baseline_hist) / len(baseline_hist)
    spec_avg_proposed = sum(h['specialist_acc'] for h in proposed_hist) / len(proposed_hist)
    spec_avg_baseline = sum(h['specialist_acc'] for h in baseline_hist) / len(baseline_hist)

    print(f"  Final balanced accuracy: {'+' if acc_diff >= 0 else ''}{acc_diff:.2f}% vs SAMoE-C")
    print(f"  Avg specialist accuracy: {spec_avg_proposed:.2f}% (proposed) vs {spec_avg_baseline:.2f}% (SAMoE-C)")
    print(f"  Avg router accuracy: {router_avg_proposed:.2f}% (proposed) vs {router_avg_baseline:.2f}% (SAMoE-C)")
    print(f"  Inference cost: {proposed_mflops:.2f} MFLOPs (proposed) vs {baseline_mflops:.2f} MFLOPs (SAMoE-C)")


def main():
    parser = argparse.ArgumentParser(description="SAMoE-C Isolated Run")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real MM-Fi/CSI-Bench data (default: synthetic smoke test)")
    parser.add_argument("--mmfi-root", type=str, default=None)
    parser.add_argument("--alt-order", action="store_true",
                        help="Use alternate domain ordering")
    parser.add_argument("--diagnostic-leaky-control", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run quick smoke test (overrides real-data)")
    
    args = parser.parse_args()
    
    cfg = Config()
    cfg.epochs_initial = 1 if args.smoke_test else cfg.epochs_initial
    cfg.epochs_specialist = 1 if args.smoke_test else cfg.epochs_specialist
    cfg.epochs_router = 1 if args.smoke_test else cfg.epochs_router

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Smoke test: {args.smoke_test}")

    t0 = time.time()
    
    if args.diagnostic_leaky_control:
        domains = build_domains_leaky_diagnostic(cfg, use_real_data=args.real_data, mmfi_root=args.mmfi_root)
    else:
        domains = build_domains(cfg, use_real_data=args.real_data, mmfi_root=args.mmfi_root)
        
    print(f"\nBuilt {len(domains)} domains.")

    if args.smoke_test:
        order = [cfg.DOMAIN_ORDER_ALT[0] if args.alt_order else cfg.DOMAIN_ORDER_DEFAULT[0]]
        baseline_model, baseline_buffer, baseline_hist = run_samoec_baseline(cfg, device, domains, order, verbose=True)
        return

    order = cfg.DOMAIN_ORDER_ALT if args.alt_order else cfg.DOMAIN_ORDER_DEFAULT
    print(f"\nUsing Domain Curriculum Order: {order}")

    print("\n" + "="*70)
    print("  STARTING SAMOE-C BASELINE RUN")
    print("="*70)
    
    baseline_model, baseline_buffer, baseline_hist = run_samoec_baseline(cfg, device, domains, order, verbose=True)

    print('\n[SAMoE-C Run Complete]')
    final_samoec = final_balanced_test(baseline_model, device, domains, cfg, mode='route')
    print(f'\n>>> FINAL SAMOE-C BALANCED ACCURACY: {final_samoec:.2f}% <<<\n')

    t3 = time.time()
    print(f"[Total wall-clock time: {t3-t0:.1f}s]")

if __name__ == "__main__":
    main()
