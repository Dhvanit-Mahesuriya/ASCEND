"""
Configuration for the ASCEND framework.

Base values from the SAMoE-C paper (Zheng, Mao & Ho, arXiv:2605.06447v1).
CIG-MAE SSL values from (Liu, Hao & Zou, IEEE IoT-J 2026).

"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---- CSI tensor dims (Table I, Sec. II-A): X in R^{C x T x S} ----
    C: int = 3        # N_tx * N_rx (antenna pairs)
    T: int = 10        # time instances per activity window
    S: int = 114       # OFDM subcarriers

    # ---- Task setup (Sec. V-A) ----
    K: int = 27        # number of human activity classes (MM-Fi)
    samples_per_domain: int = 150 # number of samples per domain
    N_DOMAINS: int = 4  # D1..D4 (two living rooms, two meeting rooms)
    DOMAIN_ORDER_DEFAULT: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    DOMAIN_ORDER_ALT: List[int] = field(default_factory=lambda: [2, 0, 1, 3])

    # ---- Shared backbone channel plan (Table I, Part A) ----
    backbone_channels: List[int] = field(default_factory=lambda: [64, 96, 144, 256])

    # ---- Specialist channel plan (Table I, Part C) ----
    specialist_channels: List[int] = field(default_factory=lambda: [128, 96, 64])
    gru_hidden: int = 128
    gru_layers: int = 2

    # ---- Router (Table I, Part B) ----
    router_dim: int = 256
    router_gate_blocks: int = 3

    # ---- Attention ----
    attn_hidden: int = 128

    # ---- Training protocol (Sec. IV & V-A) ----
    batch_size: int = 64         # Matched to exactly 64 from paper Section V-A
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs_initial: int = 35     # initial backbone+specialist-1 training
    epochs_specialist: int = 35  # incremental specialist training
    epochs_router: int = 20      # Matched to exactly 20 from paper Section V-A
    rho: float = 0.05            # replay-buffer fraction (matched paper exactly)
    use_dann: bool = False       # SAMoE-C does NOT use DANN.
    seed: int = 42
    device: str = "cuda"  # falls back to cpu automatically if unavailable

    # ---- CIG-MAE-adapted SSL pretraining of the shared backbone ----
    # (Liu, Hao & Zou, "CIG-MAE", IEEE IoT-J 2026 -- adapted single-stream,
    # amplitude-only, since this repo's preprocessing has no phase. See
    # ssl_pretrain.py for the full mapping and what was dropped/kept.)
    use_ssl_pretrain: bool = False   # if True, main.py replaces supervised
                                      # Phase 1 backbone training with this
    ssl_patch_size: int = 6          # subcarrier patch width (S=114 -> 19 patches)
    ssl_mask_ratio: float = 0.75     # fraction of patches HIDDEN (rest visible)
    ssl_aim_hidden: int = 128        # AIM policy transformer hidden dim
    ssl_epochs: int = 100
    ssl_lr: float = 1e-4
    ssl_batch_size: int = 64
    ssl_waim: float = 1e-4           # weight on the AIM policy-gradient loss
    ssl_decoder_channels: tuple = (144, 96, 64)  # mirrors backbone_channels, reversed

    # ---- Barlow Twins intra-stream consistency (single-stream substitute for
    # CIG-MAE's dual-stream cross-modal alignment, Sec. III-D / Eq. 9-11) ----
    ssl_bt_dim: int = 1024           # projector output dim (CIG-MAE Sec. IV-A)
    ssl_bt_lambda: float = 0.005     # off-diagonal penalty (Zbontar et al. default)
    ssl_wbt: float = 0.2            # weight of BT loss vs reconstruction (CIG-MAE Sec. IV-A)

    # ======================================================================= #
    # Continual SSL with EWC regularization
    # ======================================================================= #
    lambda_ewc: float = 1000.0      # EWC regularization strength (reduced to allow real adaptation)
    ewc_gamma: float = 0.90         # online EWC consolidation decay
    ssl_adapt_epochs: int = 50      # SSL adaptation epochs per new domain
    ssl_adapt_lr: float = 1e-4      # adaptation LR (same as initial, EWC prevents drift)

    # ======================================================================= #
    # SSL-Informed Router
    # ======================================================================= #
    use_ssl_router: bool = True     # enable hybrid gate + prototype routing

    # ======================================================================= #
    # SSL-Anchored Specialist Distillation
    # ======================================================================= #
    alpha_kd: float = 0.3           # weight of KD loss vs CE loss

    # ======================================================================= #
    # Domain-Conditioned Masking
    # ======================================================================= #
    use_domain_conditioned_aim: bool = True  # domain-aware AIM policy

    # ======================================================================= #
    # Domain-Specific Adapter Layers
    # Instead of adapting the backbone (which causes drift), each domain
    # gets a tiny bottleneck adapter that refines frozen backbone features.
    # Eliminates feature space drift entirely. No rehearsal needed.
    # ======================================================================= #
    adapter_bottleneck_dim: int = 32    # bottleneck dimension (~16K params per adapter)

