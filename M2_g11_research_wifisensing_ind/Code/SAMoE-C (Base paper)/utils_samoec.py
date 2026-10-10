import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(logits, targets):
    pred = logits.argmax(dim=-1)
    return (pred == targets).float().mean().item() * 100.0


def get_device(cfg):
    return torch.device(cfg.device if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# MFLOPs/sample estimation (Table III). Tries `thop`; falls back to a manual
# analytic count of Conv1d/Linear/GRU FLOPs if thop isn't installed.
# --------------------------------------------------------------------------- #
def estimate_mflops_per_sample(model, sample_input, mode="route", specialist_idx=0):
    try:
        from thop import profile
        model.eval()
        with torch.no_grad():
            if hasattr(model, "specialists"):
                macs, _ = profile(
                    model, inputs=(sample_input,),
                    verbose=False,
                )
            else:  # BasicCL
                macs, _ = profile(model, inputs=(sample_input,), verbose=False)
        flops = 2 * macs  # 1 MAC = 2 FLOPs
        return flops / 1e6 / sample_input.shape[0]
    except Exception:
        return _manual_flop_count(model, sample_input, mode=mode, specialist_idx=specialist_idx)


def _manual_flop_count(model, sample_input, mode="route", specialist_idx=0):
    """Rough analytic fallback: sums Conv1d + Linear + GRU FLOPs for a forward pass."""
    total = {"flops": 0}
    hooks = []

    def conv1d_hook(m, inp, out):
        B, Cout, Lout = out.shape
        Cin = m.in_channels
        K = m.kernel_size[0]
        groups = m.groups
        flops = 2 * (Cin // groups) * K * Cout * Lout * B
        total["flops"] += flops

    def linear_hook(m, inp, out):
        B = inp[0].shape[0] if inp[0].dim() > 1 else 1
        flops = 2 * m.in_features * m.out_features * B
        total["flops"] += flops

    def gru_hook(m, inp, out):
        x = inp[0]
        B, T = x.shape[0], x.shape[1]
        input_size, hidden = m.input_size, m.hidden_size
        num_dir = 2 if m.bidirectional else 1
        # 3 gates, each ~ (input_size+hidden)*hidden mults, *2 for FLOPs, *layers*directions*T
        per_step = 3 * 2 * (input_size * hidden + hidden * hidden)
        flops = per_step * T * m.num_layers * num_dir * B
        total["flops"] += flops

    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            hooks.append(m.register_forward_hook(conv1d_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, torch.nn.GRU):
            hooks.append(m.register_forward_hook(gru_hook))

    model.eval()
    with torch.no_grad():
        if hasattr(model, "specialists"):
            model(sample_input, mode=mode, specialist_idx=specialist_idx if mode == "force" else None)
        else:
            model(sample_input)

    for h in hooks:
        h.remove()

    return total["flops"] / 1e6 / sample_input.shape[0]
