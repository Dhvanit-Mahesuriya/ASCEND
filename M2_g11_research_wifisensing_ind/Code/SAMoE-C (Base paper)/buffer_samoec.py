"""
Dual-purpose experience-replay buffer (ENHANCED from SAMoE-C's router-only
buffer, Sec. IV-C).

Original: Stores rho-fraction of each past domain's data ONLY for router
stabilization — never used for SSL or specialist training.

ENHANCED (Change 5): The buffer now serves DUAL purposes:
  1. Router calibration (original role, with BALANCED sampling to prevent
     the router from biasing toward the most recent domain)
  2. SSL replay: buffered past-domain samples are used as contrastive
     anchors during continual SSL adaptation (Change 1), preventing the
     backbone from forgetting previously learned representations.
"""
import random
import torch
from collections import defaultdict


class DualPurposeBuffer:
    """
    Extends the original ReplayBuffer with:
      - get_balanced_mixed(): oversamples buffer data to match current domain
        size, preventing the router from forgetting old domains.
      - get_ssl_replay(): returns past domain samples (unlabeled) for
        continual SSL adaptation with EWC regularization.
    """
    def __init__(self, rho: float = 0.15, seed: int = 42):
        self.rho = rho
        self.rng = random.Random(seed)
        self.store = defaultdict(list)  # domain_idx -> [X_tensor]
        self._labels = {}               # domain_idx -> y_tensor (activity labels)

    def add_domain(self, X: torch.Tensor, y: torch.Tensor, domain_idx: int):
        """
        Balanced sampling: pick a fixed fraction `rho` of the domain's samples,
        stratified so every activity class is represented proportionally.
        """
        chosen = []
        classes = torch.unique(y)
        for c in classes:
            c_idx = (y == c).nonzero(as_tuple=True)[0]
            c_k = max(1, int(round(self.rho * len(c_idx))))
            idx_list = c_idx.tolist()
            self.rng.shuffle(idx_list)
            chosen.extend(idx_list[:c_k])

        self.rng.shuffle(chosen)
        self.store[domain_idx] = [X[chosen].detach().cpu()]
        self._labels[domain_idx] = y[chosen].detach().cpu()

    def get_mixed(self, current_X: torch.Tensor, current_domain_idx: int):
        """
        Returns (X_mix, y_mix) combining current-domain data with buffer.
        Uses basic concatenation (original behavior).
        """
        Xs = [current_X.cpu()]
        ys = [torch.full((current_X.shape[0],), current_domain_idx, dtype=torch.long)]
        for d, tensors in self.store.items():
            if d == current_domain_idx:
                continue
            Xd = tensors[0]
            Xs.append(Xd)
            ys.append(torch.full((Xd.shape[0],), d, dtype=torch.long))
        return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)

    def get_balanced_mixed(self, current_X: torch.Tensor, current_domain_idx: int):
        """
        Returns (X_mix, y_mix) with buffer data OVERSAMPLED to match the
        current domain's size per class. This prevents the router from
        being biased toward the most recent domain.

        For each past domain in buffer, we repeat its samples until it
        reaches approximately the same count as the current domain.
        """
        n_current = current_X.shape[0]
        Xs = [current_X.cpu()]
        ys = [torch.full((n_current,), current_domain_idx, dtype=torch.long)]

        n_past_domains = sum(1 for d in self.store if d != current_domain_idx)
        if n_past_domains == 0:
            return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)

        # Target: each past domain gets ~same number of samples as current
        target_per_domain = n_current

        for d, tensors in self.store.items():
            if d == current_domain_idx:
                continue
            Xd = tensors[0]
            n_buf = Xd.shape[0]
            if n_buf == 0:
                continue

            # Oversample by repeating + random subset of remainder
            repeats = target_per_domain // n_buf
            remainder = target_per_domain % n_buf
            parts = [Xd] * repeats
            if remainder > 0:
                idx = torch.randperm(n_buf)[:remainder]
                parts.append(Xd[idx])
            Xd_balanced = torch.cat(parts, dim=0).cpu()
            Xs.append(Xd_balanced)
            ys.append(torch.full((Xd_balanced.shape[0],), d, dtype=torch.long))

        return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)

    def get_ssl_replay(self):
        """
        Returns unlabeled past-domain samples for SSL replay (NOVEL).
        Returns: (X_replay, domain_ids) or (None, None) if buffer is empty.
        """
        if len(self.store) == 0:
            return None, None

        Xs, domain_ids = [], []
        for d, tensors in self.store.items():
            Xd = tensors[0]
            Xs.append(Xd)
            domain_ids.append(torch.full((Xd.shape[0],), d, dtype=torch.long))

        if len(Xs) == 0:
            return None, None

        return torch.cat(Xs, dim=0), torch.cat(domain_ids, dim=0)

    def has_data(self):
        return len(self.store) > 0

    def __len__(self):
        return sum(t[0].shape[0] for t in self.store.values())
