"""
Dataset utilities.

MM-Fi is not redistributable here, so two things are provided:

1. `SyntheticCSIDataset` -- generates domain-shifted synthetic CSI tensors
   with the exact shape (C, T, S) = (3, 10, 114) used in the paper, so you
   can run/debug/unit-test the whole pipeline end-to-end right now.

2. `MMFiCSIDataset` -- a loader stub that reads pre-extracted CSI amplitude
   tensors + activity labels from .npy files. Point it at your own
   pre-processed MM-Fi WiFi-CSI export (see README.md for the expected
   on-disk layout) to reproduce the paper's numbers on real data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticCSIDataset(Dataset):
    """
    Generates `n_samples` CSI tensors for a single "domain". Each domain gets
    its own random per-class mean amplitude pattern + a domain-specific
    multipath "signature" (a random linear transform + noise), so that:
      - P(X|y) truly differs across domains (Eq. 2 in the paper)
      - the label space Y is shared across domains
    This is only meant to validate the training/eval pipeline, NOT to
    reproduce the paper's reported accuracies.
    """
    def __init__(self, num_classes, C, T, S, n_samples=800, domain_seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(domain_seed)
        self.num_classes = num_classes
        self.C, self.T, self.S = C, T, S

        # per-class prototype pattern (shared structure) + domain-specific
        # multipath transform (random per-subcarrier scaling + phase-like shift)
        self.class_protos = torch.randn(num_classes, C, T, S, generator=g) * 0.5
        self.domain_scale = 0.6 + 0.8 * torch.rand(1, C, 1, S, generator=g)
        self.domain_shift = 0.3 * torch.randn(1, C, 1, S, generator=g)

        self.labels = torch.randint(0, num_classes, (n_samples,), generator=g)
        noise = 0.25 * torch.randn(n_samples, C, T, S, generator=g)
        base = self.class_protos[self.labels]
        self.X = base * self.domain_scale + self.domain_shift + noise
        self.X = self.X.float()
        
        # Dummy subject labels for synthetic dataset to keep API consistent
        self.subj_y = torch.zeros(n_samples, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]

    def full_tensor(self):
        """Returns (X, y) for the entire domain (useful for buffer sampling)."""
        return self.X, self.labels


class MMFiCSIDataset(Dataset):
    """
    Loader stub for real, pre-processed MM-Fi CSI data.

    Expected on-disk layout (adapt to however you export MM-Fi):
        root/
          D1/csi_amplitude.npy   # shape (N, C, T, S) float32
          D1/labels.npy          # shape (N,) int64, activity ids in [0, K)
          D2/csi_amplitude.npy
          D2/labels.npy
          ...

    MM-Fi CSI is public but requires filling in an access request with the
    dataset authors [13]; once you have the raw .mat/.csv captures, use
    `preprocess_mmfi.py` (stub also provided) to window them into the
    (C, T, S) tensors this class expects.
    """
    def __init__(self, root_dir, domain_name):
        import os
        x_path = os.path.join(root_dir, domain_name, "csi_amplitude.npy")
        y_path = os.path.join(root_dir, domain_name, "labels.npy")
        group_path = os.path.join(root_dir, domain_name, "group_ids.npy")
        
        X_np = np.load(x_path)
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
        self.X = torch.from_numpy(X_np).float()
        self.y = torch.from_numpy(np.load(y_path)).long()
        
        if os.path.exists(group_path):
            group_ids = np.load(group_path, allow_pickle=True)
            # Extract Subject IDs (e.g., "S01_A01" -> 0)
            subj_list = []
            for gid in group_ids:
                subj_str = gid.split('_')[0]
                if subj_str.startswith('S') and subj_str[1:].isdigit():
                    subj_list.append(int(subj_str[1:]) - 1)
                else:
                    subj_list.append(0) # fallback
            self.subj_y = torch.tensor(subj_list, dtype=torch.long)
        else:
            self.subj_y = torch.zeros(len(self.y), dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def full_tensor(self):
        return self.X, self.y
