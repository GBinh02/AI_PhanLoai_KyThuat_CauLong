"""Doc cac tap train/val/test (.npz) do build_dataset_splits.py tao ra."""
from pathlib import Path

import numpy as np


def load_split(data_dir, split):
    """Tra ve X (N, 32, 132) float32 va y (N,) int64 (0-3: clear, net_shot,
    short_service, smash). Tu nhan dien key trong file npz: mang 3 chieu la X,
    mang 1 chieu kieu so nguyen la y."""
    path = Path(data_dir) / f"{split}.npz"
    d = np.load(path, allow_pickle=True)
    X = next((d[k] for k in d.files if d[k].ndim == 3), None)
    y = next((d[k] for k in d.files
              if d[k].ndim == 1 and np.issubdtype(d[k].dtype, np.integer)), None)
    if X is None or y is None:
        raise KeyError(f"Khong nhan dien duoc X/y trong {path}, cac key: {d.files}")
    return X.astype(np.float32), y.astype(np.int64)
