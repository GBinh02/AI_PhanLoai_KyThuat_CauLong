"""Tang cuong du lieu tren chuoi keypoint THO (N, 32, 132) - truoc khi build_features.

Moi phep bien doi mo phong mot nguon sai lech that giua video ShuttleSet va video dien thoai:
  - scale_x  : khung crop co ty le rong/cao khac luc huan luyen (0.65) -> toa do x bi gian/nen.
  - scale    : nguoi to/nho khac nhau sau chuan hoa do dai than (sai so uoc luong than).
  - shift_t  : danh dau thoi diem cham cau lech vai khung hinh.
  - warp_t   : toc do dong tac / fps khac nhau (co gian truc thoi gian).
  - mirror   : nguoi choi thuan tay trai (lat ngang + doi khop trai/phai).
  - noise    : rung keypoint cua MediaPipe tren video chat luong thap.
Bo cuc: 132 = 33 khop x (x, y, z, visibility); x,y da tru tam hong, chia do dai than.
"""
import numpy as np

N_LM = 33
_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
          (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
MIRROR_IDX = np.arange(N_LM)
for a, b in _PAIRS:
    MIRROR_IDX[a], MIRROR_IDX[b] = b, a


def _lm(X):
    return np.asarray(X, dtype=np.float32).reshape(len(X), X.shape[1], N_LM, 4).copy()


def _flat(lm):
    return lm.reshape(lm.shape[0], lm.shape[1], -1)


def scale_x(X, s):
    lm = _lm(X); lm[..., 0] *= np.reshape(s, (-1, 1, 1)); return _flat(lm)


def scale_xy(X, s):
    lm = _lm(X); lm[..., :2] *= np.reshape(s, (-1, 1, 1, 1)); return _flat(lm)


def mirror(X, mask=None):
    lm = _lm(X)
    m = np.ones(len(lm), bool) if mask is None else np.asarray(mask)
    flipped = lm[m][:, :, MIRROR_IDX]
    flipped[..., 0] *= -1
    lm[m] = flipped
    return _flat(lm)


def resample_time(X, shift, rate):
    """Lay lai 32 frame tai vi tri t' = (t - c) * rate + c + shift (don vi frame), kep o 2 dau."""
    X = np.asarray(X, dtype=np.float32)
    n, T, F = X.shape
    t = np.arange(T, dtype=np.float32)
    c = (T - 1) / 2
    out = np.empty_like(X)
    shift, rate = np.broadcast_to(shift, n), np.broadcast_to(rate, n)
    for i in range(n):
        src = np.clip((t - c) * rate[i] + c + shift[i], 0, T - 1)
        lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, T - 1); w = (src - lo)[:, None]
        out[i] = X[i, lo] * (1 - w) + X[i, hi] * w
    return out


def random_augment(X, rng, p_mirror=0.5):
    """Bien doi ngau nhien cho 1 batch (dung khi huan luyen)."""
    n = len(X)
    X = resample_time(X, rng.uniform(-2.5, 2.5, n), rng.uniform(0.85, 1.15, n))
    X = scale_x(X, np.exp(rng.uniform(np.log(0.75), np.log(1.33), n)))
    X = scale_xy(X, rng.uniform(0.9, 1.1, n))
    X = mirror(X, rng.random(n) < p_mirror)
    lm = _lm(X)
    lm[..., :3] += rng.normal(0, 0.02, lm[..., :3].shape).astype(np.float32) * (lm[..., 3:4] > 0)
    return _flat(lm)
