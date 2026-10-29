"""Tao dac trung chuyen dong tu chuoi keypoint (N, 32, 132).

Van de: smash va clear giong nhau ve tu the, khac nhau chu yeu o TOC DO va LUC
vung tay, nhung dac trung tho (vi tri khop tung frame) khong the hien ro dieu do.

Che do (--features):
  raw         : 132 dac trung goc (x, y, z, visibility cua 33 khop)
  motion      : raw + van toc khop (hieu giua 2 frame lien tiep cua x,y,z) -> 231
  motion_eng  : motion + 18 dac trung sinh co hoc (goc khuyu, toc do co tay,
                do cao co tay so voi vai, do rong vai/hong, do nghieng vai)
                va dao ham theo thoi gian cua chung -> 249

Gia dinh bo cuc: 132 = 33 khop x (x, y, z, visibility), khop lien tiep nhau.
Chi so MediaPipe Pose: 11/12 vai T/P, 13/14 khuyu, 15/16 co tay, 23/24 hong T/P.
"""
import numpy as np

N_LM = 33
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24
FEATURE_MODES = ["raw", "motion", "motion_eng"]


def _delta(a):
    """Hieu giua frame t va t-1 theo truc thoi gian (axis=1); frame dau = 0."""
    d = np.zeros_like(a)
    d[:, 1:] = a[:, 1:] - a[:, :-1]
    return d


def _angle(a, b, c):
    """Goc tai b giua a-b va c-b (radian), tinh tren mat phang anh (x, y)."""
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6
    return np.arccos(np.clip((v1 * v2).sum(-1) / denom, -1.0, 1.0))


def _engineered(lm):
    """lm: (N, T, 33, 4) -> (N, T, 18)."""
    xy = lm[..., :2]
    sh_mid_y = (xy[:, :, L_SH, 1] + xy[:, :, R_SH, 1]) / 2
    feats = []
    for sh, el, wr in ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)):
        feats.append(_angle(xy[:, :, sh], xy[:, :, el], xy[:, :, wr]))      # goc khuyu
        feats.append(np.linalg.norm(_delta(xy[:, :, wr]), axis=-1))         # toc do co tay
        feats.append(sh_mid_y - xy[:, :, wr, 1])                            # co tay cao hon vai
    sh_vec = xy[:, :, R_SH] - xy[:, :, L_SH]
    hip_vec = xy[:, :, R_HIP] - xy[:, :, L_HIP]
    feats.append(np.arctan2(sh_vec[..., 1], np.abs(sh_vec[..., 0])))        # nghieng vai
    feats.append(np.linalg.norm(sh_vec, axis=-1))                           # do rong vai (xoay than)
    feats.append(np.linalg.norm(hip_vec, axis=-1))                          # do rong hong
    e = np.stack(feats, axis=-1)                                            # (N, T, 9)
    return np.concatenate([e, _delta(e)], axis=-1)                          # (N, T, 18)


def build_features(X, mode="raw"):
    if mode not in FEATURE_MODES:
        raise ValueError(f"mode khong hop le: {mode}. Chon trong {FEATURE_MODES}")
    X = np.asarray(X, dtype=np.float32)
    if mode == "raw":
        return X
    n, t, f = X.shape
    if f != N_LM * 4:
        raise ValueError(f"Mong doi {N_LM * 4} dac trung/frame, nhan {f}")
    lm = X.reshape(n, t, N_LM, 4)
    parts = [X, _delta(lm[..., :3].reshape(n, t, -1))]
    if mode == "motion_eng":
        parts.append(_engineered(lm))
    return np.concatenate(parts, axis=-1).astype(np.float32)
