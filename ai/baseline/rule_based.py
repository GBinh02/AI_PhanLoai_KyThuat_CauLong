"""
rule_based.py - Phuong phap doi chung (baseline) rule-based, muc 8 dac ta.

Khong hoc mo hinh: chi 3 dac trung tho trich thu cong tu keypoint + cay luat 2 tang:

    Tree A (mac dinh):  peak_height >= H ?
                          co   -> <vfeat> >= V ? smash : clear   (vfeat = peak_speed | down_speed)
                          khong -> ankle_spread >= S ? net_shot : short_service
    Tree B:             ankle_spread < S ? short_service
                          khong -> peak_height >= H ? (<vfeat> >= V ? smash : clear) : net_shot

Dac trung (toa do da duoc chuan hoa trong extract_pose_features.py: goc = tam hong,
don vi = do dai than vai-hong; y anh huong xuong -> do cao = -y):
    peak_height   : do cao lon nhat cua co tay (2 tay, moi frame) so voi tam hong
    down_speed    : nhu peak_speed nhung chi tinh chuyen dong co tay HUONG XUONG (smash danh xuong)
    peak_speed    : quang duong co tay di duoc lon nhat giua 2 frame lien tiep
                    (don vi: do dai than / buoc resample; chuoi da resample ve T=32)
    ankle_spread  : khoang cach ngang lon nhat giua 2 mat ca chan (dau hieu lao nguoi / chong chan)
Frame co visibility < 0.5 bi bo qua. Neu clip khong thay co tay nao -> peak_height = -2 (coi la "thap").

CACH DUNG (chay tu goc repo):
    # 1. Xem phan bo dac trung theo nhan tren train/val (chon nguong bang mat)
    python -m ai.baseline.rule_based stats --splits train val

    # 2a. Chon nguong tu dong tren TRAIN (3 nguong don le, khong dung val/test)
    python -m ai.baseline.rule_based tune --tree A --vfeat peak_speed
    # 2b. Hoac tu dat nguong bang tay:  --h 0.9 --v 0.35 --s 0.8

    # 3. Danh gia tren val de chinh
    python -m ai.baseline.rule_based eval --split val --vfeat down_speed

    # 4. Chay test DUNG 1 LAN khi da chot nguong (giong run_test.py)
    python -m ai.baseline.rule_based eval --split test --confirm
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ai.models.classifier.data import load_split

LABELS = ["clear", "net_shot", "short_service", "smash"]
CLEAR, NET, SERVICE, SMASH = 0, 1, 2, 3
L_WRIST, R_WRIST, L_ANKLE, R_ANKLE = 15, 16, 27, 28
VIS_THR = 0.5
MISSING_H = -2.0
FEATURES = ["peak_height", "peak_speed", "down_speed", "ankle_spread"]
DEFAULT_THR_PATH = Path("ai/baseline/thresholds.json")


def compute_features(X):
    N, T, _ = X.shape
    lm = X.reshape(N, T, 33, 4)
    xy, vis = lm[..., :2], lm[..., 3]

    def pts(i):
        p = xy[:, :, i, :].copy()
        p[vis[:, :, i] < VIS_THR] = np.nan
        return p

    wrists = [pts(L_WRIST), pts(R_WRIST)]

    heights = np.stack([-w[..., 1] for w in wrists], axis=-1)            # (N,T,2)
    peak_h = np.where(np.isnan(heights), -np.inf, heights).max(axis=(1, 2))
    peak_h[~np.isfinite(peak_h)] = MISSING_H

    speeds = []
    for w in wrists:
        d = np.linalg.norm(w[:, 1:] - w[:, :-1], axis=-1)                # nan neu thieu 1 trong 2 frame
        speeds.append(np.nan_to_num(d, nan=0.0).max(axis=1))
    peak_v = np.maximum(speeds[0], speeds[1])

    downs = []
    for w in wrists:
        dy = w[:, 1:, 1] - w[:, :-1, 1]                                  # y anh huong xuong: dy>0 la di xuong
        downs.append(np.clip(np.nan_to_num(dy, nan=0.0), 0, None).max(axis=1))
    down_v = np.maximum(downs[0], downs[1])

    la, ra = pts(L_ANKLE), pts(R_ANKLE)
    spread = np.nan_to_num(np.abs(la[..., 0] - ra[..., 0]), nan=0.0).max(axis=1)

    return {"peak_height": peak_h, "peak_speed": peak_v, "down_speed": down_v, "ankle_spread": spread}


def predict(F, h, v, s, vfeat="peak_speed", tree="A"):
    high = F["peak_height"] >= h
    fast = F[vfeat] >= v
    wide = F["ankle_spread"] >= s
    if tree == "A":   # chieu cao co tay truoc
        return np.where(high, np.where(fast, SMASH, CLEAR), np.where(wide, NET, SERVICE))
    # tree B: ankle_spread truoc
    return np.where(wide, np.where(high, np.where(fast, SMASH, CLEAR), NET), SERVICE)


def metrics(y, p):
    cm = np.zeros((4, 4), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    per = {}
    for i, name in enumerate(LABELS):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i].sum() if cm[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[name] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i].sum())}
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_f1": float(np.mean([v["f1"] for v in per.values()])),
        "per_class": per,
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(name, m):
    print(f"\n=== {name}: acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} ===")
    for k, v in m["per_class"].items():
        print(f"  {k:14s} P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1']:.3f}  n={v['support']}")
    print("  confusion matrix (hang = that, cot = du doan; thu tu", LABELS, ")")
    for row in m["confusion_matrix"]:
        print("   ", row)


def best_threshold(pos, neg):
    """Nguong t toi da hoa balanced accuracy cho luat 'pos >= t, neg < t'."""
    vals = np.unique(np.concatenate([pos, neg]))
    cands = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else vals
    best_t, best = cands[0], -1
    for t in cands:
        score = 0.5 * ((pos >= t).mean() + (neg < t).mean())
        if score > best:
            best, best_t = score, t
    return float(best_t), float(best)


def cmd_stats(a):
    for split in a.splits:
        X, y = load_split(a.data_dir, split)
        F = compute_features(X)
        print(f"\n##### {split} (N={len(y)}) - phan vi 25/50/75 theo nhan #####")
        for f in FEATURES:
            print(f"[{f}]")
            for i, name in enumerate(LABELS):
                q = np.percentile(F[f][y == i], [25, 50, 75])
                print(f"  {name:14s} {q[0]:7.3f} {q[1]:7.3f} {q[2]:7.3f}")


def cmd_tune(a):
    X, y = load_split(a.data_dir, "train")
    F = compute_features(X)
    if a.tree == "A":
        h, sh = best_threshold(F["peak_height"][np.isin(y, [CLEAR, SMASH])], F["peak_height"][np.isin(y, [NET, SERVICE])])
        s, ss = best_threshold(F["ankle_spread"][y == NET], F["ankle_spread"][y == SERVICE])
    else:
        s, ss = best_threshold(F["ankle_spread"][y != SERVICE], F["ankle_spread"][y == SERVICE])
        h, sh = best_threshold(F["peak_height"][np.isin(y, [CLEAR, SMASH])], F["peak_height"][y == NET])
    v, sv = best_threshold(F[a.vfeat][y == SMASH], F[a.vfeat][y == CLEAR])
    print(f"Nguong tren TRAIN (tree={a.tree}, vfeat={a.vfeat}): H={h:.4f} ({sh:.3f}), V={v:.4f} ({sv:.3f}), S={s:.4f} ({ss:.3f})")
    Path(a.thresholds).parent.mkdir(parents=True, exist_ok=True)
    Path(a.thresholds).write_text(json.dumps({"h": h, "v": v, "s": s, "vfeat": a.vfeat, "tree": a.tree}, indent=2))
    print("Da luu", a.thresholds)
    print_metrics("train", metrics(y, predict(F, h, v, s, a.vfeat, a.tree)))


def cmd_eval(a):
    if a.split == "test" and not a.confirm:
        raise SystemExit("Tap test duoc niem phong: them --confirm khi da chot nguong va chi chay 1 lan.")
    thr = json.loads(Path(a.thresholds).read_text()) if Path(a.thresholds).exists() else {}
    h = a.h if a.h is not None else thr.get("h")
    v = a.v if a.v is not None else thr.get("v")
    s = a.s if a.s is not None else thr.get("s")
    vfeat = a.vfeat or thr.get("vfeat", "peak_speed")
    tree = a.tree or thr.get("tree", "A")
    if None in (h, v, s):
        raise SystemExit("Thieu nguong: chay 'tune' hoac truyen --h --v --s.")
    X, y = load_split(a.data_dir, a.split)
    m = metrics(y, predict(compute_features(X), h, v, s, vfeat, tree))
    m["thresholds"] = {"h": h, "v": v, "s": s, "vfeat": vfeat, "tree": tree}
    print_metrics(f"rule-based / {a.split} / tree {tree} / {vfeat}", m)
    out = Path(f"ai/baseline/results_{a.split}_{tree}_{vfeat}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print("Da luu", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/dataset_splits")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stats")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("tune")
    p.add_argument("--thresholds", default=str(DEFAULT_THR_PATH))
    p.add_argument("--vfeat", choices=["peak_speed", "down_speed"], default="peak_speed")
    p.add_argument("--tree", choices=["A", "B"], default="A")
    p.set_defaults(fn=cmd_tune)

    p = sub.add_parser("eval")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--thresholds", default=str(DEFAULT_THR_PATH))
    p.add_argument("--h", type=float)
    p.add_argument("--v", type=float)
    p.add_argument("--s", type=float)
    p.add_argument("--vfeat", choices=["peak_speed", "down_speed"], default=None,
                   help="Dac trung tach smash/clear (mac dinh lay tu thresholds.json)")
    p.add_argument("--tree", choices=["A", "B"], default=None,
                   help="A: chieu cao co tay truoc; B: ankle_spread truoc (mac dinh lay tu thresholds.json)")
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(fn=cmd_eval)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()