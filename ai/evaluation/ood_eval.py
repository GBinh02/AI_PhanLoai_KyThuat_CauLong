"""
ood_eval.py - Danh gia out-of-domain (muc 6-7 dac ta) tren video tu quay.

Chay CUNG MOT tap clip OOD qua 2 phuong phap da CHOT tu truoc (khong chinh gi them):
  - Mo hinh hoc sau: BiLSTM + motion (checkpoint da chon tren val)
  - Baseline rule-based: nguong trong ai/baseline/thresholds.json (fit tren train)

Giong tap test ShuttleSet, tap OOD chi dung de DANH GIA, chay 1 lan -> bat buoc --confirm.

Bao cao 2 goc nhin (deu nen dua vao bao cao):
  [A] Tren cac clip trich duoc pose (status OK): so sanh cong bang 2 phuong phap.
  [B] Toan he thong: tinh tren TAT CA clip da gan nhan; clip khong crop duoc / khong
      trich duoc pose tinh la du doan sai -> phan anh dung trai nghiem nguoi dung that.

CACH DUNG (chay tu goc repo):
    python -m ai.evaluation.ood_eval --confirm

Dau ra (ai/evaluation/results/ood/):
    ood_predictions.csv        tung clip: nhan that, du doan + do tin cay BiLSTM, du doan baseline
    ood_report.json            toan bo so lieu
    <tag>_ood_cm.png, rule_based_ood_cm.png   ma tran nham lan
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai.baseline import rule_based as rb
from ai.evaluation.metrics import LABELS, evaluate, plot_confusion_matrix
from ai.features.motion import build_features

LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}


def base(p):
    return Path(str(p).replace("\\", "/")).name


def load_ood(features_manifest, clips_manifest):
    fm = pd.read_csv(features_manifest)
    fm["clip_name"] = fm["clip"].map(base)
    cm = pd.read_csv(clips_manifest)
    cm["clip_name"] = cm["clip"].map(base)
    cm = cm.rename(columns={"label": "label_true"})
    df = cm.merge(fm[["clip_name", "status", "visible_ratio", "wrist_ratio", "feature_path"]],
                  on="clip_name", how="left")
    df["status"] = df["status"].fillna("NOT_CROPPED")
    bad = set(df["label_true"]) - set(LABELS)
    if bad:
        raise SystemExit(f"Nhan la trong clips_manifest: {bad}")
    return df


def load_X(paths):
    X = np.stack([np.load(str(p).replace("\\", "/")) for p in paths]).astype(np.float32)
    if X.shape[1:] != (32, 132):
        raise SystemExit(f"Shape dac trung OOD {X.shape[1:]} khac luc huan luyen (32, 132)")
    return X


def predict_bilstm(ckpt, X):
    import tensorflow as tf
    meta = Path(ckpt).with_suffix(".json")
    feats = json.loads(meta.read_text())["features"] if meta.exists() else "raw"
    model = tf.keras.models.load_model(ckpt)
    proba = model.predict(build_features(X, feats), verbose=0)
    return proba.argmax(axis=1), proba.max(axis=1), feats


def predict_rule(thr_path, X):
    thr = json.loads(Path(thr_path).read_text())
    F = rb.compute_features(X)
    pred = rb.predict(F, thr["h"], thr["v"], thr["s"], thr.get("vfeat", "peak_speed"), thr.get("tree", "A"))
    return pred, thr


def to_jsonable(m):
    out = dict(m)
    if "confusion_matrix" in out:
        out["confusion_matrix"] = np.asarray(out["confusion_matrix"]).tolist()
    return out


def show(name, m):
    print(f"\n=== {name}: acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} ===")
    for lab, s in m["per_class"].items():
        f1 = s.get("f1")
        f1s = f"  F1 {f1:.3f}" if f1 is not None else ""
        print(f"  {lab:14s} P {s['precision']:.3f}  R {s['recall']:.3f}{f1s}  n={s['support']}")
    print("  confusion matrix (hang = that, cot = du doan;", LABELS, ")")
    for row in np.asarray(m["confusion_matrix"]):
        print("   ", list(map(int, row)))


def group_acc(df, by, cols):
    rows = []
    for key, g in df.groupby(by):
        r = {by: key, "n": len(g)}
        for c in cols:
            r[f"acc_{c}"] = round(float((g[c] == g["label_true"]).mean()), 3)
        rows.append(r)
    return pd.DataFrame(rows)


def read_test_ref(path):
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("accuracy"), d.get("macro_f1")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="ai/models/checkpoints/bilstm_motion_seed5.keras")
    ap.add_argument("--thresholds", default="ai/baseline/thresholds.json")
    ap.add_argument("--features-manifest", default="data/ood/pose_features/features_manifest.csv")
    ap.add_argument("--clips-manifest", default="data/ood/clips_manifest.csv")
    ap.add_argument("--out-dir", default="ai/evaluation/results/ood")
    ap.add_argument("--rule-test-json", default="ai/baseline/results_test_A_peak_speed.json")
    ap.add_argument("--confirm", action="store_true",
                    help="Xac nhan: model va nguong da chot, tap OOD chi danh gia 1 lan")
    a = ap.parse_args()
    if not a.confirm:
        raise SystemExit("Tap OOD dung de danh gia cuoi: them --confirm (chi chay 1 lan, khong chinh model theo ket qua).")

    df = load_ood(a.features_manifest, a.clips_manifest)
    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        raise SystemExit("Khong co clip OOD nao status OK.")

    X = load_X(ok["feature_path"])
    y = ok["label_true"].map(LABEL_TO_IDX).to_numpy()

    p_dl, conf, feats = predict_bilstm(a.checkpoint, X)
    p_rb, thr = predict_rule(a.thresholds, X)
    ok["pred_bilstm"] = [LABELS[i] for i in p_dl]
    ok["conf_bilstm"] = np.round(conf, 4)
    ok["pred_rule"] = [LABELS[i] for i in p_rb]

    tag = Path(a.checkpoint).stem
    m_dl, m_rb = evaluate(y, p_dl), evaluate(y, p_rb)

    # ---- Tong quan du lieu
    print("=== Tap OOD ===")
    print(f"Clip da gan nhan: {len(df)}")
    print(df["status"].value_counts().to_string())
    print("\nSo clip OK theo nhan:")
    print(ok["label_true"].value_counts().reindex(LABELS, fill_value=0).to_string())

    # ---- [A] Tren clip co pose
    show(f"[A] BiLSTM ({tag}, features={feats}) - {len(ok)} clip co pose", m_dl)
    show(f"[A] Rule-based (tree {thr.get('tree', 'A')}, {thr.get('vfeat', 'peak_speed')}) - {len(ok)} clip co pose", m_rb)

    # ---- [B] Toan he thong
    all_df = df.merge(ok[["clip_name", "pred_bilstm", "conf_bilstm", "pred_rule"]], on="clip_name", how="left")
    sys_dl = float((all_df["pred_bilstm"] == all_df["label_true"]).mean())
    sys_rb = float((all_df["pred_rule"] == all_df["label_true"]).mean())
    print(f"\n[B] Accuracy toan he thong ({len(all_df)} clip, clip loi tinh la sai): "
          f"BiLSTM {sys_dl:.4f} | Rule-based {sys_rb:.4f}")

    # ---- Phan tich theo nhom
    cols = ["pred_bilstm", "pred_rule"]
    by_side = group_acc(ok, "player_side", cols)
    by_person = group_acc(ok, "person", cols)
    print("\nAccuracy theo vi tri nguoi danh:")
    print(by_side.to_string(index=False))
    print("\nAccuracy theo nguoi choi:")
    print(by_person.to_string(index=False))

    correct = ok["pred_bilstm"] == ok["label_true"]
    print(f"\nDo tin cay BiLSTM: trung binh khi dung {ok.loc[correct, 'conf_bilstm'].mean():.3f}, "
          f"khi sai {ok.loc[~correct, 'conf_bilstm'].mean():.3f}" if (~correct).any() and correct.any() else "")

    # ---- Bang so sanh voi ShuttleSet test
    ref_dl = read_test_ref(Path("ai/evaluation/results/test") / f"{tag}_test.json")
    ref_rb = read_test_ref(a.rule_test_json)
    print("\n=== So sanh macro-F1 (ShuttleSet test -> OOD [A]) ===")
    for name, ref, m in [("BiLSTM", ref_dl, m_dl), ("Rule-based", ref_rb, m_rb)]:
        ref_s = f"{ref[1]:.4f}" if ref and ref[1] is not None else "  (chua co json)"
        print(f"  {name:11s} {ref_s} -> {m['macro_f1']:.4f}")

    # ---- Luu ket qua
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keep = ["clip_name", "video", "person", "player_side", "label_true", "status",
            "pred_bilstm", "conf_bilstm", "pred_rule"]
    all_df[[c for c in keep if c in all_df.columns]].to_csv(out / "ood_predictions.csv", index=False, lineterminator="\n")
    plot_confusion_matrix(np.asarray(m_dl["confusion_matrix"]), out / f"{tag}_ood_cm.png", f"{tag} (OOD)")
    plot_confusion_matrix(np.asarray(m_rb["confusion_matrix"]), out / "rule_based_ood_cm.png", "Rule-based (OOD)")
    report = {
        "labels": LABELS,
        "n_annotated": int(len(df)),
        "status_counts": df["status"].value_counts().to_dict(),
        "bilstm": {"checkpoint": a.checkpoint, "features": feats, **to_jsonable(m_dl),
                   "system_accuracy": sys_dl, "shuttleset_test": ref_dl},
        "rule_based": {"thresholds": thr, **to_jsonable(m_rb),
                       "system_accuracy": sys_rb, "shuttleset_test": ref_rb},
        "by_side": by_side.to_dict(orient="records"),
        "by_person": by_person.to_dict(orient="records"),
    }
    (out / "ood_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float))
    print(f"\nDa luu ket qua vao {out}")


if __name__ == "__main__":
    main()
