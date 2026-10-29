"""Evaluate a frozen model and baseline on the SAME OOD samples, retaining failures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from ai.baseline.rule_based import compute_features, predict, metrics, LABELS
from ai.features.motion import build_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='data/ood/cropped/manifest.csv')
    p.add_argument('--checkpoint',default='ai/models/checkpoints/bilstm_motion_seed5.keras')
    p.add_argument('--pose-model',default='models/pose_landmarker_full.task')
    p.add_argument('--thresholds',default='ai/baseline/thresholds.json')
    p.add_argument('--out-dir',default='ai/evaluation/results/ood')
    p.add_argument('--confirm',action='store_true',help='Confirm ROI/labels/config are frozen before viewing predictions')
    a=p.parse_args()
    if not a.confirm:
        p.error('Review labels and ROI first, then use --confirm')
    ckpt=Path(a.checkpoint)
    meta=json.loads(ckpt.with_suffix('.json').read_text())
    if meta.get('labels') != LABELS:
        raise SystemExit('Checkpoint label order is missing or incompatible')
    digest=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if meta.get('sha256') != digest:
        raise SystemExit('Checkpoint hash mismatch')
    import tensorflow as tf
    from ai.pose_extraction.extract_pose_features import process_clip
    model=tf.keras.models.load_model(ckpt)
    thr=json.loads(Path(a.thresholds).read_text())
    rows=list(csv.DictReader(open(a.manifest,encoding='utf-8-sig')))
    if not rows or any(r['label'] not in LABELS for r in rows):
        raise SystemExit('Empty manifest or invalid labels')
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    features=[]; good=[]
    for r in rows:
        r.update(status=r['crop_status'],error=r.get('crop_error',''),bilstm='',baseline='',confidence='')
        if r['crop_status'] != 'OK':
            continue
        try:
            X,status,visible,wrist=process_clip(r['cropped_clip'],a.pose_model,0.7)
            r.update(status=status,visible_ratio=visible,wrist_ratio=wrist)
            if status == 'OK':
                if X.shape != (32,132) or not np.isfinite(X).all():
                    raise ValueError('Invalid pose feature shape/values')
                features.append(X);good.append(r)
        except Exception as e:
            r.update(status='FAILED',error=str(e))
    report={'total':len(rows),'valid':len(good),'coverage':len(good)/len(rows),'checkpoint_sha256':digest,
            'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),'thresholds':thr,
            'features':meta['features'],'labels':LABELS}
    if good:
        X=np.stack(features); prepared=build_features(X,meta['features'])
        if tuple(prepared.shape[1:]) != tuple(model.input_shape[1:]):
            raise SystemExit('Feature shape does not match model')
        probs=model.predict(prepared,verbose=0); pred=probs.argmax(1)
        base=predict(compute_features(X),thr['h'],thr['v'],thr['s'],thr['vfeat'],thr['tree'])
        y=np.array([LABELS.index(r['label']) for r in good])
        for i,r in enumerate(good):
            r.update(bilstm=LABELS[pred[i]],baseline=LABELS[base[i]],confidence=float(probs[i,pred[i]]))
        for name, values in [('bilstm',pred),('baseline',base)]:
            report[name]=metrics(y,values)
            report[name]['end_to_end_accuracy']=float((values==y).sum()/len(rows))
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    ref=json.loads(Path('ai/evaluation/results/test/bilstm_motion_seed5_test.json').read_text())
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['metric','ShuttleSet test - BiLSTM','OOD - BiLSTM','OOD - baseline'])
        for key in ['accuracy','macro_f1']:
            w.writerow([key,ref[key],report.get('bilstm',{}).get(key,''),report.get('baseline',{}).get(key,'')])
        w.writerow(['valid_samples',sum(c['support'] for c in ref['per_class'].values()),len(good),len(good)])
        w.writerow(['coverage','',report['coverage'],report['coverage']])
        w.writerow(['end_to_end_accuracy','',report.get('bilstm',{}).get('end_to_end_accuracy',''),report.get('baseline',{}).get('end_to_end_accuracy','')])
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
