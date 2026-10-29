"""Select per-video/side ROIs, with per-clip overrides, then crop every OOD sample.
Run from repository root. ROI coordinates are normalized to the source frame.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path


def local_path(value):
    return Path(value.replace('\\', '/'))


def roi_pixels(roi, width, height):
    x, y, w, h = map(float, roi)
    if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x+w <= 1.000001 and y+h <= 1.000001):
        raise ValueError('Invalid normalized ROI')
    px, py = int(x*width)//2*2, int(y*height)//2*2
    pw, ph = min(int(w*width)//2*2, width-px), min(int(h*height)//2*2, height-py)
    if min(pw, ph) < 2:
        raise ValueError('ROI too small')
    return px, py, pw, ph


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['select', 'crop'])
    p.add_argument('--manifest', default='data/ood/clips_manifest.csv')
    p.add_argument('--rois', default='data/ood/rois.json')
    p.add_argument('--clip', help='Clip path to select; required for select')
    p.add_argument('--time', type=float, default=0.98)
    p.add_argument('--override', action='store_true', help='Save ROI for this clip only')
    p.add_argument('--out-dir', default='data/ood/cropped')
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.manifest, encoding='utf-8-sig')))
    cfg_path = Path(a.rois)
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {'groups': {}, 'clips': {}}
    if a.command == 'select':
        import cv2
        import numpy as np
        if not a.clip:
            p.error('--clip is required')
        row = next((r for r in rows if local_path(r['clip']) == local_path(a.clip)), None)
        if row is None:
            p.error('Clip not found in manifest')
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(a.time), '-i', str(local_path(a.clip)), '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'])
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit('Cannot decode selection frame')
        # Fit portrait videos on laptop displays, retain normalized coordinates.
        h, w = frame.shape[:2]
        scale = min(1, 800/h, 1100/w)
        frame = cv2.resize(frame, (round(w*scale), round(h*scale)))
        h, w = frame.shape[:2]
        box = cv2.selectROI(f"Target: {row['player_side']} - Enter to save, Esc to cancel", frame, False)
        cv2.destroyAllWindows()
        if min(box[2:]) == 0:
            return
        roi = [box[0]/w, box[1]/h, box[2]/w, box[3]/h]
        key = local_path(row['clip']).as_posix() if a.override else row['video']+'|'+row['player_side']
        cfg['clips' if a.override else 'groups'][key] = roi
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=2)+'\n')
        print('Saved', key, roi)
        return
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, row in enumerate(rows):
        r = dict(row, sample_id=f'ood_{i:04d}', cropped_clip='', crop_status='', crop_error='')
        clip = local_path(row['clip'])
        roi = cfg.get('clips', {}).get(clip.as_posix(), cfg.get('groups', {}).get(row['video']+'|'+row['player_side']))
        try:
            if roi is None:
                raise ValueError('MISSING_ROI')
            info = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','json',str(clip)]))['streams'][0]
            x,y,w,h = roi_pixels(roi, info['width'], info['height'])
            dest = out / (r['sample_id']+'.mp4')
            subprocess.run(['ffmpeg','-v','error','-y','-i',str(clip),'-vf',f'crop={w}:{h}:{x}:{y}','-an','-c:v','libx264','-pix_fmt','yuv420p',str(dest)],check=True)
            r.update(cropped_clip=dest.as_posix(), crop_status='OK')
        except Exception as e:
            r.update(crop_status='FAILED', crop_error=str(e))
        results.append(r)
    if not results:
        raise SystemExit('Empty manifest')
    with (out/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=results[0].keys());writer.writeheader();writer.writerows(results)
    print(f"Cropped {sum(r['crop_status']=='OK' for r in results)}/{len(results)}; see {out/'manifest.csv'}")


if __name__ == '__main__':
    main()
