import unittest
import json
import zipfile
import hashlib
from pathlib import Path
from ai.pose_extraction.ood_crop import roi_pixels, local_path

class WorkflowTests(unittest.TestCase):
    def test_windows_manifest_path(self):
        self.assertEqual(local_path(r'data\ood\clips\a.mp4').as_posix(),'data/ood/clips/a.mp4')
    def test_roi_bounds(self):
        self.assertEqual(roi_pixels([.1,.2,.5,.5],100,200),(10,40,50,100))
        for box in [[-.1,0,.5,.5],[0,0,0,1],[.9,0,.2,1],[float('nan'),0,1,1]]:
            with self.assertRaises(ValueError): roi_pixels(box,100,200)
    def test_checkpoint_metadata(self):
        p=Path('ai/models/checkpoints/bilstm_motion_seed5.keras')
        meta=json.loads(p.with_suffix('.json').read_text())
        self.assertEqual(meta['sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
        with zipfile.ZipFile(p) as z:
            config=json.loads(z.read('config.json'))
        self.assertEqual(config['config']['layers'][0]['config']['batch_shape'][1:],meta['input_shape'])
        self.assertEqual(meta['features'],'motion')

if __name__=='__main__': unittest.main()
