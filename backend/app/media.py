"""Doc thong tin video bang ffprobe (khong phu thuoc OpenCV)."""
import json
import shutil
import subprocess
from pathlib import Path


def probe(path: Path) -> dict:
    """Tra ve duration/fps/width/height; gia tri None neu khong doc duoc (ffprobe vang)."""
    out = {"duration": None, "fps": None, "width": None, "height": None}
    if shutil.which("ffprobe") is None:
        return out
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,duration:format=duration",
           "-of", "json", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return out
    try:
        data = json.loads(r.stdout)
        st = data["streams"][0]
        out["width"], out["height"] = int(st["width"]), int(st["height"])
        num, den = st["r_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        out["fps"] = fps if fps > 0 else None
        dur = st.get("duration") or data.get("format", {}).get("duration")
        out["duration"] = float(dur) if dur else None
    except (KeyError, IndexError, ValueError, ZeroDivisionError):
        pass
    return out
