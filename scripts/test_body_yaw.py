import sys, time, threading, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from app.config import Config
from app.camera import Camera
from app.face_detector import FaceDetector
from app.reachy import kill_stale_camera_holders, connect as connect_reachy
from reachy_mini.utils import create_head_pose
from rich.console import Console

OUT = open("/tmp/bodyyaw_stats.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n"); OUT.flush()

cfg = Config.load(); console = Console()
reachy = connect_reachy(cfg, console)
if reachy is None:
    log("NO ROBOT"); sys.exit(1)

# 1) Does look_at_image work in our no_media setup?
try:
    pose = reachy.look_at_image(960, 540, duration=0.0, perform_movement=False)
    log("look_at_image OK, pose translation:", np.round(pose[:3, 3], 3))
except Exception as e:
    log("look_at_image FAILED:", repr(e))

# 2) Stream a body_yaw command in a thread
cmd = {"yaw": 0.0, "pitch": 0.0, "body": 0.0, "run": True}
ant = np.array([0.0, 0.0])
def _stream():
    while cmd["run"]:
        try:
            reachy.set_target(
                head=create_head_pose(yaw=cmd["yaw"], pitch=cmd["pitch"], degrees=True),
                antennas=ant, body_yaw=cmd["body"])
        except Exception as e:
            log("set_target err:", repr(e)); break
        time.sleep(0.01)
threading.Thread(target=_stream, daemon=True).start()
time.sleep(2.0)

kill_stale_camera_holders(cfg.vision.camera_device, console)
cam = Camera(device=cfg.vision.camera_device, width=cfg.vision.width, height=cfg.vision.height,
             jpeg_quality=cfg.vision.jpeg_quality, capture_fps=cfg.vision.capture_fps)
cam.start(); det = FaceDetector(); det.load(); time.sleep(1.0)

def measure(n=14):
    xs = []; ys = []; w = h = 0
    for _ in range(n):
        f = cam.read_raw_live()
        if f is not None:
            h, w = f.shape[:2]
            b = det.detect(f)
            if b is not None: xs.append((b[0]+b[2])/2); ys.append((b[1]+b[3])/2)
        time.sleep(0.06)
    if not xs: return None
    return (float(np.median(xs)), float(np.median(ys)), w, h)

def goto_body(body_rad, secs=2.5):
    cmd["body"] = body_rad; time.sleep(secs)

log("Sit in view. Waiting up to 30s for a face at neutral...")
goto_body(0.0, secs=0.5)
base = None
t_wait = time.time()
while time.time() - t_wait < 30.0:
    base = measure(n=6)
    if base is not None:
        break
    log("  ...no face yet, keep sitting in front of the robot")
if base is None:
    log("no face after 30s - aborting"); cam.close(); cmd["run"] = False; sys.exit(0)
log(f"neutral body=0: cx={base[0]:.0f}/{base[2]}")

for b in (+0.4, -0.4):
    goto_body(b); m = measure()
    if m: log(f"body_yaw={b:+.2f}rad ({math.degrees(b):+.0f}deg): cx={m[0]:.0f} (dcx={m[0]-base[0]:+.0f})")
    else: log(f"body_yaw={b:+.2f}rad: face lost")
    goto_body(0.0, secs=2.0)

cam.close(); cmd["run"] = False; OUT.close()
print("done")
