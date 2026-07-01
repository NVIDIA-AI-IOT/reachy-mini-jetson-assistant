import sys, time, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from app.config import Config
from app.camera import Camera
from app.face_detector import FaceDetector
from app.reachy import kill_stale_camera_holders, connect as connect_reachy
from reachy_mini.utils import create_head_pose
from rich.console import Console

OUT = open("/tmp/sign_stats.txt", "w")
def log(*a):
    s=" ".join(str(x) for x in a); print(s); OUT.write(s+"\n"); OUT.flush()

cfg = Config.load(); console = Console()
reachy = connect_reachy(cfg, console)
if reachy is None:
    log("NO ROBOT"); sys.exit(1)

cmd = {"yaw":0.0, "pitch":0.0, "run":True}
ant = np.array([0.0,0.0])
def _stream():
    while cmd["run"]:
        try: reachy.set_target(head=create_head_pose(yaw=cmd["yaw"], pitch=cmd["pitch"], degrees=True), antennas=ant, body_yaw=0.0)
        except Exception: pass
        time.sleep(0.01)
threading.Thread(target=_stream, daemon=True).start()
time.sleep(2.0)

kill_stale_camera_holders(cfg.vision.camera_device, console)
cam = Camera(device=cfg.vision.camera_device, width=cfg.vision.width, height=cfg.vision.height,
             jpeg_quality=cfg.vision.jpeg_quality, capture_fps=cfg.vision.capture_fps)
cam.start(); det = FaceDetector(); det.load(); time.sleep(1.0)

def measure(n=14):
    xs=[]; ys=[]; w=h=0
    for _ in range(n):
        f = cam.read_raw_live()
        if f is not None:
            h,w=f.shape[:2]
            b=det.detect(f)
            if b is not None: xs.append((b[0]+b[2])/2); ys.append((b[1]+b[3])/2)
        time.sleep(0.06)
    if not xs: return None
    return (float(np.median(xs)), float(np.median(ys)), w, h)

def goto(yaw=0.0, pitch=0.0, secs=1.8):
    cmd["yaw"]=yaw; cmd["pitch"]=pitch; time.sleep(secs)

log("Face the robot. Probing yaw sign...")
goto(0,0); base = measure()
if base is None: log("no face at neutral - sit in view & rerun"); cam.close(); cmd["run"]=False; sys.exit(0)
log(f"neutral: cx={base[0]:.0f}/{base[2]} cy={base[1]:.0f}/{base[3]}")

goto(yaw=-7); right = measure(); goto(0,0)
goto(yaw=+7); left = measure(); goto(0,0,secs=1.5)

if right: log(f"yaw -7 (head RIGHT): cx={right[0]:.0f}  (dcx={right[0]-base[0]:+.0f})")
else:     log("yaw -7: face lost")
if left:  log(f"yaw +7 (head LEFT) : cx={left[0]:.0f}  (dcx={left[0]-base[0]:+.0f})")
else:     log("yaw +7: face lost")

# Conclusion: tracker uses delta_yaw = -gain*err_x. For face at right (err_x>0)
# it commands yaw<0 (head right). That is CORRECT only if head-right moves the
# face toward center, i.e. decreases cx.
if right:
    if right[0] < base[0]:
        log("=> CORRECT: head-right lowers cx (face toward center). Sign OK.")
    else:
        log("=> WRONG: head-right raised cx (face toward edge). FLIP yaw sign.")
cam.close(); cmd["run"]=False; OUT.close()
