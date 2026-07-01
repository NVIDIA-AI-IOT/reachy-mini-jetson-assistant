import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
from app.config import Config
from app.camera import Camera
from app.face_detector import FaceDetector, _resolve_model_path
from app.reachy import kill_stale_camera_holders
from rich.console import Console

console = Console()
cfg = Config.load()
kill_stale_camera_holders(cfg.vision.camera_device, console)
cam = Camera(device=cfg.vision.camera_device, width=cfg.vision.width,
             height=cfg.vision.height, jpeg_quality=cfg.vision.jpeg_quality,
             capture_fps=cfg.vision.capture_fps)
cam.start()
mp = str(_resolve_model_path())
det = cv2.FaceDetectorYN.create(mp, "", (320,320), score_threshold=0.3, nms_threshold=0.3, top_k=5000)
time.sleep(1.0)
frame=None
for _ in range(10):
    frame = cam.read_raw_live()
    if frame is not None: break
    time.sleep(0.1)
if frame is None:
    print("NO FRAME"); cam.close(); sys.exit(1)
h,w=frame.shape[:2]; print(f"FRAME {w}x{h}, mean_brightness={frame.mean():.1f}")
det.setInputSize((w,h))
_,faces=det.detect(frame)
print("faces:", 0 if faces is None else len(faces))
if faces is not None:
    for f in faces:
        x,y,fw,fh,sc=int(f[0]),int(f[1]),int(f[2]),int(f[3]),f[-1]
        print(f"  box=({x},{y},{fw}x{fh}) score={sc:.2f}")
        cv2.rectangle(frame,(x,y),(x+fw,y+fh),(0,255,0),3)
cv2.line(frame,(w//2,0),(w//2,h),(0,128,255),1)
cv2.line(frame,(0,h//2),(w,h//2),(0,128,255),1)
cv2.imwrite("/tmp/face_debug.jpg", frame)
print("saved /tmp/face_debug.jpg")
cam.close()
