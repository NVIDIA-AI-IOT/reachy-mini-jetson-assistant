import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import Config
from app.camera import Camera
from app.face_detector import FaceDetector
from app.movement_manager import MovementManager
from app.face_tracker import FaceTracker
from app.reachy import kill_stale_camera_holders, connect as connect_reachy
from rich.console import Console

OUT = open("/tmp/lookat_stats.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    try: OUT.write(s + "\n"); OUT.flush()
    except Exception: pass

cfg = Config.load(); console = Console()
reachy = connect_reachy(cfg, console)
if reachy is None:
    log("NO ROBOT"); sys.exit(1)

kill_stale_camera_holders(cfg.vision.camera_device, console)
cam = Camera(device=cfg.vision.camera_device, width=cfg.vision.width, height=cfg.vision.height,
             jpeg_quality=cfg.vision.jpeg_quality, capture_fps=cfg.vision.capture_fps)
cam.start()
det = FaceDetector(); det.load()
time.sleep(1.0)

mgr = MovementManager(reachy); mgr.start()
trk = FaceTracker(cam, det, mgr, reachy,
                  fps=cfg.reachy.tracking_fps,
                  dead_zone=cfg.reachy.tracking_dead_zone,
                  lock_zone=cfg.reachy.tracking_lock_zone,
                  reacquire_zone=cfg.reachy.tracking_reacquire_zone,
                  good_frame_zone=cfg.reachy.tracking_good_frame_zone,
                  min_face_size=cfg.reachy.tracking_min_face_size,
                  stable_frames=cfg.reachy.tracking_stable_frames,
                  face_lost_delay=cfg.reachy.tracking_face_lost_delay,
                  head_yaw_max_deg=cfg.reachy.tracking_head_yaw_max_deg,
                  head_yaw_gain=cfg.reachy.tracking_head_yaw_gain,
                  head_yaw_step=cfg.reachy.tracking_head_yaw_step,
                  soft_center_head_yaw_max_deg=cfg.reachy.tracking_soft_center_head_yaw_max_deg,
                  soft_center_head_yaw_step=cfg.reachy.tracking_soft_center_head_yaw_step,
                  body_max_deg=cfg.reachy.tracking_body_max_deg,
                  body_gain=cfg.reachy.tracking_body_gain,
                  body_step=cfg.reachy.tracking_body_step,
                  invert_body=cfg.reachy.tracking_invert_body,
                  body_enabled=cfg.reachy.tracking_body_enabled,
                  vertical=cfg.reachy.tracking_vertical,
                  return_to_neutral=cfg.reachy.tracking_return_to_neutral,
                  scan_enabled=cfg.reachy.tracking_scan_enabled,
                  scan_body_range_deg=cfg.reachy.tracking_scan_body_range_deg,
                  scan_speed_deg_per_sec=cfg.reachy.tracking_scan_speed_deg_per_sec)
trk.start()

log(f"INSTRUMENTED tracking. fps={cfg.reachy.tracking_fps} body_max={cfg.reachy.tracking_body_max_deg} "
    f"invert={cfg.reachy.tracking_invert_body}. Sit in view; robot should rotate toward you.")
log("Columns: t det centered stable err_x err_y -> body_target pitch_target | current_body")

t0 = time.time()
while time.time() - t0 < 30.0:
    box = trk.last_face_box
    det_f = trk.face_detected
    cen = trk.centered
    stable = trk.stable
    body_t = trk._body
    pitch_t = trk._pitch
    cur_body = mgr._c_body
    if box is not None:
        f = cam.read_raw_live()
        if f is not None:
            h, w = f.shape[:2]
            cx = (box[0] + box[2]) / 2; cy = (box[1] + box[3]) / 2
            ex = (cx - w/2)/(w/2); ey = (cy - h/2)/(h/2)
            log(f"t={time.time()-t0:4.1f} det={det_f} cen={cen} stable={stable} err_x={ex:+.2f} err_y={ey:+.2f} "
                f"-> body={body_t:+6.1f} pitch={pitch_t:+5.1f} | cur_body={cur_body:+6.1f}")
    else:
        log(f"t={time.time()-t0:4.1f} det=False (no face)        "
            f"-> body={body_t:+6.1f} pitch={pitch_t:+5.1f} | cur_body={cur_body:+6.1f}")
    time.sleep(0.4)

log("resetting + stopping")
trk.stop(); time.sleep(1.0); mgr.stop(); cam.close()
try: OUT.close()
except Exception: pass
print("DONE")
