"""Full range-of-motion diagnostic for Reachy Mini.

Runs a labeled, smooth sequence to check visible flexibility:
  - head yaw left/right
  - head pitch up/down
  - head roll left/right
  - diagonal head poses
  - body yaw left/right

The values are intentionally near the ranges we have already used during
calibration, not hard mechanical limits. Keep hands clear while it runs.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config
from app.reachy import connect as connect_reachy
from reachy_mini.utils import create_head_pose

console = Console()
_NEUTRAL_ANTENNAS = np.array([0.0, 0.0], dtype=np.float64)


def _ease(t: float) -> float:
    """Smoothstep easing for gentle motor motion."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _send(reachy, *, yaw: float, pitch: float, roll: float, body: float) -> None:
    head = create_head_pose(yaw=yaw, pitch=pitch, roll=roll, degrees=True)
    reachy.set_target(
        head=head,
        antennas=_NEUTRAL_ANTENNAS,
        body_yaw=math.radians(body),
    )


def move_to(
    reachy,
    label: str,
    target: dict[str, float],
    *,
    current: dict[str, float],
    duration: float = 2.0,
    hold: float = 0.8,
    hz: float = 100.0,
) -> dict[str, float]:
    console.print(
        f"  -> {label:<32} "
        f"head(yaw={target['yaw']:+.0f}, pitch={target['pitch']:+.0f}, roll={target['roll']:+.0f}) "
        f"body={target['body']:+.0f}"
    )
    steps = max(1, int(duration * hz))
    period = 1.0 / hz

    for i in range(steps):
        a = _ease((i + 1) / steps)
        pose = {
            key: current[key] + (target[key] - current[key]) * a
            for key in ("yaw", "pitch", "roll", "body")
        }
        _send(reachy, **pose)
        time.sleep(period)

    end = target.copy()
    t0 = time.monotonic()
    while time.monotonic() - t0 < hold:
        _send(reachy, **end)
        time.sleep(period)
    return end


def main() -> None:
    config = Config.load()
    reachy = connect_reachy(config, console)
    if reachy is None:
        console.print("[red]No robot connection.[/red]")
        return

    neutral = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "body": 0.0}
    current = neutral.copy()

    # Near-maximum practical values observed/used safely in calibration.
    sequence = [
        ("CENTER", neutral),
        ("HEAD YAW LEFT", {"yaw": +30.0, "pitch": 0.0, "roll": 0.0, "body": 0.0}),
        ("HEAD YAW RIGHT", {"yaw": -30.0, "pitch": 0.0, "roll": 0.0, "body": 0.0}),
        ("CENTER", neutral),
        ("HEAD PITCH UP", {"yaw": 0.0, "pitch": +20.0, "roll": 0.0, "body": 0.0}),
        ("HEAD PITCH DOWN", {"yaw": 0.0, "pitch": -20.0, "roll": 0.0, "body": 0.0}),
        ("CENTER", neutral),
        ("HEAD ROLL LEFT", {"yaw": 0.0, "pitch": 0.0, "roll": +22.0, "body": 0.0}),
        ("HEAD ROLL RIGHT", {"yaw": 0.0, "pitch": 0.0, "roll": -22.0, "body": 0.0}),
        ("CENTER", neutral),
        ("DIAGONAL UP-LEFT", {"yaw": +25.0, "pitch": +15.0, "roll": 0.0, "body": 0.0}),
        ("DIAGONAL DOWN-RIGHT", {"yaw": -25.0, "pitch": -15.0, "roll": 0.0, "body": 0.0}),
        ("CENTER", neutral),
        ("BODY LEFT", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "body": +55.0}),
        ("BODY RIGHT", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "body": -55.0}),
        ("BODY + HEAD LEFT", {"yaw": +25.0, "pitch": 0.0, "roll": 0.0, "body": +45.0}),
        ("BODY + HEAD RIGHT", {"yaw": -25.0, "pitch": 0.0, "roll": 0.0, "body": -45.0}),
        ("CENTER / NEUTRAL", neutral),
    ]

    console.print("\n[bold cyan]Full range-of-motion diagnostic[/bold cyan]")
    console.print("Keep the area clear. Watch each labeled movement.\n")

    try:
        for label, target in sequence:
            current = move_to(reachy, label, target, current=current)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted; returning to neutral.[/yellow]")
    finally:
        move_to(reachy, "FINAL NEUTRAL", neutral, current=current, duration=1.5, hold=1.0)
        console.print("\n[green]Done.[/green]")


if __name__ == "__main__":
    main()
