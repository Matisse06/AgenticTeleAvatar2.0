#!/usr/bin/env python3
"""Play the nominal model in real time. Space pauses; Backspace resets home."""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=HERE / "scene.xml")
    parser.add_argument("--speed", type=float, default=1.0, help="real-time factor")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 runs until viewer closes")
    parser.add_argument("--paused", action="store_true", help="start paused")
    args = parser.parse_args()
    if args.speed <= 0 or args.duration < 0:
        parser.error("--speed must be positive and --duration non-negative")
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise RuntimeError("model does not contain the required 'home' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    state = {"paused": args.paused, "reset": False}
    def key_callback(keycode: int) -> None:
        if keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode == 259:
            state["reset"] = True
    wall_start = time.monotonic()
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and (args.duration == 0 or time.monotonic() - wall_start < args.duration):
            frame_start = time.monotonic()
            if state["reset"]:
                mujoco.mj_resetDataKeyframe(model, data, home_id); state["reset"] = False
            if not state["paused"]:
                mujoco.mj_step(model, data)
            viewer.sync()
            target = model.opt.timestep / args.speed
            remaining = target - (time.monotonic() - frame_start)
            if remaining > 0:
                time.sleep(remaining)

if __name__ == "__main__":
    main()
