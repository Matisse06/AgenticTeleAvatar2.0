#!/usr/bin/env python3
"""Play with the TeleAvatar model in the MuJoCo viewer: the robot's camera views, actuator sliders, keyboard driving.

  python3 model/play.py                # the base fixed, as in the vendor's simulator (model/scene.xml)
  python3 model/play.py --mobile-base  # the drivable base we added (model/scene_mobile.xml)
  python3 model/play.py --no-cameras   # without the camera insets
  python3 model/play.py --model scenes/table_cubes.xml  # a scene: the robot at a table with two cubes
  python3 model/play.py --model scenes/table_cubes.xml --mobile-base  # the same with the drivable base

It needs a display (on the cluster, a Remote Desktop: see CLAUDE.md). It starts at the home keyframe and runs in real
time, with the view following the robot (Esc frees it).
- Insets (right): what the robot's eyes see, through their fisheye lenses (model/cameras.py): the three the vendor's
  policy uses, the head's left eye (square, as the robot's 960 x 960 head images) and each wrist's inner eye (1.6:1,
  as its 640 x 400 images).
- Control panel (right): a slider per actuator. Arms in rad; lift in m (0 is the top, positive lowers the torso);
  grippers in N m of motor torque, from -1.6 (closing) to +2.0 (opening), the range the robot's 0..1 command spans
  (model/gripper.py). With --mobile-base also the base's velocity in the world frame: base_x and
  base_y in m/s, base_yaw in rad/s (0 holds the pose); while the keyboard drives, it sets those sliders.
- Keys: Backspace returns home; Space pauses. With --mobile-base: Up / Down change the forward speed and Left / Right
  the turn rate, one step per press; on the keypad, 8 / 2 forward and back, 4 / 6 sideways, 7 / 9 turn; End or keypad
  5 stops. The viewer's own keys also work: [ and ] cycle the cameras, Esc frees the view, Tab hides the left panel,
  double-click selects a body, and Ctrl + right-drag pushes it.
"""
from __future__ import annotations

import argparse
import math
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # for the offscreen insets; the viewer window itself uses GLFW

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cameras import LENSES, POLICY_EYES, FisheyeCamera, face_size
from mobile_base import MobileBase, drivable_variant

HERE = Path(__file__).resolve().parent
# (camera, label): the eyes the vendor's policy sees (cameras.POLICY_EYES)
INSETS = ((POLICY_EYES["head_camera"], "head, left eye"), (POLICY_EYES["left_color"], "left wrist, right eye"),
          (POLICY_EYES["right_color"], "right wrist, left eye"))
STEPS = np.array([0.1, 0.1, 0.25])  # forward, left (m/s) and turn (rad/s) change per key press
KEYS = {  # GLFW key code -> direction of the change; None stops
    265: (1, 0, 0), 264: (-1, 0, 0), 263: (0, 0, 1), 262: (0, 0, -1),  # arrows up, down, left, right
    328: (1, 0, 0), 322: (-1, 0, 0), 324: (0, 1, 0), 326: (0, -1, 0),  # keypad 8, 2, 4, 6
    327: (0, 0, 1), 329: (0, 0, -1), 325: None, 269: None,             # keypad 7, 9, 5; End
}
SPACE, BACKSPACE = 32, 259
MARGIN = 8


def join_viewer_thread() -> None:
    """Wait for the passive viewer's daemon thread to finish tearing down. Otherwise the interpreter's exit can run
    glfw.terminate while that thread is still drawing, which crashes the exit (segfault 139)."""
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and thread.daemon and "_launch_internal" in thread.name:
            thread.join(timeout=5.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mobile-base", action="store_true",
                        help="load the drivable-base variant, which we added (the vendor's simulator has a fixed "
                             "base): model/scene_mobile.xml, or the --model file's <name>_mobile.xml")
    parser.add_argument("--model", type=Path, help="a model or scene file (default: model/scene.xml)")
    parser.add_argument("--no-cameras", action="store_true", help="no camera insets")
    parser.add_argument("--inset-width", type=int, default=192,
                        help="pixels (the head's inset is square, the wrists' 1.6:1, as the robot's images)")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 runs until the window is closed")
    args = parser.parse_args()

    path = args.model or HERE / "scene.xml"
    if args.mobile_base:
        path, source = drivable_variant(path), path
        if not path.is_file():
            parser.error(f"--mobile-base: {source} has no drivable-base variant ({path} does not exist)")
    model = mujoco.MjModel.from_xml_path(str(path.resolve()))
    data = mujoco.MjData(model)
    home = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, home)
    drivable = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_x") >= 0
    base = MobileBase(model) if drivable else None
    limits = model.actuator_ctrlrange[base.actuators, 1] if drivable else None
    state = {"command": np.zeros(3), "reset": False, "stop": False, "paused": False}

    def on_key(key: int) -> None:  # runs on the viewer's thread
        if key == SPACE:
            state["paused"] = not state["paused"]
        elif key == BACKSPACE:
            state["reset"] = True
        elif not drivable:
            return
        elif KEYS.get(key, 0) is None:
            state.update(command=np.zeros(3), stop=True)  # also what the base sliders set
        elif key in KEYS:
            state["command"] = np.clip(state["command"] + STEPS * KEYS[key], -limits, limits)

    # The insets' offscreen (EGL) context must exist before the viewer's window: with NVIDIA's driver, no new EGL
    # context can be made current once the process has a GLX context (tested on this workstation, 2026-10-01).
    renderers, eyes = {}, []  # one renderer per cube-face size (one for every inset at the default width)
    if not args.no_cameras:
        try:
            for camera, label in INSETS:
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
                    continue  # a model without the eye cameras (the vendor's)
                lens = LENSES[camera]
                lens = lens.scaled(args.inset_width, round(args.inset_width * lens.height / lens.width))
                size = face_size(lens)
                if size not in renderers:
                    renderers[size] = mujoco.Renderer(model, size, size)
                eyes.append((FisheyeCamera(model, camera, lens, renderer=renderers[size]), label))
        except Exception as error:  # e.g. no EGL on this machine: run without the insets
            print(f"no camera insets ({error})")
            eyes = []
        font = ImageFont.load_default(size=max(10, args.inset_width // 14))

    def insets(viewport: mujoco.MjrRect) -> list:
        images, top = [], viewport.bottom + viewport.height
        for eye, label in eyes:
            pixels = eye.render(data)
            image = Image.fromarray(pixels)
            ImageDraw.Draw(image).text((6, 3), label, font=font, fill=(255, 255, 255), stroke_width=2,
                                       stroke_fill=(0, 0, 0))
            height, width = pixels.shape[:2]
            top -= height + MARGIN
            rect = mujoco.MjrRect(viewport.left + viewport.width - width - MARGIN, top, width, height)
            images.append((rect, np.asarray(image)))
        return images

    def status() -> tuple:
        paused = "PAUSED" if state["paused"] else ""
        if not drivable:
            return (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    "base\nkeys\n" + paused,
                    "fixed, as in the vendor's simulator (--mobile-base drives it)\nBackspace: home   Space: pause\n")
        x, y, yaw = base.pose(data)
        forward, left, turn = state["command"]
        wheels = "  ".join(f"{speed:+.1f}" for speed in base.wheel_speeds(data))
        return (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                "base\ncommand\nwheels\nkeys\n" + paused,
                f"x {x:+.2f} m  y {y:+.2f} m  yaw {math.degrees(yaw):+.0f} deg\n"
                f"{forward:+.1f} m/s ahead  {left:+.1f} m/s left  {turn:+.2f} rad/s\n"
                f"{wheels} rad/s\narrows, keypad: drive   End: stop\nBackspace: home   Space: pause")

    try:
        run(args, model, data, base, state, on_key, insets if eyes else None, status)
    finally:
        join_viewer_thread()
        for eye, _ in eyes:
            eye.close()
        for renderer in renderers.values():
            renderer.close()


def run(args, model, data, base, state, on_key, insets, status) -> None:
    home = model.key("home").id
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        with viewer.lock():  # follow the robot as it drives (the lookat point tracks its centre of mass)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body("base_link").id
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.4, 140.0, -18.0
        started = time.monotonic()
        offset = started - data.time  # wall clock = simulated time + offset
        next_overlay = 0.0
        while viewer.is_running() and (not args.duration or time.monotonic() - started < args.duration):
            now = time.monotonic()
            with viewer.lock():
                if state["reset"]:
                    state.update(reset=False, command=np.zeros(3))
                    mujoco.mj_resetDataKeyframe(model, data, home)
                    if base is not None:
                        base.velocity[:] = 0.0
                    mujoco.mj_forward(model, data)
                    offset = now - data.time
                if state["stop"]:  # a keyboard command ramps down by itself; speeds set with the sliders stop here
                    state["stop"] = False
                    if not base.velocity.any():
                        data.ctrl[base.actuators] = 0.0
                if state["paused"]:
                    offset = now - data.time
                else:
                    for _ in range(50):  # catch up with the wall clock, at most 0.1 s per frame
                        if data.time + offset >= now:
                            break
                        if base is not None:
                            if state["command"].any() or base.velocity.any():
                                base.command(data, *state["command"])
                            base.roll_wheels(data)
                        mujoco.mj_step(model, data)
                    else:
                        offset = now - data.time  # fell behind: run slower than real time instead of jumping
            viewer.sync()
            if now >= next_overlay:
                viewport = viewer.viewport
                if insets and viewport is not None and viewport.width > 0:
                    viewer.set_images(insets(viewport))
                viewer.set_texts(status())
                # 20 Hz, less often when drawing takes long (software rendering), so the physics keeps up
                next_overlay = now + max(1.0 / 20.0, 4.0 * (time.monotonic() - now))
            time.sleep(max(0.0, 1.0 / 120.0 - (time.monotonic() - now)))


if __name__ == "__main__":
    main()
