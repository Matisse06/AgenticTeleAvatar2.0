#!/usr/bin/env python3
"""Render a TeleAvatar rollout to MP4 (or the home pose to PNG) offscreen, to open in VS Code.

  cd ~/TeleAvatar2.0 && ./sim.sh python3 scripts/render_video.py --output outputs/wave.mp4
  ./sim.sh python3 scripts/render_video.py --output outputs/home.png       # one still of the home keyframe
  ./sim.sh python3 scripts/render_video.py --model mujoco/scene.xml ...    # the vendor's original model

The default model is the new one (model/scene.xml). Motions: 'wave' swings every arm joint sinusoidally around the
home keyframe (clipped to the actuator ranges) and, on the new model, opens and closes the grippers; 'hold' commands
the home keyframe, which shows how well the position actuators hold the arms against gravity; 'drive' (new model
only) drives the base forward, sideways and turning, around a square back to its start, with the wheels rolling. Other
actuators (the lift, and the base except in 'drive') hold their home targets. Renders on the CPU (Mesa) or a GPU,
whichever EGL provides.
"""
import argparse
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering, no display needed; must be set before importing mujoco

from pathlib import Path

import imageio
import mujoco
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "model" / "scene.xml"
ARM_JOINT = re.compile(r"^(l_joint|r_joint)\d$|^arm[LR]\d_joint$")  # vendor model / new model
# 'drive': (seconds, forward m/s, left m/s, turn rad/s) in the robot's frame. 0.6 m ahead, 0.6 m left, a quarter turn
# left, then the same legs back (now sideways and reversing), and a quarter turn right: back at the start.
QUARTER = 0.5 * np.pi
DRIVE = [(1.5, 0.4, 0, 0), (0.5, 0, 0, 0), (1.5, 0, 0.4, 0), (0.5, 0, 0, 0), (QUARTER, 0, 0, 1.0), (0.5, 0, 0, 0),
         (1.5, 0, 0.4, 0), (0.5, 0, 0, 0), (1.5, -0.4, 0, 0), (0.5, 0, 0, 0), (QUARTER, 0, 0, -1.0), (0.8, 0, 0, 0)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("teleavatar.mp4"), help=".mp4 video, or .png for a still")
    parser.add_argument("--motion", choices=("wave", "hold", "drive"), default="wave")
    parser.add_argument("--amplitude", type=float, default=0.3, help="wave amplitude per joint, rad")
    parser.add_argument("--period", type=float, default=4.0, help="wave period, s")
    parser.add_argument("--duration", type=float, help="simulated seconds (default 8; 'drive': the whole path, 12 s)")
    parser.add_argument("--camera", default="overview", help="camera name from the model, or 'free' for the default view")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--show-collision", action="store_true",
                        help="also draw the collision geometry: on the vendor model grey copies of the visual meshes "
                             "that cover parts of the robot in the wrong colour (as the vendor viewer shows them; about "
                             "2x slower), on the new model its convex collision pieces")
    parser.add_argument("--no-shadows", action="store_true",
                        help="quick previews only: about 2x faster, but changes the images, so never use it for "
                             "policy training or evaluation data")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    if not args.show_collision:
        # The collision geoms reuse the full-detail visual meshes (671k triangles) and sit in group 0 with the floor.
        # Move them to group 3, which is hidden by default. That only affects drawing: collisions use contype/conaffinity.
        collision = (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & (model.geom_group == 0)
        model.geom_group[collision] = 3
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise RuntimeError("model does not contain the required 'home' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    if args.camera == "free":
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
    else:
        camera = args.camera
    renderer = mujoco.Renderer(model, args.height, args.width)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = not args.no_shadows
    option = mujoco.MjvOption()
    option.geomgroup[3] = args.show_collision  # the new model keeps its collision pieces in group 3
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.suffix.lower() in (".png", ".jpg", ".jpeg"):
        renderer.update_scene(data, camera=camera, scene_option=option)
        imageio.imwrite(args.output, renderer.render())
        renderer.close()
        print(f"wrote {args.output.resolve()} (home keyframe, camera {args.camera})")
        return

    base = None
    if args.motion == "drive":
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_x") < 0:
            parser.error("--motion drive needs the new model, which has a drivable base")
        sys.path.insert(0, str(DEFAULT_MODEL.parent))
        from mobile_base import MobileBase
        base = MobileBase(model)
        ends = np.cumsum([leg[0] for leg in DRIVE])
    if args.duration is None:
        args.duration = float(ends[-1]) if base is not None else 8.0
    home = model.key_ctrl[home_id].copy()
    low, high = model.actuator_ctrlrange.T
    joints = model.actuator_trnid[:, 0]
    arm = np.array([bool(ARM_JOINT.match(model.joint(j).name)) for j in joints])
    phase = 2 * np.pi * (np.arange(model.nu) % 7) / 7  # stagger joints 1..7 within each arm
    grippers = [model.actuator(name).id for name in ("left_gripper", "right_gripper")
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0]
    qadr = model.jnt_qposadr[joints]
    frames, next_frame, max_error = 0, 0.0, 0.0
    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8) as writer:
        while data.time < args.duration:
            data.ctrl[:] = home
            if args.motion == "wave":
                ramp = min(1.0, data.time / 1.0)  # ease in over the first second
                wave = home + ramp * args.amplitude * np.sin(2 * np.pi * data.time / args.period + phase)
                data.ctrl[arm] = np.clip(wave, low, high)[arm]
                for index in grippers:  # open (1) and close (0) once per period, starting open
                    data.ctrl[index] = 0.5 + 0.5 * np.cos(2 * np.pi * data.time / args.period)
            if base is not None:
                leg = min(int(np.searchsorted(ends, data.time, side="right")), len(DRIVE) - 1)
                base.command(data, *DRIVE[leg][1:] if data.time < ends[-1] else (0, 0, 0))
                base.roll_wheels(data)
            mujoco.mj_step(model, data)
            max_error = max(max_error, float(np.max(np.abs(data.qpos[qadr[arm]] - data.ctrl[arm]))))
            if data.time >= next_frame:
                renderer.update_scene(data, camera=camera, scene_option=option)
                writer.append_data(renderer.render())
                frames += 1
                next_frame += 1.0 / args.fps
    renderer.close()
    print(f"wrote {args.output.resolve()}: {frames} frames, {args.duration:g} s '{args.motion}' at {args.fps} fps; "
          f"max arm |qpos - ctrl| {max_error:.3f} rad; finite state: {bool(np.all(np.isfinite(data.qpos)))}")
    if base is not None:
        x, y, yaw = base.pose(data)
        print(f"base back at x {x:+.4f} m, y {y:+.4f} m, yaw {np.degrees(yaw):+.2f} deg")


if __name__ == "__main__":
    main()
