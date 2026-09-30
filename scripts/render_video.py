#!/usr/bin/env python3
"""Render a TeleAvatar rollout to MP4 (or the home pose to PNG) offscreen on the CPU, to open in VS Code.

  cd ~/TeleAvatar2.0 && ./sim.sh python3 scripts/render_video.py --output outputs/wave.mp4
  ./sim.sh python3 scripts/render_video.py --output outputs/home.png       # one still of the home keyframe

Motions: 'wave' swings every joint sinusoidally around the home keyframe (clipped to the actuator ranges);
'hold' commands the home keyframe, which shows how well the position actuators hold the arms against gravity.
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering, no display needed; must be set before importing mujoco

from pathlib import Path

import imageio
import mujoco
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "mujoco" / "scene.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("teleavatar.mp4"), help=".mp4 video, or .png for a still")
    parser.add_argument("--motion", choices=("wave", "hold"), default="wave")
    parser.add_argument("--amplitude", type=float, default=0.3, help="wave amplitude per joint, rad")
    parser.add_argument("--period", type=float, default=4.0, help="wave period, s")
    parser.add_argument("--duration", type=float, default=8.0, help="simulated seconds")
    parser.add_argument("--camera", default="overview", help="camera name from the model, or 'free' for the default view")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--show-collision", action="store_true",
                        help="also draw the collision meshes, as the vendor viewer does: grey copies of the visual "
                             "meshes that cover parts of the robot in the wrong colour; about 2x slower")
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
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.suffix.lower() in (".png", ".jpg", ".jpeg"):
        renderer.update_scene(data, camera=camera)
        imageio.imwrite(args.output, renderer.render())
        renderer.close()
        print(f"wrote {args.output.resolve()} (home keyframe, camera {args.camera})")
        return

    home = model.key_ctrl[home_id].copy()
    low, high = model.actuator_ctrlrange.T
    phase = 2 * np.pi * (np.arange(model.nu) % 7) / 7  # stagger joints 1..7 within each arm
    frames, next_frame, max_error = 0, 0.0, 0.0
    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8) as writer:
        while data.time < args.duration:
            if args.motion == "wave":
                ramp = min(1.0, data.time / 1.0)  # ease in over the first second
                data.ctrl[:] = np.clip(home + ramp * args.amplitude * np.sin(2 * np.pi * data.time / args.period + phase),
                                       low, high)
            else:
                data.ctrl[:] = home
            mujoco.mj_step(model, data)
            max_error = max(max_error, float(np.max(np.abs(data.qpos - data.ctrl))))
            if data.time >= next_frame:
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
                frames += 1
                next_frame += 1.0 / args.fps
    renderer.close()
    print(f"wrote {args.output.resolve()}: {frames} frames, {args.duration:g} s '{args.motion}' at {args.fps} fps; "
          f"max |qpos - ctrl| {max_error:.3f} rad; finite state: {bool(np.all(np.isfinite(data.qpos)))}")


if __name__ == "__main__":
    main()
