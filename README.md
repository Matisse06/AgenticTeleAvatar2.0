# AgenticTeleAvatar2.0

> **Known discrepancy, waiting for the vendor's answer (2026-10-01).** For joint 3 of both arms, the limits in the
> vendor's newest URDF, which the simulator uses, disagree with the limits the real robot's software enforces
> (vendor developer docs, §4.13). The left arm's joint 3 may turn from -85 to +180 degrees in the URDF, but from -149
> to +74 degrees on the robot; the right arm is the mirror image. So in simulation joint 3 can reach angles the robot
> refuses, and it stops short of some the robot allows. Until the vendor confirms which is right, keep joint 3 within
> what both allow: left -1.48 to 1.30 rad (-85 to +74 degrees), right -1.30 to 1.48 rad. Joints 2 and 6 differ
> slightly (the robot allows up to 0.2 rad more than the URDF). The simulated cameras are also simplified (see
> [The model](#the-model)).

A MuJoCo simulation of the TeleAvatar 2.0 robot: two 7-joint arms with parallel grippers, a 0.74 m lift carrying the
torso, a head camera and two wrist cameras, and an omni-wheel base. The base is fixed by default, as in the vendor's
simulator; a drivable base, which we added, is opt-in (`--mobile-base`). A ROS 2 simulator speaks the robot's joint
API, so clients written for the real arms run against it unchanged.

## Quick start

Tested with Python 3.10 and MuJoCo 3.14.0 on Ubuntu 22.04.

1. Create a Python environment and install the packages:

   ```bash
   python3 -m venv --system-site-packages .venv   # if this fails on Ubuntu: sudo apt install python3-venv
   source .venv/bin/activate                      # again in every new terminal
   pip install -r setup/requirements.lock py7zr
   ```

   `--system-site-packages` keeps the system's Python packages visible: ROS 2's `rclpy` needs them.
2. Download the robot's meshes. They are not in git (590 MB unpacked; GitHub refuses files over 100 MB). Get
   `urdf_20260825_textured9.21.7z` (118.6 MB) from the robot's documentation page,
   **https://www.dexteleop.com/docs/teleavatar-2/resources/urdf** (the 9.14 and 9.17 archives hold the same meshes).
3. Unpack it and build the model:

   ```bash
   python3 setup/unpack_assets.py /path/to/urdf_20260825_textured9.21.7z   # meshes into model/vendor/, checksummed
   python3 model/convert.py                                                 # textures, and model/robot.xml
   ```

4. Try it: `python3 model/play.py`

On the FASRC cluster, skip step 1 and run the other commands through `./sim.sh`, which provides ROS 2 Humble and the
Python environment (add py7zr with `./sim.sh uv pip install py7zr`; see `CLAUDE.md`).

Don't unpack the archive into the repository by hand: it has its own `README.md`, which would replace this one.

## Commands

From the repository root, with the environment active (`source .venv/bin/activate`):

| what | command |
|---|---|
| Interactive viewer: the robot's camera views and a slider per actuator | `python3 model/play.py` |
| ... with the drivable base (keyboard driving) | `python3 model/play.py --mobile-base` |
| MuJoCo's own viewer | `python3 -m mujoco.viewer --mjcf=model/scene.xml` |
| ... with the drivable base (its `base_*` sliders; the wheels don't turn there) | `python3 -m mujoco.viewer --mjcf=model/scene_mobile.xml` |
| Video or still, rendered offscreen (`--motion hold\|wave`, `--camera head`, ...) | `python3 scripts/render_video.py --output outputs/wave.mp4` |
| ... the base driving around a square | `python3 scripts/render_video.py --mobile-base --motion drive --output outputs/drive.mp4` |
| ROS 2 simulator with the robot's arm API (`--viewer` to watch it) | `./model/run_sim.sh` |
| The vendor's ROS client against it | `python3 mujoco/test_control.py --reordered-names` |
| Unit tests (21, about 10 s) | `python3 -m unittest discover -s model/tests -v` |

`play.py` is MuJoCo's viewer with additions: it starts in the `home` pose, shows what the head and wrist cameras see,
follows the robot, and with `--mobile-base` drives the base from the keyboard and turns the wheels to match. Backspace
returns home, Space pauses, and `[` and `]` switch the view to the robot's cameras. With `--mobile-base`, Up and Down
also change the driving speed, Left and Right the turn rate, keypad 4 and 6 move sideways, and End stops.

MuJoCo's own viewer runs the model and nothing else. It starts, and resets with Backspace, with every joint at 0 (arms
straight out); for the home pose, set Key to 0 in its Simulation panel and click Load key. The sliders are in its
Control panel. With `scene_mobile.xml`, `base_x`, `base_y` and `base_yaw` set the base's velocity in the world frame
(m/s, rad/s; 0 holds the pose), and the wheels stay still.

The ROS commands need ROS 2 (Humble or later): `run_sim.sh` sources it; before the others, run
`source /opt/ros/<distro>/setup.bash`. The simulator uses ROS domain 90 with localhost-only discovery and refuses
domain 29, which is the production robot's.

From Python (headless machines: set `MUJOCO_GL=egl` before importing mujoco):

```python
import mujoco

model = mujoco.MjModel.from_xml_path("model/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
data.ctrl[model.actuator("lift").id] = 0.4      # lift target in m below the top: lowers the torso
for _ in range(1000):                           # 2 s
    mujoco.mj_step(model, data)
renderer = mujoco.Renderer(model, 480, 640)
renderer.update_scene(data, camera="left_wrist")
image = renderer.render()                       # what the left wrist camera sees, 480 x 640 x 3
```

## The model

`model/convert.py` builds it from the vendor's newest URDF (`model/vendor/urdf/teleavatar_urdf_20260928.urdf`) and
textured meshes:
- From the vendor: kinematics, masses, inertias, joint ranges and torque limits (URDF); arm joint damping and
  friction, measured on the robot.
- Joint angles mean the same as in the robot's ROS API: `l_jointN` is `armLN_joint`, `r_jointN` is `armRN_joint`.
- 17 actuators: the 14 arm joints (rad); `lift` (m, 0 is the top); `left_gripper` and `right_gripper` (0 closed, 1
  open). The drivable-base variant adds `base_x`, `base_y` and `base_yaw` (the base's velocity).
- Collision uses simplified convex shapes made from the vendor's meshes (`model/assets/collision/`, in git).
- Gains and the gripper coupling are nominal: the vendor doesn't publish them.
- The cameras are simplified: each is a single undistorted (pinhole) camera with a nominal 58 degree field of view.
  The real robot has stereo fisheye pairs (head 960 x 960 per eye, wrists 640 x 400), calibrated for each robot, so
  simulated images don't look exactly like real ones. Account for that before training a policy for the real robot on
  them.

**The base and wheels.** By default (`model/scene.xml`) the base is fixed to the floor and the wheels don't turn, as
in the vendor's simulator. We added a drivable base ourselves, so for safety it is opt-in: `--mobile-base` in
`play.py` and `render_video.py`, or load `model/scene_mobile.xml`. There the base moves kinematically: three
world-frame joints (`base_x`, `base_y`, `base_yaw`) carry it, and their actuators take velocity commands and hold the
reached pose at 0. The three omni wheels are hinges that `model/mobile_base.py` spins at the speed of rolling without
slip; they don't touch the floor, so the base can't slip or tip. With three wheels 120 degrees apart, every move turns
all of them: driving sideways spins the rear wheel at full speed and the front two at half speed, while their rollers
slide along their axles. A unit test checks each wheel against MuJoCo's own kinematics. The robot's API has no base
interface, so the ROS simulator uses the fixed base.

## Repository

- `model/`: the model, its tools and tests. `model/README.md` has the details.
- `mujoco/`: the vendor's original dual-arm simulator, kept as delivered.
- `setup/`: environment, checks, and `unpack_assets.py`. `scripts/render_video.py`: videos and stills.
- `CLAUDE.md`: cluster setup (FASRC), every check, and notes on rendering and the model.

Git keeps the code, the small vendor files (URDFs, the joint damping document, the mesh checksums) and the collision
shapes. The collision shapes (`model/assets/collision/`, 323 `.obj` files, 5 MB) are ours, not the vendor's: convex
pieces that `model/build_collision.py` cuts from the vendor's meshes. They can't be downloaded, and regenerating them
needs extra packages (CoACD) and may not give identical pieces on another machine, so git keeps them and everyone
simulates the same contacts. Renders and logs (`outputs/`, `logs/`) stay local. The old model in `mujoco/` keeps its
meshes in git, because they are not on the website.
