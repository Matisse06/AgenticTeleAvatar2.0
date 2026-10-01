# AgenticTeleAvatar2.0

A MuJoCo simulation of the TeleAvatar 2.0 robot: two 7-joint arms with parallel grippers, a 0.74 m lift carrying the
torso, a head camera and two wrist cameras, and an omni-wheel base that drives on the floor. A ROS 2 simulator speaks
the robot's joint API, so clients written for the real arms run against it unchanged.

## Quick start

Tested with Python 3.10 and MuJoCo 3.14.0 on Ubuntu 22.04.

1. Install the Python packages: `pip install -r setup/requirements.lock py7zr`
2. Download the robot's meshes. They are not in git (590 MB unpacked; GitHub refuses files over 100 MB). Get
   `urdf_20260825_textured9.21.7z` (118.6 MB) from the robot's documentation page,
   **https://www.dexteleop.com/docs/teleavatar-2/resources/urdf** (the 9.14 and 9.17 archives hold the same meshes).
3. Unpack it and build the model:

   ```bash
   python3 setup/unpack_assets.py /path/to/urdf_20260825_textured9.21.7z   # meshes into model/vendor/, checksummed
   python3 model/convert.py                                                 # textures, and model/robot.xml
   ```

4. Try it: `python3 model/play.py`

On the FASRC cluster, run the same commands through `./sim.sh`, which provides ROS 2 Humble and the Python
environment (see `CLAUDE.md`).

Don't unpack the archive into the repository by hand: it has its own `README.md`, which would replace this one.

## Commands

| what | command |
|---|---|
| Interactive viewer: the robot's camera views, keyboard driving, a slider per actuator | `python3 model/play.py` |
| MuJoCo's own viewer (load the `home` key in its Simulation panel; the wheels don't roll there) | `python3 -m mujoco.viewer --mjcf=model/scene.xml` |
| Video or still, rendered offscreen (`--motion hold\|wave\|drive`, `--camera head`, ...) | `python3 scripts/render_video.py --output outputs/wave.mp4` |
| ROS 2 simulator with the robot's arm API (`--viewer` to watch it) | `./model/run_sim.sh` |
| The vendor's ROS client against it | `python3 mujoco/test_control.py --reordered-names` |
| Unit tests (19, about 10 s) | `python3 -m unittest discover -s model/tests -v` |

In `play.py`: Up and Down change the driving speed, Left and Right the turn rate, keypad 4 and 6 move sideways, End
stops, Backspace returns home and Space pauses. `[` and `]` switch the view to the robot's cameras.

The ROS simulator needs ROS 2 (Humble or later). It uses ROS domain 90 with localhost-only discovery and refuses domain
29, which is the production robot's.

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
- 20 actuators: the 14 arm joints (rad); `lift` (m, 0 is the top); `left_gripper` and `right_gripper` (0 closed, 1
  open); `base_x`, `base_y`, `base_yaw` (the base's velocity).
- Collision uses simplified convex shapes made from the vendor's meshes (`model/assets/collision/`, in git).
- Gains, the gripper coupling and the camera fields of view are nominal: the vendor doesn't publish them.

**The base and wheels.** In the URDF the three omni wheels are free-turning joints. In the simulation the base moves
kinematically: three world-frame joints (`base_x`, `base_y`, `base_yaw`) carry it, and their actuators take velocity
commands and hold the reached pose at 0. The wheels are hinges that `model/mobile_base.py` spins at the speed of
rolling without slip, and they don't touch the floor. So the base can't slip or tip. That is an assumption: the
robot's API has no base interface, and the ROS simulator keeps the base still.

## Repository

- `model/`: the model, its tools and tests. `model/README.md` has the details.
- `mujoco/`: the vendor's original dual-arm simulator, kept as delivered.
- `setup/`: environment, checks, and `unpack_assets.py`. `scripts/render_video.py`: videos and stills.
- `CLAUDE.md`: cluster setup (FASRC), every check, and notes on rendering and the model.

Git keeps the code, the small vendor files (URDFs, the joint damping document, the mesh checksums) and the collision
shapes. Renders and logs (`outputs/`, `logs/`) stay local. The old model in `mujoco/` keeps its meshes in git, because
they are not on the website.
