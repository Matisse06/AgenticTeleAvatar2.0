# TeleAvatar 2.0: whole-robot MuJoCo model (URDF 20260928)

The whole robot, built from the vendor's newest description (`vendor/urdf/teleavatar_urdf_20260928.urdf`) and their
textured meshes:
- an omni-wheel base that drives on the floor
- a 0.74 m lift carrying the torso
- a head camera
- two 7-joint arms
- parallel grippers with wrist cameras

It replaces the vendor's dual-arm model in `../mujoco/` (kept as delivered) and keeps that simulator's ROS 2 API.

## First run on a machine: unpack, convert, run

The meshes and textures are not in git (590 MB; GitHub refuses files over 100 MB). On a fresh checkout,
`scene.xml` does not load until both steps below have run:

```bash
pip install py7zr                                               # on the cluster: ./sim.sh uv pip install py7zr
python3 setup/unpack_assets.py urdf_20260825_textured9.21.7z    # -> model/vendor/{visual,collision}/, checksummed
python3 model/convert.py                                        # -> model/robot.xml + model/assets/textures_1024/
python3 -m unittest discover -s model/tests -v                  # 19 tests, about 10 s
```

Download the archive from the robot's documentation page,
https://www.dexteleop.com/docs/teleavatar-2/resources/urdf. Its 9.14, 9.17 and 9.21 versions hold identical
meshes, so any of them works.

## Using the model

- Try it: `python3 model/play.py` opens the MuJoCo viewer with the head and wrist cameras' views, keyboard driving
  and a slider per actuator (its docstring lists the keys).
- Python: `mujoco.MjModel.from_xml_path("model/scene.xml")`; keyframe `home`.
- 20 actuators:
  - `armL1`..`armL7`, `armR1`..`armR7` (rad; the first 14, in this order)
  - `lift` (m: 0 is the top, positive lowers the torso, range 0..0.74)
  - `left_gripper`, `right_gripper` (0 = closed .. 1 = open)
  - `base_x`, `base_y` (m/s), `base_yaw` (rad/s): the base's velocity in the world frame. 0 holds the pose it reached.
- The base moves kinematically: world-frame joints `base_x`, `base_y`, `base_yaw` carry it (the first three in
  `qpos`), so it does not slip or tip. Drive it in its own frame with `mobile_base.py`, which also turns the three
  omni wheels at the speed of rolling without slip (they do not touch the floor). Without its `roll_wheels`, the
  wheels stay still while the base moves:

  ```python
  sys.path.insert(0, "model")  # from the repository root
  from mobile_base import MobileBase
  base = MobileBase(model)
  base.command(data, forward=0.3, left=0.0, turn=0.2)  # m/s, m/s, rad/s; every step (it ramps the speed)
  base.roll_wheels(data)
  mujoco.mj_step(model, data)
  ```
- Cameras:
  - on the robot: `head`, `left_wrist`, `right_wrist`
  - in the scene: `overview`, `front`, `left_side`
- Sites (the URDF's reference frames): `left_ee`, `right_ee`, `left_shoulder_base`, `right_shoulder_base`
  (each shoulder's rotation centre), `virtual_base`; `left/right_base_virtual` are aliases of the shoulder bases.
- Joint angles mean the same as in the vendor model and the ROS API: `l_jointN` is `armLN_joint`, `r_jointN` is
  `armRN_joint`.
- Videos and stills: `python3 scripts/render_video.py --output outputs/new_wave.mp4` (`--camera head`, etc.;
  `--motion drive` drives the base).

## ROS 2 simulator

It has the vendor simulator's API: the same topics, FSM and `l_joint1..7` / `r_joint1..7` names. The lift,
grippers and base hold their home targets (`--lift`, `--gripper` change them): the API has no base.

```bash
./model/run_sim.sh                                   # headless, 200 Hz; --viewer for the MuJoCo viewer
python3 model/smoke_test.py                          # starts and stops its own simulator
python3 mujoco/test_control.py --reordered-names     # the vendor's client, with ./model/run_sim.sh running
```

The vendor's `test_control.py` reads its joint ranges from the vendor model (its `--model` default), so it clips its
targets to the old, narrower ranges. That is harmless for its default joints 1 and 4.

## Converter options (`model/convert.py`)

| option | effect |
|---|---|
| `--texture-size N` | colour textures scaled to N (default 1024, 132 MB in memory; 2048 is the vendor's, about 4x; 0 = vendor files) |
| `--visual-faces N` | lighter visual meshes, after `python3 model/build_visual_lite.py N` (see below) |
| `--no-gravcomp` | no gravity compensation in the joint controllers |
| `--damping empirical` | the damping document's empirical table instead of the measured one |
| `--lift-home H` | lift position of the `home` keyframe (default 0.136 m) |
| `--check-only` | build into a temporary file and compile it |

`--visual-faces 50000` cuts the drawn triangles from 3.5M to 1.15M and the model from 831 to 395 MB. CPU rendering
gets 2.7x faster, and at most 0.12% of a frame's pixels change by more than 40/255. Use one setting for a whole
dataset (training and evaluation alike).

## Where the values come from

| from the vendor | nominal or assumed (see the constants in `convert.py`) |
|---|---|
| kinematics, masses, inertias, joint ranges and torque limits (URDF) | position gains (`kp` 120, `kv` 8, the vendor model's), armature 0.02 |
| arm joint damping and friction, measured (`vendor/各关节阻尼参数.docx`, table 2) | gripper coupling: finger angle = -0.98 x input |
| gripper input joint and range (`manifest.json`), its damping and friction (docx table 1) | gravity compensation in the joint controllers |
| visual meshes and colour textures; collision meshes (made convex, see below) | camera fields of view (58 deg), wrist camera image orientation |
| camera mounts (`eye_Link`, `lg_link8` / `rg_link8`) | lift home 0.136 m (shoulders 1.27 m up, like the old model) |
| wheel layout; base force limits, from the wheel motors' torque | kinematic base: its gains, speed limits and ramp |

## Regenerating derived files

- `assets/collision/` (tracked): convex pieces of the collision meshes, from `build_collision.py`. It needs `coacd`,
  `trimesh` and `fast-simplification`, and takes about 2 minutes. MuJoCo reads at most 200k faces per STL, and the
  vendor's chassis has 472k, so the raw STLs cannot be used.
- `assets/visual_<N>/` (not tracked): from `build_visual_lite.py`, which needs `pymeshlab`.
- `assets/textures_<N>/` (not tracked) and `robot.xml` (tracked): from `convert.py`. The tests rebuild `robot.xml`
  and fail if that changed it.
