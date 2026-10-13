# TeleAvatar 2.0: whole-robot MuJoCo model (URDF 20260928-v2)

The whole robot, built from the vendor's newest description (`vendor/urdf/teleavatar_urdf_20260928-v2.urdf`) and their
textured meshes:
- an omni-wheel base: fixed to the floor by default, as in the vendor's simulator, and drivable in an opt-in variant
  that we added (`scene_mobile.xml`)
- a 0.74 m lift carrying the torso
- a head camera
- two 7-joint arms
- parallel grippers with wrist cameras

It replaces the vendor's dual-arm model in `../mujoco/` (kept as delivered) and keeps that simulator's ROS 2 API.

## First run on a machine: unpack, convert, run

The meshes and textures are not in git (590 MB; GitHub refuses files over 100 MB). On a fresh checkout,
`scene.xml` does not load until both steps below have run. Run them in the Python environment from `../README.md`
(Quick start, step 1):

```bash
pip install -r setup/requirements.lock py7zr                    # on the cluster: ./sim.sh uv pip install py7zr
python3 setup/unpack_assets.py urdf_20260825_textured9.21.7z    # -> model/vendor/{visual,collision}/, checksummed
python3 model/convert.py                                        # -> robot.xml, robot_mobile.xml, textures_1024/
python3 -m unittest discover -s model/tests -v                  # 26 tests, about 12 s
```

Download the archive from the robot's documentation page,
https://www.dexteleop.com/docs/teleavatar-2/resources/urdf. Its 9.14, 9.17 and 9.21 versions hold identical
meshes, so any of them works.

## Using the model

- Try it: `python3 model/play.py` opens the MuJoCo viewer with the head and wrist cameras' views and a slider per
  actuator; `--mobile-base` adds keyboard driving (its docstring lists the keys).
- Scenes put this model, unchanged, at a table with objects, with the base fixed or drivable: `../scenes/` (for
  example `python3 model/play.py --model scenes/table_cubes.xml --mobile-base`).
- Python: `mujoco.MjModel.from_xml_path("model/scene.xml")`; keyframe `home`.
- 17 actuators:
  - `armL1`..`armL7`, `armR1`..`armR7` (rad; the first 14, in this order)
  - `lift` (m: 0 is the top, positive lowers the torso, range 0..0.74)
  - `left_gripper`, `right_gripper`: motor torque in N m, -1.6 (closing) to +2.0 (opening). On the robot a 0 to 1
    command sets that torque; `gripper.torque(command)` (`model/gripper.py`) converts it: 0 opens fully, 0.6 to 1
    grasps
- The arm joint ranges lie inside the real robot's software limits (developer docs §4.13): equal on joints 3 to 7,
  narrower on joints 1 and 2 (the note at the top of `../README.md`). Joint 2 stops at about 1.72 rad, where the
  shoulder meets the torso.
- The base is fixed and the wheels are welded by default, as in the vendor's simulator. The drivable base is ours,
  so it is opt-in: `scene_mobile.xml` (`robot_mobile.xml`), loaded by `--mobile-base` in `play.py` and
  `render_video.py` (with `--model`, that file's `_mobile` variant: every scene in `../scenes/` has one). It adds three
  actuators after the 17: `base_x`, `base_y` (m/s) and `base_yaw` (rad/s), the base's velocity in the world frame; 0
  holds the pose it reached. World-frame joints of the same names carry the base (the first three in `qpos`), so it
  does not slip or tip. Drive it in its own frame with `mobile_base.py`, which also turns the three omni wheels at the
  speed of rolling without slip (they do not touch the floor). Every move turns all three: sideways, the rear wheel at
  full speed and the front two at half speed. Without `roll_wheels` the wheels stay still while the base moves:

  ```python
  sys.path.insert(0, "model")  # from the repository root
  from mobile_base import MobileBase
  model = mujoco.MjModel.from_xml_path("model/scene_mobile.xml")
  data = mujoco.MjData(model)
  base = MobileBase(model)
  base.command(data, forward=0.3, left=0.0, turn=0.2)  # m/s, m/s, rad/s; every step (it ramps the speed)
  base.roll_wheels(data)
  mujoco.mj_step(model, data)
  ```
- Cameras:
  - the robot's eyes, one per lens of its three stereo pairs: `head_left_eye`, `head_right_eye` (65 mm apart),
    `left_wrist_left_eye`, `left_wrist_right_eye`, `right_wrist_left_eye`, `right_wrist_right_eye` (50 mm apart).
    The real lenses are fisheyes and their images reach a policy unrectified, so render the eyes through them with
    `cameras.py` (`FisheyeCamera(model, "head_left_eye").render(data)`: 960 x 960 for a head eye, 640 x 400 for a
    wrist eye, as the robot sends them). The head's lens is the vendor's calibration of one head eye; the wrists' is
    our estimate (120 deg across, 75 up and down) until the robot's calibration is known.
  - at each pair's centre, plain pinhole cameras for quick views: `head` (120 deg, the spec), `left_wrist`,
    `right_wrist` (58 deg, nominal). Rendered directly by MuJoCo, they don't look like the robot's images.
  - in the scene: `overview`, `front`, `left_side`
  - The eyes the vendor's pi0.5 policy uses are `cameras.POLICY_EYES`: the head's left eye and each wrist's inner one
    (the left wrist's right eye, the right wrist's left eye).
- Sites (the URDF's reference frames): `left_ee`, `right_ee`, `left_shoulder_base`, `right_shoulder_base`
  (each shoulder's rotation centre), `virtual_base`; `left/right_base_virtual` are aliases of the shoulder bases.
- Joint angles mean the same as in the vendor model and the ROS API: `l_jointN` is `armLN_joint`, `r_jointN` is
  `armRN_joint`.
- Videos and stills: `python3 scripts/render_video.py --output outputs/new_wave.mp4` (`--camera head`, etc.;
  `--mobile-base --motion drive` drives the base).

## ROS 2 simulator

It has the vendor simulator's API: the same topics, FSM and `l_joint1..7` / `r_joint1..7` names. The lift and
grippers hold their home targets (`--lift`, and `--gripper` with the robot's 0 to 1 command, change them), and the
base is fixed: the API has no base.

```bash
./model/run_sim.sh                                   # headless, 200 Hz; --viewer for the MuJoCo viewer
python3 model/smoke_test.py                          # starts and stops its own simulator
python3 mujoco/test_control.py --reordered-names     # the vendor's client, with ./model/run_sim.sh running
```

The vendor's `test_control.py` clips its targets to the vendor model's ranges (its `--model` default). They equal
this model's except joint 1 (+-1.8, wider than this model's -1.2 / 1.2 bound, so this simulator clips the rest) and
joint 7 (+-0.68, narrower). That is harmless for its default, joints 1 and 4 moved 0.12 rad from home.

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

Everything that is not in the vendor's URDF or MuJoCo files is listed here: values we chose, estimated or assumed,
and facts taken from the vendor's online documentation. If the simulation behaves differently from the real robot,
look here first. The constants are in `convert.py` unless noted (the lenses: `cameras.py`); `CLAUDE.md` has the
details.

**From the vendor's files**
- URDF (`teleavatar_urdf_20260928-v2.urdf`, published 2026-10-08; it differs from 20260928 only in the arm joint
  limits): kinematics, masses, inertias, joint ranges, torque limits, the wheel layout and the wheel motors' torque,
  camera mounts (`eye_Link`, `lg_link8`, `rg_link8`) and the reference frames (sites)
- the archive `urdf_20260825_textured9.21.7z`: visual and collision meshes, colour textures, and `manifest.json`
  (the gripper input joint and its range, 0 to 1 rad)
- the damping document (`vendor/各关节阻尼参数.docx`): arm damping and friction measured on the robot (table 2), and
  the gripper input's damping and friction (table 1)
- the vendor's MuJoCo model (`../mujoco/`): arm position gains (kp 120, kv 8), arm armature 0.02, and the `home` arm
  pose, which is also the start pose of their pi0.5 deployment (`zero.py` in github.com/dexteleop/openpi)

**From the vendor's online documentation** (https://www.dexteleop.com/docs/teleavatar-2), not in the files

| fact | used in the model? |
|---|---|
| the robot's software joint limits (§4.13, also their openpi `arm_config.yml`), to which the robot clips arm targets | as a check: the URDF's ranges lie inside them (`API_LIMITS` in `convert.py`, tested); joints 1 and 2 are narrower in the URDF |
| the gripper takes a force command from 0 to 1: +2.0 N m at 0 (opening), 0 at 0.1, -1.6 N m at 1 (closing) (§4.4.1); the URDF's 2 N m effort limit matches | yes: the grippers are torque motors, and `gripper.py` applies this curve |
| the gripper is closed at motor angle 0 and opens as it grows (§4.4.2) | yes, the same direction |
| the head cameras' centre lies between their two eyes, looking 26 deg down (appendix A.3) | yes, it is the URDF's `eye_Link`, where the `head` camera sits |
| the cameras are stereo fisheye pairs, OpenCV fisheye model, calibrated for each robot; raw and recorded images are not rectified (§5.5) | yes: a camera per eye, rendered through its lens (`cameras.py`) |
| the stream: each eye downsampled before stitching (§3.1.2), cut out as 960 x 960 (head) and 640 x 400 (wrist) images (§5.2; their `rtp_video_interface.py` only crops) | yes: the eyes' images have these sizes |
| the cameras' fields of view, from the user manual (https://www.dexteleop.com/docs/user-manual §2.3): head 120 x 120 deg per eye (sensors 1920 x 1920 x 2), wrists 120 x 76 deg (2560 x 800 x 2), 45 Hz | head: the calibration below gives 117.9 x 118.0 deg; wrists: our lens estimate spans 120 x 75 |
| the eyes a pi0.5 policy gets (their openpi `examples/teleavatar_v2/ros2_interface.py`): the head's left eye, the left wrist's right eye, the right wrist's left eye | yes: `cameras.POLICY_EYES`, `play.py`'s insets, `tasks/openpi.py` |
| the base accelerates at up to 0.3 m/s^2 and brakes at up to 0.6 m/s^2 (§4.5) | no, `mobile_base.py` ramps at 1 m/s^2 |

**From the vendor by email** (2026-10-08, not published)

| fact | used in the model? |
|---|---|
| one head eye's calibration, OpenCV fisheye, for 1920 x 1920 images: fx 957.7522723586153, fy 957.058449178608, cx 953.0942242558152, cy 961.8638796212753; k1..k4 -0.021403661461248613, 0.0025639478168715288, -0.0031704737916179947, -0.0018083392738524986 | yes: both head eyes' lens, halved for 960 x 960 (`cameras.HEAD_1920`, `HEAD_EYE`); not the SDK's built-in default (checked against horus_client_demo); which robot and eye it comes from is not said |
| eye sensors: head 1920 x 1920 (the output image uncropped), wrist 1280 x 800 | yes: the stream's images are these halved |
| stereo baselines: head 65 mm, wrist 50 mm | yes: the eye cameras' spacing (the CAD's lens rings are centred 65.0 and 50.0 mm apart) |
| every camera's calibration, extrinsics included, is in the robot's `calibration.json` (`camera_parameters`: `head_stereo_camera`, `left_wrist_stereo_camera`, `right_wrist_stereo_camera`) | not yet: the file is not public; it would replace the wrist estimate and the eyes' assumed placement |

**Chosen, estimated or assumed by us**

| what | value | status |
|---|---|---|
| lift controller | kp 20000, kv 2000, damping 100, armature 0.1 | nominal |
| gripper drive | the documented torque, applied to the input joint; armature 0.001 | assumed: the input joint is the motor's output angle (the docs' q, §4.4.2) |
| gripper stops and linkage | MuJoCo time constant 5 ms (the default is 20 ms) | numerical: with the default, the documented torques stretch them by up to 0.035 rad |
| gripper speed | about 0.1 s for a full stroke (the vendor's damping with the documented torques) | known difference: the URDF lists 2 rad/s (about 0.5 s per stroke), which MuJoCo does not enforce |
| joint 2 toward the body | stops at 1.716 rad (left +, right -), where the shoulder's collision shapes meet the torso's; the CAD meshes touch at 1.72 to 1.73 | known difference: the URDF allows 1.8 and the robot's software 1.9 (a question for the vendor) |
| gripper fingers | finger angle = -0.98 x input | inferred from the joint ranges: the URDF does not link the input to the fingers |
| gripper linkage armature | 0.0002 | numerical stability of the 10 g linkage bars |
| arm joint 4 damping | 0 | the measured value is negative |
| gravity compensation | in the arm, lift and gripper controllers (at home, 0.0000 N m on the gripper input) | assumed (`--no-gravcomp` turns it off) |
| `home` keyframe | lift 0.136 m, grippers open | the lift puts the shoulders 1.27 m up, as in the old model |
| wrist lenses (`cameras.WRIST_EYE`) | an ideal equidistant fisheye (k1..k4 = 0), f = 305.6 px on the 640 x 400 image (120 deg across, 75 up and down), centred | estimated from the spec until the robot's calibration is known |
| head lenses (`cameras.HEAD_EYE`) | the emailed calibration for both eyes (it is one eye's), halved with pixel centres at integers: c' = (c + 0.5) / 2 - 0.5 | assumed: the robot's downscale filter is not published |
| eye placement | the eyes half a baseline either side of the pair's centre camera, along its image x axis, looking the same way | assumed (the per-robot extrinsics are in `calibration.json`) |
| fisheye rendering | up to five 90 deg views (cube faces) with 1 px margin, 960 px square for the robot's image sizes (`cameras.FACE_SIZE`; the image's larger side otherwise), the same in every tool; remapped bilinearly with 8-bit fixed-point weights; pixels past the lens model black (the head image's corners, 0.75%) | ours (`cameras.py`); checked: 25 markers over the head eye's image land within 0.23 px of their projections |
| centre cameras | pinholes: `head` 120 deg (the spec), the wrists 58 deg | nominal, for quick views only |
| wrist cameras | at the centre of the lens face (measured on the mesh), fingers at the bottom of the image | estimated; the orientation is assumed, consistent with the vendor's choice of each wrist's inner eye |
| textures | scaled to 1024 px (the vendor's are 2048) | memory; `--texture-size 0` uses the originals |
| base height | axles 89.7 mm up | measured on the wheel meshes |
| collision shapes | 323 convex pieces (CoACD) | the raw meshes cannot be used (see below) |
| contact exclusions | 63 pairs (66 with the drivable base) | from sweeping the lift and 3000 random poses; `sweep_contacts.py` re-checks them (over v2's ranges on 2026-10-08) |
| simulation settings | timestep 2 ms, `implicitfast`, elliptic friction cones, `impratio` 10 | MuJoCo's advice for grasping |
| drivable base (`robot_mobile.xml` only) | kinematic joints and gains, limits of 1 m/s and 1.5 rad/s, a 1 m/s^2 ramp (`mobile_base.py`), 4.5 m of travel, wheels turned to match | all ours |
| scene (`scene.xml`) | floor, lights, three scene cameras | ours |
| scenes (`../scenes/`) | tables, objects, a sky, scene cameras | ours: listed in `../scenes/README.md` |

## Regenerating derived files

- `assets/collision/` (tracked): convex pieces of the collision meshes, from `build_collision.py`. It needs `coacd`,
  `trimesh` and `fast-simplification`, and takes about 2 minutes. MuJoCo reads at most 200k faces per STL, and the
  vendor's chassis has 472k, so the raw STLs cannot be used.
- `assets/visual_<N>/` (not tracked): from `build_visual_lite.py`, which needs `pymeshlab`.
- The contact exclusions in `convert.py`: `sweep_contacts.py` lists the body pairs that touch in random poses (5 to
  10 s for 10000) or in one pose (`--at joint=value`); `--cad` also measures the CAD gap at each pair's deepest
  overlap, or with `--cad-poses N` at up to N of its poses (about a minute more; needs `scipy` and `trimesh`).
  CLAUDE.md has the 2026-10-08 results (`--poses 10000 --seed 7`).
- `assets/textures_<N>/` (not tracked), and `robot.xml`, `robot_mobile.xml` and `scene_mobile.xml` (tracked): from
  `convert.py`. The tests rebuild the three tracked files and fail if that changed them.
