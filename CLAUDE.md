# TeleAvatar 2.0 MuJoCo simulation

MuJoCo model and ROS 2 simulator of the lab's TeleAvatar 2.0 dual-arm robot, as delivered in `mujoco/` (read
`mujoco/README.md` first), set up to run on FASRC's Cannon cluster. **CPU only: nothing here needs a GPU**, so never
request one for it (FASRC's Job Defense Shield flags idle GPUs; see ~/pi05_maniskill/CLAUDE.md). The goal is testing,
and later training, robot policies, including ones that take camera images: keep rendering faithful (below).

## Layout

- `mujoco/`: the vendor folder, kept as delivered (`mujoco.zip` is the original). The README steps and the unit tests
  rewrite `robot.xml`, but its content comes out identical. Put our own code outside this folder.
- `sim.sh`: runs a command inside the container (ROS 2 Humble + the MuJoCo venv). With no arguments it opens a shell.
- `env.sh`: paths and the pinned image. `container_rc.sh` is sourced inside the container by `sim.sh`.
- `setup/setup_env.sbatch`: one-time setup (pulls the image, builds the venv). `setup/requirements.lock`: exact versions.
- `setup/verify.sh`: every README check plus a render. `setup/viewer_check.sh`: the live viewer under a virtual display.
- `scripts/render_video.py`: offscreen MP4/PNG renders on the CPU.
- `outputs/`: renders and logs (`outputs/verify/` is rewritten by the checks). `logs/`: Slurm logs.

## Environment

- Container: the official `osrf/ros:humble-desktop` image (Ubuntu 22.04, Python 3.10.12, ROS 2 Humble, Mesa 23.2
  OpenGL/EGL with the llvmpipe CPU renderer), pinned by digest in `env.sh`, at
  `$HL/containers/ros2-humble-desktop_2026-09-10.sif` (1.0 GB). That is the stack the vendor scripts target:
  `run_sim.sh` sources `/opt/ros/humble/setup.bash`, and the shipped `.pyc` files are CPython 3.10.
- venv `$HL/envs/teleavatar_ros2`: mujoco 3.14.0, numpy 1.26.4 (must stay below 2: ROS Humble's compiled message
  modules are built against numpy 1.x), imageio 2.38 with a bundled ffmpeg 7.0.2. It is built with the container's
  Python, so it only works inside the container (through `sim.sh`). Adding a package: `./sim.sh uv pip install <pkg>`.
- The model needs MuJoCo 3 or newer: it uses the `kv` actuator attribute, which MuJoCo 2.3.7 (the openpi venv) rejects.
- `sim.sh` passes `--cleanenv`, so host settings (modules, `PYTHONPATH`, `LD_PRELOAD`, `VK_ICD_FILENAMES`) stay out of
  the container. It forces `ROS_LOCALHOST_ONLY=1`, defaults to ROS domain 90 (`SIM_ROS_DOMAIN_ID` overrides it),
  refuses domain 29 (the production robot), and passes the node's timezone (the image defaults to UTC). Python
  bytecode goes to `~/.cache/teleavatar_pycache`, so the vendor's shipped `.pyc` files are never loaded. (They were
  checked on 2026-09-29: five match their sources, and `smoke_test`'s is stale, so Python would recompile it anyway.)
- Nothing on the cluster provided this before: the nodes have no ROS 2, and FASRC's shared SEAS containers
  (`/n/singularity_images/SEAS`) are ROS 1 Melodic and a 2020 mujoco-py image.

## Running

Run on a compute node, not the login node (checks that take a few seconds are fine there). On 2026-09-29 the `test`
partition was down for maintenance until 2026-10-02; `shared` works.

- Check everything (about 3 minutes, most of it the video render):
  `srun -p shared -c 4 --mem 8G -t 20 ~/TeleAvatar2.0/sim.sh bash ~/TeleAvatar2.0/setup/verify.sh`
  and the live viewer: `srun -p shared -c 4 --mem 8G -t 15 bash ~/TeleAvatar2.0/setup/viewer_check.sh`
- Interactive work, with terminals on the same node (ROS discovery is localhost only):
  `salloc -p shared -c 4 --mem 8G -t 2:00:00`, then in more terminals `srun --jobid <jobid> --overlap --pty bash`.
  In each terminal: `cd ~/TeleAvatar2.0/mujoco && ~/TeleAvatar2.0/sim.sh`, then use the README commands as written.
- Videos to open in VS Code: `./sim.sh python3 scripts/render_video.py --output outputs/wave.mp4`
  (`--motion hold|wave`, `--duration`, `--camera`; a `.png` output renders one still of the home keyframe).
  An 8 s clip at 960x720 takes about 80 s on 4 cores.
- Live viewer (`play.py`, `run_sim.sh --viewer`): VS Code has no display, so use an Open OnDemand Remote Desktop
  (https://rcood.rc.fas.harvard.edu, Interactive Apps, Remote Desktop, CPU partition, no GPU) and run the same
  `sim.sh` commands from its terminal. Software rendered, it runs at about 1.7 fps and takes 10 to 20 s to show the
  robot. Known issue: stopped from code (the end of `play.py --duration`, Ctrl-C of `run_sim.sh --viewer`), it usually
  exits with a segfault (139). MuJoCo's passive viewer does not wait for its render thread, and `glfw.terminate()` runs
  at exit while a slow frame is still being drawn. It happens after the scripts' own cleanup, so it is harmless.

## Rendering and camera images

- Every link is in the model twice: the URDF's `<visual>` and `<collision>` use the same STL file at the same origin,
  and `convert_urdf.py` makes a visual geom (URDF colour, group 1, never collides) and a collision geom (contacts,
  MuJoCo's default 0.5 grey, group 0 with the floor). Both are drawn, and the grey copy covers parts of the robot: on
  the home frame 5% of all pixels differ by up to 127/255 (`outputs/verify/collision_compare.png`).
  `render_video.py` therefore moves the collision geoms to hidden group 3 for drawing only (contacts use
  `contype`/`conaffinity`, not the group); `--show-collision` restores the delivered look. The vendor viewer still
  draws both. The proper fix is two lines in the vendor's `convert_urdf.py`, to send them:
  `outputs/compare/convert_urdf_collision_group.patch` (tested on a copy on 2026-09-30: the 14 vendor tests pass, the
  render is pixel-identical to `render_video.py`'s fix, and masses, inertias and 1000 steps of dynamics are
  bit-identical, so it only changes drawing). Until the vendor applies it, the fix lives only in
  `render_video.py`: any other code that renders images (camera observations for policies) must apply it too, so put
  model loading and render settings in one shared helper that all rendering code uses.
- Shadows stay on by default. `--no-shadows` is for quick previews only; it changes the images, so never use it for
  policy training or evaluation data. Use identical render settings for training and evaluation.
- Before/after figures (1600x1200 overview renders made with `render_video.py`) are in `outputs/compare/`. Hiding the
  collision copies changes 4.8% of the frame (4.5% by more than 10/255). Turning shadows off changes 42% of the frame
  slightly (16.5% by more than 10/255): the cast shadows plus shading across the floor and robot.
- CPU rendering (llvmpipe, 4 threads, 960x720, 671k mesh triangles; the base alone is 106k): 578 ms per frame with
  collision copies and shadows (the viewer), 317 ms with visual meshes and shadows (`render_video.py`), 128 ms
  without shadows. Fine for checks and videos; generating training images at scale needs GPU rendering (EGL) in a
  job that keeps the GPU busy.
- The model has no robot cameras: `scene.xml` only has a fixed `overview` camera. Policies with image inputs need head
  and wrist cameras that match the real robot's mounts and intrinsics. Checked on MuJoCo 3.14 (2026-09-30):
  - The head is part of `base_link`, one fixed mesh from the floor (z = -1.27 m) to the top of the head (+0.26 m), so
    a head camera attaches to `base_link`. Wrist cameras attach to `Link-L7` / `Link-R7`. The model has no head
    pan/tilt or lift joints.
  - Cameras can be added at load time without editing vendor files: `spec = mujoco.MjSpec.from_file("scene.xml")`
    (its `<include>` works), `spec.body("Link-L7").add_camera(name=..., pos=..., quat=...)`, `spec.compile()`.
  - Real intrinsics are supported: MJCF `resolution` (W H), `focalpixel` (fx fy) and `principalpixel`, plus
    `sensorsize`, which MuJoCo requires whenever focal/principal are set (it refuses to compile otherwise; any size
    with the image's aspect ratio gives identical images). `MjsCamera` has the same fields. Images are not mirrored:
    camera +x is image right and camera +y is image up. The principal point sign is the opposite of OpenCV's: for a
    real camera's cx, cy (from `camera_info`), use `principalpixel = (W/2 - cx, H/2 - cy)`. Test renders showed +100
    moves the on-axis point to x = 220 and y = 140. The gripper links (`Link-L10/L11`, `Link-R10/R11`) are rigidly
    fixed to `Link-L7`/`Link-R7`, so a wrist camera there moves with the gripper.
  - Gotchas: a ROS optical frame looks along +z with y down, a MuJoCo camera looks along -z with y up (rotate 180
    degrees about x); MuJoCo renders a pinhole camera, with no lens distortion; the vendor ROS simulator publishes
    no images, so a policy fed over ROS needs an image publisher; each image costs about 0.3 s on the CPU.

## Verified

- 2026-09-29 (CPU node holy7c04111, 4 cores): all README steps pass. `convert_urdf.py` (regenerated `robot.xml`
  byte-identical), `--check-only`, the 14 unit tests, `smoke_test.py` (all checks), and
  `test_control.py --reordered-names` (target reached within 0.064 rad, start restored within 0.070 rad).
- 2026-09-30 (holy7c04109): `viewer_check.sh` passes. `test_control.py` also passes with `run_sim.sh --viewer` open,
  but with a thin margin (0.097 rad against its 0.10 tolerance), because the slow viewer slows the simulator.
  `play.py` renders the robot (`outputs/verify/viewer_play.png`).

## Model facts

14 position-controlled revolute joints (`l_joint1..7`, `r_joint1..7`), fixed torso, fixed grippers (appearance only);
65.6 kg in total, no contacts at the home keyframe. Physics runs about 100x real time on one core. Holding home, the
arms sag up to 0.067 rad under gravity (position actuators with `kp=120` and no gravity compensation), which is why
the READY tolerance is 0.08 rad. Damping, armature, gains (`kp=120`, `kv=8`), force limits and contact settings are
nominal simulation values, not calibrated to the real robot. The simulator does not implement end-effector pose,
gripper, chassis, lift or IK APIs.

| joint | range (rad) | home | | joint | range (rad) | home |
|---|---|---|---|---|---|---|
| l_joint1 | -1.80 .. 1.80 | 0.41 | | r_joint1 | -1.80 .. 1.80 | -0.50 |
| l_joint2 | 0.00 .. 1.80 | 1.16 | | r_joint2 | -1.80 .. 0.00 | -0.92 |
| l_joint3 | -2.60 .. 1.30 | -0.47 | | r_joint3 | -1.30 .. 2.60 | 0.52 |
| l_joint4 | 0.25 .. 2.20 | 0.90 | | r_joint4 | -2.20 .. -0.25 | -1.28 |
| l_joint5 | -1.80 .. 1.80 | 0.23 | | r_joint5 | -1.80 .. 1.80 | 0.32 |
| l_joint6 | -1.40 .. 1.40 | -0.15 | | r_joint6 | -1.40 .. 1.40 | 0.55 |
| l_joint7 | -0.68 .. 0.68 | 0.60 | | r_joint7 | -0.68 .. 0.68 | -0.52 |
