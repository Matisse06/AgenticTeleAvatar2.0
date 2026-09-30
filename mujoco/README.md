# TeleAvatar MuJoCo model

This directory is self-contained: the source model is `urdf/urdf20260625/urdf20260625.urdf`, and its 19 STL files are in `urdf/urdf20260625/meshes/`. Generated `robot.xml` references only bundled assets. `scene.xml` adds lighting and a floor.

## Generate and test

```bash
python3 convert_urdf.py --verbose
python3 convert_urdf.py --check-only --verbose
python3 -m unittest discover -s tests -v
python3 smoke_test.py
python3 play.py
```

## ROS 2 simulator

Use the same isolated domain in both terminals. Domain 29 is the production robot domain and is deliberately rejected. Domain 90 is the default; `SIM_ROS_DOMAIN_ID` overrides it.

The smoke test starts and stops its own simulator process, so run it by itself:

```bash
cd /absolute/path/to/mujoco
export SIM_ROS_DOMAIN_ID=90
python3 smoke_test.py
```

To run the simulator manually instead:

```bash
cd /absolute/path/to/mujoco
export SIM_ROS_DOMAIN_ID=90
./run_sim.sh                 # headless, 200 Hz joint states
# ./run_sim.sh --viewer      # optional viewer
```

### Interactive control test

`test_control.py` attaches to an already-running simulator; unlike `smoke_test.py`, it does not start or stop the simulator.

Terminal 1:

```bash
export SIM_ROS_DOMAIN_ID=90
./run_sim.sh --viewer
```

Terminal 2:

```bash
export SIM_ROS_DOMAIN_ID=90
python3 test_control.py --reordered-names
```

The client latches the current joint state, moves joints 1 and 4 by a small relative offset, verifies READY and convergence, returns to the latched start, then disables to PAUSE. Useful options:

```bash
python3 test_control.py --arms left --joints 1 --amplitude 0.08
python3 test_control.py --joints 2,5 --duration 5 --hold 2
python3 test_control.py --dry-run
```

The amplitude is limited to ±0.5 rad and targets are clipped to the model control ranges. Both control scripts refuse production domain 29 and force localhost-only ROS discovery.

`run_sim.sh` and `smoke_test.py` force `ROS_LOCALHOST_ONLY=1`. `smoke_test.py` inherits the selected non-29 domain rather than choosing a different one.

The simulator subscribes to:

- `/api/fsm/enable` (`std_msgs/Float32`)
- `/api/left_arm/joint_cmd` and `/api/right_arm/joint_cmd` (`sensor_msgs/JointState`)

It publishes:

- `/left_arm/joint_states` and `/right_arm/joint_states` at 200 Hz
- `/fsm_state` (`std_msgs/Int32`) at 20 Hz
- `/api/current_mode` (`std_msgs/String`, JSON) at 1 Hz

The minimal FSM is `PAUSE=0`, `SLOW_START=1`, `READY=2`. READY requires valid commands for both arms and actual MuJoCo `qpos` within tolerance for consecutive 20 Hz checks. Heartbeat timeout or disable returns to PAUSE. Commands require seven finite positions; non-empty names must be the seven expected unique names and are reordered before model `ctrlrange` clipping.

The simulator intentionally does not implement end-effector pose, gripper, chassis, lift, or IK interfaces.

## Model scope

The torso is fixed. Each arm has seven actuated revolute joints; the gripper links remain fixed for appearance. The 14 position actuators use each URDF joint range as their control range. The `home` keyframe contains the nominal dual-arm home targets. Contact exclusions are base_link-Link-L1, base_link-Link-R1, Link-L5-Link-L7, and Link-R5-Link-R7.

All timestep, solver, damping (`0.5`), armature (`0.02`), actuator gain (`kp=120`, `kv=8`), force range (`-150..150`), gravity, contact, inertia and friction behavior are nominal simulation parameters. They are not calibrated physical-robot or hardware-control values.

`play.py` accepts `--model`, `--speed`, `--duration`, and `--paused`. In the viewer, Space toggles pause, Backspace restores the home keyframe, mouse controls the camera, and Esc closes the window.
