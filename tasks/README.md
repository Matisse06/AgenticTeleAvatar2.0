# Tasks: goals on top of the scenes

A task turns a scene (`../scenes/`) into something a policy can be scored on: seeded starting layouts, a success
check, an instruction, and the observation and action a policy exchanges with it. The first is stacking:
`stack_cubes.py`, on `scenes/table_cubes.xml`.

| file | what |
|---|---|
| `stack_cubes.py` | `StackCubes`: put the red cube on the green one (or `top="green_cube"`); layouts, 30 Hz control, success |
| `openpi.py` | the vendor's pi0.5 interface (their `openpi` repository): the observation dict their client sends, with the three eye images |
| `expert.py` | a scripted expert: humanoid side grasps planned with inverse kinematics, from the cubes' true poses |
| `ik.py` | the grippers' grasp frame and an IK solver for side grasps |
| `run.py` | runs a policy on seeded episodes and records results, per-step logs and videos |
| `tests/test_stack_cubes.py` | 9 tests |

## Running it

From the repository root, with the environment active (`source .venv/bin/activate`):

| what | command |
|---|---|
| The scripted expert on three layouts | `python3 tasks/run.py --policy expert --seeds 0 1 2 --out outputs/tasks/expert` |
| ... with a video per episode (needs a GPU to be quick) | add `--video` |
| A pi0/pi0.5 policy server (the vendor's `serve_policy.py`), on held-out layouts | `python3 tasks/run.py --policy openpi --port 8000 --seeds 100000 100001 --out outputs/tasks/pi05` |
| Tests (9, about 7 s) | `python3 -m unittest discover -s tasks/tests -v` |

`--policy openpi` needs the vendor's client (`pip install openpi-client`). Seeds from 100000 on are for evaluation
only, as in the ManiSkill runs.

From Python:

```python
from tasks.stack_cubes import StackCubes
from tasks.openpi import OpenpiView

task = StackCubes()                     # scenes/table_cubes.xml, base fixed, red on green
view = OpenpiView(task)                 # renders the three eyes the vendor's policy uses
task.reset(seed=0)
request = view.observation()            # the vendor client's dict: state, three images, prompt
obs, info = task.step(action)           # 16 numbers; info["success"], info["on"], info["static"], info["grasped"]
```

## The task

- **Start**: the robot at `home`, the base fixed; each cube somewhere in the band that both arms reach with 0.10 rad
  of joint margin (x 0.32 to 0.55 m ahead; |y| up to 0.20 m near the robot, narrowing to 0.05 m at 0.55 m), at least
  0.10 m from the other, turned at random about the vertical. The draws are seeded (`sample_layout(seed)`).
- **Action**, every 1/30 s: 16 numbers, per arm (left first) the seven absolute joint targets in rad and the gripper's
  motor torque in N m: +2.0 opens with the most force, -1.6 closes with the most (the robot's 0..1 gripper command
  maps to it through `model/gripper.py`; grasps use 0.6 to 1, -0.89 to -1.60 N m). Arm targets are clipped to the
  model's ranges, which lie inside the robot's software limits.
- **Control**: physics (2 ms steps) runs to the next multiple of 1/30 s, 16 or 17 steps, while each arm target ramps
  linearly from the previous one, as the vendor's client interpolates its 30 Hz targets at 200 Hz. The lift holds
  `home`.
- **Observation** (`StackCubes.observation`): what the robot's sensors report, no object poses: per arm the joint
  angles, velocities and torques, the gripper input angle (rad, 0 closed .. 1 open) and torque; the lift; the time.
  Images come from the eye cameras (`model/cameras.py`), rendered through the real lenses' model.
- **Success** (ManiSkill StackCube-v1's checks and thresholds): the top cube's centre within 3.33 cm of the bottom
  one's horizontally and 4.0 +- 0.5 cm above it; moving slower than 1 cm/s and turning slower than 0.5 rad/s; and
  held by neither gripper (a gripper holds it when both fingers push on it with at least 0.5 N, within 85 deg of
  their opening directions). An episode ends at its first successful step, or after 15 s (450 steps).

## The vendor's interface (`openpi.py`)

The vendor's pi0.5 setup for this robot (github.com/dexteleop/openpi, `examples/teleavatar_v2/`) fixes what a policy
sees and returns, and `OpenpiView` reproduces it, so a policy trained with their configs runs here unchanged:
- `observation/state`, 48 numbers: the left arm's joint angles at 0..6, the right arm's at 8..14, then velocities
  (16..30) and torques (32..46); the gripper slots stay 0, as on the robot. Their policy reads only 0..6 and 8..14.
- `observation/images/head_camera` (960 x 960, the head's left eye), `left_color` and `right_color` (400 x 640, each
  wrist's inner eye: the left wrist's right eye, the right wrist's left eye), RGB.
- `prompt`: the instruction, "stack the red cube on the green cube" (their tokenizer keeps the case).
- The answer: 30 actions in the layout above; `run.py` executes 16, then asks again, as their client does.

## The scripted expert (`expert.py`)

It plans one side grasp with IK from the cubes' true poses (it is privileged: no policy gets those), then executes
it open loop. It tries both arms and every approach square to the top cube's faces within 75 deg of straight ahead,
pitched 30 deg down (then 20 and 40 deg, then 20 deg off square); every waypoint, the swings between them and the
carried cube are checked against the table, the other cube and the robot itself, and the plan farthest from the joint
limits wins. Then: to 8 cm behind the cube (via 10 cm above, if the swing there would hit something), straight in,
close (command 0.8), lift 12 cm, carry it across at that height, lower it onto the other cube, open, back off, and
return home.

On 2026-10-08 it stacked red on green in 100 of 100 layouts (seeds 0 to 49 and 100000 to 100049), in 7 to 11 s of
simulated time (median 8.5 s), using each arm about half the time.

## Where the values come from

Nothing here comes from the vendor except what the table says; if a policy behaves differently here than on the
robot, look here.

| what | value | source |
|---|---|---|
| success thresholds | 3.33 cm horizontally (the cube's half diagonal + 5 mm), 4.0 +- 0.5 cm up; 1 cm/s, 0.5 rad/s; 0.5 N per finger within 85 deg | ManiSkill StackCube-v1 (`mani_skill/envs/tasks/tabletop/stack_cube.py`, Panda `is_grasping`); our fingers: links 2, 3, 4, 9 and 5, 6, 7, 10 of each gripper, opening along the jaw axis |
| layout band | x 0.32 to 0.55 m; \|y\| as in `scenes/README.md` (0.10 rad for both arms) | the reach study under v2 (2026-10-08); x from 0.32 so a turned cube stays on the table |
| cube separation | at least 0.10 m, centre to centre | chosen: closer, the open jaws (8.66 cm) graze the other cube |
| cube yaw | uniform, drawn before the positions | chosen (ManiSkill turns its cubes too) |
| settling | 50 physics steps (0.1 s) after placing the cubes | chosen: they sink 0.1 mm into the table's soft contact |
| control | 30 Hz, targets ramped linearly over each period; chunks of 30, 16 executed | the vendor's client (`main.py`: 30 Hz, `open_loop_horizon` 16; `ros2_interface.py`: 200 Hz interpolation); their datasets are recorded at 45 fps |
| episode length | 15 s (450 steps) | chosen: the expert needs 7 to 11 s; the ManiSkill runs allowed 300 steps at 20 Hz |
| instruction | "stack the red cube on the green cube" | chosen (the vendor's prompts are task strings of the same kind, for example "stack the three blocks") |
| grasp point | 2.5 cm behind the fingertips, on the jaws' mid-plane | derived from the finger collision shapes; the depth is chosen so the closed fingers clear the table pitched 30 deg |
| expert motion | pitch 30 deg (20, 40 if needed); 8 cm back, 12 cm lift, 8 cm above the place, 4 mm gap, 8 cm back and 3 cm up after release; 0.1 m/s straight (0.05 lowering), 0.5 rad/s swings lasting at least 0.5 s, 0.3 s pauses before straight approaches; close 0.6 s, open 0.5 s | chosen; the gripper commands 0.8 (close) and 0 (open) are within the vendor's grasp range (developer docs §4.4) |
| expert planning | headings within 75 deg of straight ahead, square to the cube, then 20 deg off square; place headings 0, +-15, +-30, +-45 deg from the grasp's; every joint at least 0.03 rad from its limits; a via point 10 cm above the pre-grasp when the swing there would hit something; a carry swing keeps the cube 3 cm over the other cube within 6 cm of it | chosen; the checks use the model's collision shapes |
| IK (`ik.py`) | damped least squares (damping 0.01, orientation 0.15 m per rad, 150 iterations, steps of at most 0.25 rad), a null-space push of 0.05 away from the limits; solved to 1 mm and 0.01 rad; restarts from home plus 4 perturbations (0.5 rad); along a path, at most 0.15 rad per joint between points (steps split up to 8 times) | chosen |
| gripper geometry (`ik.py`) | pads 4 cm apart at input 0.466 (6 cm at 0.689, 8.66 cm fully open); fingertips 0.2389 m along the approach; jaw mid-plane at -0.0215 m (left) / +0.0215 m (right) in the gripper base frame | derived from the model's finger collision shapes (2026-10-08) |
