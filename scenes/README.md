# Scenes: the robot in a world

A scene is a MuJoCo file that puts the robot (`../model/robot.xml`) into a world with things around it: a table,
objects, cameras for watching. `model/` builds the robot; scenes only add around it, so the robot in every scene is
exactly the robot of `model/`. Every scene comes twice: as written, with the base fixed as in the vendor's simulator,
and as its generated twin `<name>_mobile.xml`, with the drivable base (`../model/robot_mobile.xml`). Tasks (a goal, a
success check, randomised layouts) will come later, on top of these files.

| scene (and its twin with the drivable base) | what is in it |
|---|---|
| `table_cubes.xml` (`table_cubes_mobile.xml`) | a 0.75 m table in front of the robot, with a red and a green 4 cm cube on it |

## Using a scene

From the repository root, with the environment active (`source .venv/bin/activate`). Every tool that takes
`--model` loads a scene:

| what | command |
|---|---|
| Viewer, with the robot's camera views | `python3 model/play.py --model scenes/table_cubes.xml` |
| ... with the drivable base (arrows and keypad drive, End stops; `play.py`'s docstring lists the keys) | add `--mobile-base` |
| MuJoCo's own viewer (starts with the arms straight out; Load key 0 for `home`) | `python3 -m mujoco.viewer --mjcf=scenes/table_cubes.xml` |
| ... with the drivable base (its `base_*` sliders; the wheels don't turn there) | `python3 -m mujoco.viewer --mjcf=scenes/table_cubes_mobile.xml` |
| A still of `home` (or a video: `.mp4`; `--mobile-base` works here too) | `python3 scripts/render_video.py --model scenes/table_cubes.xml --output outputs/table_cubes.png` |
| ... what the head camera sees (square, like the robot's head images) | add `--camera head --width 960 --height 960` |
| ROS 2 simulator with the robot's arm API (base fixed: the API has no base) | `./model/run_sim.sh --model scenes/table_cubes.xml` |
| Tests (17, about 8 s) | `python3 -m unittest discover -s scenes/tests -v` |

`--mobile-base` loads the file's twin: `scenes/x.xml` becomes `scenes/x_mobile.xml`, as `model/scene.xml` becomes
`model/scene_mobile.xml`. A model without a twin (the vendor's `mujoco/scene.xml`) is refused, not loaded fixed.

From Python, a scene loads like the robot alone (`mujoco.MjModel.from_xml_path("scenes/table_cubes.xml")`). Its
`home` keyframe is the robot's home pose with every object where the scene puts it, and `play.py`'s Backspace returns
there. An object's body, free joint and geom share its name: `data.body("red_cube").xpos` is its centre,
`data.joint("red_cube").qpos` its position and then orientation quaternion (7 numbers; set them, then call
`mujoco.mj_forward`, to move it). Load the twin to drive the base, with `MobileBase` (`../model/README.md` has the
example): `base_x`, `base_y` and `base_yaw` then come first in `qpos`.

## Adding a scene

```xml
<mujoco model="my_scene">
  <include file="common/physics.xml"/>  <!-- the robot's physics options: first -->
  <include file="common/room.xml"/>     <!-- floor, lights, sky -->
  <asset>
    <model name="teleavatar" file="../model/robot.xml"/>
  </asset>
  <worldbody>
    <attach model="teleavatar" body="base_link" prefix=""/>
    <camera name="overview" .../>  <camera name="front" .../>  <camera name="left_side" .../>
    <!-- props: bodies without joints; objects: bodies with a <freejoint> -->
  </worldbody>
</mujoco>
```

- **Attach the robot; never `<include>` it.** With `<include>`, MuJoCo pads the robot's `home` keyframe with zeros for
  the scene's free bodies, so resetting to `home` would put every object at the world origin, inside the robot's base.
  With `<attach>`, MuJoCo fills them in where the scene puts them. Attaching changes nothing else: compiled, the robot
  is field for field the one of `model/scene.xml`, and 1000 steps of arm motion are bit-identical (checked
  2026-10-02).
- `<attach>` does not bring the robot's physics options along, so `common/physics.xml` repeats them. Its
  `conflict="error"` makes MuJoCo refuse to load a scene whose options differ from `robot.xml`'s, and say which; the
  tests check the same. Update both if `model/convert.py` changes the options.
- Put the scene directly in `scenes/` (the tests look there), and write the robot's path, `../model/robot.xml`, in
  the scene file itself, never in a shared include: MuJoCo 3.14 resolves a `../` path inside an included file wrongly
  when the scene is opened by a relative path such as `scenes/x.xml`. `common/` therefore holds no file paths.
- Every scene has the views `overview` (`render_video.py`'s default camera), `front` and `left_side`.
- Give the scene's model the file's name (`<mujoco model="my_scene">` in `my_scene.xml`), not ending in `_mobile`.
- Run `python3 scenes/build_mobile.py`. It writes the drivable-base twin, `my_scene_mobile.xml`: the same file with
  `../model/robot_mobile.xml` attached. Run it again after every change to the scene, and never edit a twin; the
  tests fail while a twin is missing or out of date, so the drivable base is always there.
- Add a class for the new file at the end of `tests/test_scenes.py` (a test fails until there is one). It inherits
  the checks every scene passes: the robot is unchanged, `home` holds the robot's home and puts each object in place,
  nothing moves at home, and the head camera sees every object. Put the scene's own checks in that class. The twin
  gets a class made from it: the same checks against `robot_mobile.xml`, plus driving the base 0.2 m back.
- List the scene's values in "Where the values come from" below.
- `compiler conflict` and `<attach>` need a recent MuJoCo; this is tested on 3.14.0 only, as the model is.

## `table_cubes.xml`

The robot stands at the origin facing +x. The table's top is at 0.75 m and spans x 0.28 to 0.88 m and y -0.5 to
0.5 m; its front edge is 4 cm ahead of the base's front. The red cube rests at (0.40, +0.05), on the robot's left,
and the green one at (0.40, -0.05). At `home` the grippers hang 13 to 25 cm above the top, and the head camera sees
the table, both cubes and both grippers (each cube is about 14 px wide in a 480 x 480 head image).

With the drivable base, the robot can back away and come back. Driving straight ahead from `home`, its low chassis
passes under the top and the lift column stops against the table's edge after 0.38 m, while the arms pass over the
cubes (they don't move). `render_video.py --motion drive` drives a square meant for the open floor, so here it runs
into the table on its first leg.

The placement comes from a reach study (scratch scripts, 2026-10-02; the results, not the scripts, are here). Its
assumptions:
- **Grasps are a humanoid's**: the gripper comes from the front or the side and its fingers close horizontally on two
  side faces, never from above.
- A level gripper can't grasp a cube on the table: its housing reaches 4.7 to 6.1 cm below the finger line and would
  hit the top. Pitched 15 to 30 deg down, it clears.
- Joint limits: the URDF's ranges intersected with the robot's software limits (the vendor's openpi
  `arm_config.yml`), so joint 3 stays inside the overlap from `../README.md`.

Its results, with the robot at `home` (lift 0.136 m, shoulders 1.27 m up, 0.52 m above the top):
- Each arm grasps over x 0.30 to 0.55 m (0.60 m at the sides), from its own side to just past the centre line. Both
  arms reach a central band: |y| up to 0.15 m at x 0.30 to 0.35 m, 0.10 m at 0.40 to 0.45 m, 0.05 m at 0.50 m. The
  cubes sit in it, where both arms can grasp them at least 0.25 rad from every joint limit (with the best approach
  direction; the straight-ahead picks below come within 0.08 rad).
- Lowering the torso by 5 to 10 cm (lift 0.19 to 0.24 m) gives the arms 20 to 30% more area; it is the same, for the
  arms, as a 0.80 to 0.85 m table. Lower still, the joints run out of margin. Straight-down grasps would come within
  about 0.1 rad of a joint limit, one more reason for side grasps.
- Picks, run in the scene: each arm picks either cube, pitched 30 deg down, and the lift raises it 10 cm (the cube
  rises 99 of 100 mm, the other cube does not move). The cube on the arm's own side is approached straight ahead,
  the other with the approach turned 30 to 45 deg inward; the jaws turn the cube square as they close. Pitched only
  15 deg, the far grasps knock the other cube or let theirs slip. Grasp and motion code is the task's job, still to
  come.

## Where the values come from

Everything in the scenes is ours: nothing comes from the vendor except the robot itself and the joint limits that
the reach study used (above). If the robot behaves differently in a scene than in `model/scene.xml`, look here.

| what | value | status |
|---|---|---|
| table | top at 0.75 m, 0.6 x 1.0 m, 3 cm thick; four 4 x 4 cm legs; front edge at x 0.28 m (the base's front is at 0.243 m); top rgba 0.78 0.74 0.68, legs 0.35 0.35 0.38 | chosen: a standard table height; the edge as close as clears the base; a light top, so the cubes stand out |
| cubes | 4 cm, 64 g, red (rgba 0.85 0.1 0.1) and green (0.1 0.65 0.2) | chosen: the size of ManiSkill's StackCube-v1 cubes (`cube_half_size` 0.02, also red and green); the open jaws (8.65 cm) leave 2.3 cm either side. 64 g is water's density, MuJoCo's default |
| cube positions | red (0.40, +0.05), green (0.40, -0.05), resting on the top | chosen from the reach study above |
| contacts | MuJoCo's defaults for the table and cubes (friction 1, condim 3) | default; the side-grasp picks above work with them |
| room (`common/room.xml`) | `model/scene.xml`'s floor and lights; a sky shading from rgb 0.58 0.6 0.63 overhead to 0.18 0.19 0.2 below | the sky is ours, so the top of the head camera's 120 deg view is not black |
| views | `overview`, `front`, `left_side` | ours, for people; the robot's cameras are `head`, `left_wrist`, `right_wrist` |
| physics (`common/physics.xml`) | the robot's options, repeated | must equal `model/robot.xml`'s (MuJoCo and the tests check) |
| drivable base (the `_mobile` twins) | `model/robot_mobile.xml` as built | ours, not the vendor's: `../model/README.md` lists its values |
