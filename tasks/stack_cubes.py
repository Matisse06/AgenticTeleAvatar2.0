"""Stacking on scenes/table_cubes.xml: put the red cube on top of the green one (or the other way round).

  task = StackCubes()                       # the scene, base fixed; top="green_cube" swaps the colours
  obs = task.reset(seed=0)                  # a seeded layout of the two cubes, the robot at 'home'
  while not done:
      obs, info = task.step(action)         # one 30 Hz control step; info["success"] etc.

What a step does (as the vendor's pi0.5 client drives the real robot, examples/teleavatar_v2 in their openpi):
- action: 16 numbers, per arm (left first) the seven absolute joint targets in rad, then the gripper's motor torque in
  N m (+2.0 opens with the most force, -1.6 closes with the most; model/gripper.py converts the robot's 0..1 command).
  Arm targets are clipped to the model's ranges, which lie inside the robot's software limits (§4.13).
- control at 30 Hz: physics (2 ms steps) runs until the next multiple of 1/30 s, 16 or 17 steps, while each arm
  target ramps linearly from the previous one, as the vendor's client interpolates at 200 Hz. The lift holds 'home'.

Layouts (reset): each cube anywhere in the band that both arms reach with 0.10 rad of joint margin (scenes/README.md:
x 0.32 to 0.55 m ahead, |y| narrowing from 0.20 to 0.05 m), at least 0.10 m apart centre to centre, each turned by
a random angle about z. The draws are seeded (numpy default_rng(seed)); seeds 100000 and up are for evaluation.

Success (evaluate), ManiSkill StackCube-v1's three checks, with its thresholds:
- on: the top cube's centre within 0.0333 m of the bottom one's horizontally and 0.04 +- 0.005 m above it
- static: the top cube moving slower than 0.01 m/s and turning slower than 0.5 rad/s
- released: no gripper grasps it. A gripper grasps when both its fingers push on the cube with at least 0.5 N, each
  within 85 deg of that finger's opening direction. Fingers left touching it with no force (a gripper whose torque
  went to 0 and that friction holds still) count as released.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import gripper  # noqa: E402  (model/gripper.py)

SCENE = ROOT / "scenes" / "table_cubes.xml"
CUBES = ("red_cube", "green_cube")
CUBE_HALF = 0.02
TABLE_TOP = 0.75
CONTROL_HZ = 30
SETTLE_STEPS = 50         # physics steps (0.1 s) after placing the cubes, before the episode's clock starts
SIDES = ("left", "right")
ARM_ACTUATORS = {side: [f"arm{side[0].upper()}{i}" for i in range(1, 8)] for side in SIDES}
FINGERS = {"left": ([f"lg_link{k}" for k in (2, 3, 4, 9)], [f"lg_link{k}" for k in (5, 6, 7, 10)]),
           "right": ([f"rg_link{k}" for k in (2, 3, 4, 9)], [f"rg_link{k}" for k in (5, 6, 7, 10)])}
GRIPPER_BASE = {"left": "lg_base_link", "right": "rg_base_link"}
INSTRUCTION = "stack the {top} cube on the {bottom} cube"

# The band both arms reach with 0.10 rad of joint margin (scenes/README.md, re-checked under the v2 limits on
# 2026-10-08): the largest |y| (m) at each distance ahead x (m). Interpolated between these points.
BAND_X = (0.30, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55)
BAND_Y = (0.20, 0.20, 0.175, 0.175, 0.15, 0.15, 0.125, 0.10, 0.075, 0.05)
LAYOUT_X = (0.32, 0.55)   # CHOSEN: from 0.32 m, so a turned cube stays on the table (its front edge is at 0.28 m)
MIN_SEPARATION = 0.10     # CHOSEN: centre to centre; closer, the open jaws (8.66 cm) graze the other cube
# ManiSkill StackCube-v1 (mani_skill/envs/tasks/tabletop/stack_cube.py) thresholds
ON_XY = math.hypot(CUBE_HALF, CUBE_HALF) + 0.005
ON_Z = 0.005
STATIC_SPEED, STATIC_TURN = 0.01, 0.5
GRASP_FORCE, GRASP_ANGLE = 0.5, math.radians(85)


def band_half_width(x: float) -> float:
    return float(np.interp(x, BAND_X, BAND_Y, left=0.0, right=0.0))


def sample_layout(seed: int) -> dict:
    """{cube: (x, y, yaw)}: both cubes in the reach band, MIN_SEPARATION apart, turned at random. The yaws are drawn
    first, so they do not depend on how many positions were rejected."""
    rng = np.random.default_rng(seed)
    yaws = rng.uniform(0.0, 2.0 * math.pi, len(CUBES))
    placed = []
    for _ in range(len(CUBES)):
        for _ in range(1000):
            x = rng.uniform(*LAYOUT_X)
            y = rng.uniform(-BAND_Y[0], BAND_Y[0])
            if abs(y) <= band_half_width(x) and all(math.hypot(x - px, y - py) >= MIN_SEPARATION for px, py in placed):
                placed.append((x, y))
                break
        else:
            raise RuntimeError(f"seed {seed}: no layout found")
    return {cube: (*xy, float(yaw)) for cube, xy, yaw in zip(CUBES, placed, yaws)}


class StackCubes:
    """The task core: the scene, seeded layouts, 30 Hz control, the full state and the success check.

    model: the compiled scene (default: scenes/table_cubes.xml). top: the cube to put on the other one."""

    def __init__(self, model: mujoco.MjModel | None = None, top: str = "red_cube", max_seconds: float = 15.0) -> None:
        if top not in CUBES:
            raise ValueError(f"top must be one of {CUBES}")
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.top, self.bottom = top, next(cube for cube in CUBES if cube != top)
        self.instruction = INSTRUCTION.format(top=self.top.split("_")[0], bottom=self.bottom.split("_")[0])
        self.max_steps = round(max_seconds * CONTROL_HZ)
        m = self.model
        self.arm = {side: np.array([m.actuator(name).id for name in ARM_ACTUATORS[side]]) for side in SIDES}
        self.arm_joints = {side: np.array([m.actuator_trnid[a, 0] for a in self.arm[side]]) for side in SIDES}
        self.gripper = {side: m.actuator(f"{side}_gripper").id for side in SIDES}
        self.gripper_joint = {side: m.joint(f"{side[0]}g_joint1").id for side in SIDES}
        self.lift = m.actuator("lift").id
        self.cube_joint = {cube: m.joint(cube).id for cube in CUBES}
        self.cube_geom = {cube: m.geom(cube).id for cube in CUBES}
        self.finger_geoms = {side: tuple(frozenset(np.flatnonzero(np.isin(m.geom_bodyid, [m.body(b).id for b in f])))
                                         for f in FINGERS[side]) for side in SIDES}
        self.steps = 0
        self.layout = None

    # -- episode
    def reset(self, seed: int, layout: dict | None = None) -> dict:
        """The robot at 'home' and the cubes where `layout` (default: sample_layout(seed)) puts them, at rest."""
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        self.layout = layout if layout is not None else sample_layout(seed)
        for cube, (x, y, yaw) in self.layout.items():
            adr = m.jnt_qposadr[self.cube_joint[cube]]
            d.qpos[adr:adr + 7] = [x, y, TABLE_TOP + CUBE_HALF, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        for _ in range(SETTLE_STEPS):  # let the cubes settle onto the table (they sink 0.1 mm into its soft contact)
            mujoco.mj_step(m, d)
        d.time = 0.0
        self.steps = 0
        self.seed = seed
        return self.observation()

    def step(self, action) -> tuple:
        """One control step (1/30 s) toward `action` (16 numbers, see the module docstring)."""
        action = np.asarray(action, dtype=float)
        if action.shape != (16,) or not np.all(np.isfinite(action)):
            raise ValueError(f"action must be 16 finite numbers, got {action.shape}")
        m, d = self.model, self.data
        start = {side: d.ctrl[self.arm[side]].copy() for side in SIDES}
        goal = {side: np.clip(action[8 * i:8 * i + 7], *m.actuator_ctrlrange[self.arm[side]].T)
                for i, side in enumerate(SIDES)}
        for i, side in enumerate(SIDES):
            d.ctrl[self.gripper[side]] = np.clip(action[8 * i + 7], gripper.CLOSE_TORQUE, gripper.OPEN_TORQUE)
        begin, end = self.steps / CONTROL_HZ, (self.steps + 1) / CONTROL_HZ
        while d.time < end - 0.5 * m.opt.timestep:
            alpha = min(1.0, (d.time + m.opt.timestep - begin) * CONTROL_HZ)
            for side in SIDES:
                d.ctrl[self.arm[side]] = start[side] + alpha * (goal[side] - start[side])
            mujoco.mj_step(m, d)
        self.steps += 1
        info = self.evaluate()
        info["timeout"] = self.steps >= self.max_steps and not info["success"]
        return self.observation(), info

    # -- state
    def observation(self) -> dict:
        """The robot's state as its sensors report it (no object poses): per arm the joint angles (rad), velocities
        (rad/s) and torques (N m, the actuators' and their gravity compensation's), the gripper input angle (rad, 0
        closed .. 1 open) and torque command (N m); the lift (m); the time (s) and step."""
        m, d = self.model, self.data
        obs = {"time": float(d.time), "step": self.steps,
               "lift": float(d.qpos[m.jnt_qposadr[m.actuator_trnid[self.lift, 0]]])}
        for side in SIDES:
            joints = self.arm_joints[side]
            obs[f"{side}_arm"] = d.qpos[m.jnt_qposadr[joints]].copy()
            obs[f"{side}_arm_velocity"] = d.qvel[m.jnt_dofadr[joints]].copy()
            obs[f"{side}_arm_torque"] = d.qfrc_actuator[m.jnt_dofadr[joints]].copy()
            obs[f"{side}_gripper"] = float(d.qpos[m.jnt_qposadr[self.gripper_joint[side]]])
            obs[f"{side}_gripper_torque"] = float(d.ctrl[self.gripper[side]])
        return obs

    def cube_pose(self, cube: str) -> tuple:
        """Ground truth (for experts, logs and checks; not part of the observation): position, quaternion (w x y z),
        linear velocity (world) and angular velocity (body frame)."""
        joint = self.cube_joint[cube]
        q, v = self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint]
        return (self.data.qpos[q:q + 3].copy(), self.data.qpos[q + 3:q + 7].copy(), self.data.qvel[v:v + 3].copy(),
                self.data.qvel[v + 3:v + 6].copy())

    def grasping(self, side: str, cube: str) -> bool:
        """Whether a gripper grasps a cube: both fingers push on it with at least GRASP_FORCE, each within GRASP_ANGLE
        of its opening direction (ManiSkill's Panda check, with the jaws' axis as the opening direction)."""
        m, d = self.model, self.data
        jaw = d.xmat[m.body(GRIPPER_BASE[side]).id].reshape(3, 3)[:, 2]  # finger A opens along +z, B along -z
        force6 = np.zeros(6)
        cube_geom = self.cube_geom[cube]
        for finger, opening in zip(self.finger_geoms[side], (jaw, -jaw)):
            total = np.zeros(3)
            for i in range(d.ncon):
                contact = d.contact[i]
                pair = (contact.geom1, contact.geom2)
                if cube_geom not in pair or not (pair[0] in finger or pair[1] in finger) or contact.efc_address < 0:
                    continue
                mujoco.mj_contactForce(m, d, i, force6)
                world = contact.frame.reshape(3, 3).T @ force6[:3]  # on geom2, from geom1
                total += world if pair[1] in finger else -world
            size = np.linalg.norm(total)
            if size < GRASP_FORCE or math.acos(np.clip(total @ opening / size, -1, 1)) > GRASP_ANGLE:
                return False
        return True

    def evaluate(self) -> dict:
        top, _, speed, turn = self.cube_pose(self.top)
        bottom = self.cube_pose(self.bottom)[0]
        offset = top - bottom
        on = bool(np.linalg.norm(offset[:2]) <= ON_XY and abs(offset[2] - 2 * CUBE_HALF) <= ON_Z)
        static = bool(np.linalg.norm(speed) <= STATIC_SPEED and np.linalg.norm(turn) <= STATIC_TURN)
        grasped = {side: self.grasping(side, self.top) for side in SIDES}
        return {"success": on and static and not any(grasped.values()), "on": on, "static": static,
                "grasped": any(grasped.values()), "grasped_by": [side for side in SIDES if grasped[side]]}
