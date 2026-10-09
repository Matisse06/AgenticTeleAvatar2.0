"""A scripted expert for StackCubes: side grasps planned with inverse kinematics, from the cubes' true poses.

It shows that the task can be done with humanoid grasps, and makes demonstrations. It is privileged: it reads the
cubes' poses from the simulation, which no policy gets. Plan (made once, at the first query):
1. Choose the arm and the approach. Candidates, tried in rounds until one gives a plan: each arm approaching the top
   cube square to its faces (the four headings 90 deg apart, those within MAX_HEADING, 75 deg, of straight ahead)
   pitched 30 deg down; then the same at 20 and 40 deg; then headings OFF_SQUARE (20 deg) off square, whose closing
   jaws turn the cube square. Each candidate tries place headings 0, 15, 30 and 45 deg either side of the grasp's.
   Every waypoint is solved with IK, and a path is followed from point to point without jumping to another arm
   configuration (ik.track). A plan is kept if, at every waypoint and on the joint-space swings between them
   (sampled), the robot touches neither itself (the shoulder meets the torso before joint 2's limit, the forearm can
   meet it reaching across), nor the table, nor the other cube, the carried cube touches nothing but its gripper,
   and every joint stays MIN_MARGIN (0.03 rad) from its limits. The plan farthest from the limits wins.
2. Move to 8 cm behind the cube along the approach (or, if the swing there would hit something, to 10 cm above that
   and straight down), then straight in, 1 cm per IK step; close the gripper (command 0.8, -1.24 N m) for 0.6 s; lift
   12 cm; carry the cube straight across at that height to above the other one, turning to the place heading on the
   way (a joint-space swing that keeps it 3 cm over the other cube, if no straight carry is possible); lower it
   straight onto it (4 mm above contact); open the gripper (command 0, +2.0 N m) for 0.5 s; back off 8 cm along the
   approach and 3 cm up; return the arm home. The other arm holds home. Straight moves run at 0.1 m/s (0.05 m/s
   lowering), joint moves at 0.5 rad/s (at least 0.5 s), with a 0.3 s pause before each straight approach so the arm
   catches up with its targets.

  expert = StackExpert(task)
  chunk = expert.infer(observation, step)   # (30, 16) actions from `step` on; call again after executing some
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .ik import INPUT_4CM, SIDES, ArmIK, Solution, grasp_rotation
from .stack_cubes import CONTROL_HZ, CUBE_HALF, GRIPPER_BASE, StackCubes

PITCH = math.radians(30)          # CHOSEN: 15 deg lets far grasps slip or knock the other cube; level hits the table
BACK, LIFT, ABOVE, PLACE_GAP, VIA = 0.08, 0.12, 0.08, 0.004, 0.10  # m
STEP = 0.01                       # m between IK solutions on straight moves
JOINT_SPEED, MIN_MOVE, SETTLE = 0.5, 0.5, 0.3  # rad/s, s, s
MIN_MARGIN = 0.03                 # rad from every joint limit (the position servo presses into a limit otherwise)
MAX_HEADING = math.radians(75)    # approach headings considered, either side of straight ahead
OFF_SQUARE = math.radians(20)     # how far off square to the cube's faces the last-resort approaches turn
CLOSE, OPEN = -1.244, 2.0         # gripper torques (N m): robot commands 0.8 and 0
CLOSE_TIME, OPEN_TIME = 0.6, 0.5  # s
CHUNK = 30


def _yaw(quat) -> float:
    return 2.0 * math.atan2(quat[3], quat[0])


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class _GripperSetter:
    """Sets a gripper's input joint and the linkage joints tied to it by equality constraints (in dependency order)."""

    def __init__(self, model: mujoco.MjModel, side: str) -> None:
        prefix = side[0] + "g_"
        self.input = model.jnt_qposadr[model.joint(prefix + "joint1").id]
        rules = [(model.jnt_qposadr[model.eq_obj1id[e]], model.jnt_qposadr[model.eq_obj2id[e]], model.eq_data[e, :5])
                 for e in range(model.neq) if int(model.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT)
                 and model.joint(model.eq_obj1id[e]).name.startswith(prefix)]
        self.rules = sorted(rules, key=lambda rule: rule[1] != self.input)  # the coupling to the input first

    def __call__(self, qpos: np.ndarray, value: float) -> None:
        qpos[self.input] = value
        for target, source, c in self.rules:
            x = qpos[source]
            qpos[target] = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])))


class StackExpert:
    def __init__(self, task: StackCubes, pitch: float = PITCH) -> None:
        self.task, self.pitch = task, pitch
        model = task.model
        self.ik = {side: ArmIK(model, side) for side in SIDES}
        self.check_data = mujoco.MjData(model)
        self.grippers = {side: _GripperSetter(model, side) for side in SIDES}
        robot_root = model.body("base_link").id
        robot = {b for b in range(model.nbody) if self._under(model, b, robot_root)}
        self.robot_geoms = set(np.flatnonzero(np.isin(model.geom_bodyid, list(robot))))
        self.table_geoms = set(np.flatnonzero(model.geom_bodyid == model.body("table").id))
        hand = lambda side, g: model.body(model.geom_bodyid[g]).name.startswith(side[0] + "g_")
        self.hand_geoms = {side: {g for g in self.robot_geoms if hand(side, g)} for side in SIDES}
        self.plan, self.choice, self.rejected = None, None, []

    @staticmethod
    def _under(model, body: int, root: int) -> bool:
        while body != 0:
            if body == root:
                return True
            body = model.body_parentid[body]
        return False

    def reset(self) -> None:
        self.plan, self.choice, self.rejected = None, None, []

    def infer(self, observation: dict, step: int) -> np.ndarray:
        if self.plan is None:
            self.plan = self._make_plan()
        index = np.minimum(np.arange(step, step + CHUNK), len(self.plan) - 1)
        return self.plan[index]

    # -- planning
    def _collides(self, side: str, q: np.ndarray, opening: float, ignore: set, carrying: bool = False,
                  stacked: bool = False):
        """What the robot, with this arm at q and its gripper input at `opening`, touches (a pair of body names), or
        None: the table, a cube other than in `ignore`, or itself (its contact exclusions aside). Carrying, the top cube
        sits upright at the grasp point, square to the jaws, and must touch nothing but this gripper."""
        task, m, d = self.task, self.task.model, self.check_data
        d.qpos[:] = task.data.qpos
        d.qpos[self.ik[side].qadr] = q
        self.grippers[side](d.qpos, opening)
        top = task.cube_geom[task.top]
        if stacked:  # the top cube already placed: on the bottom one, as the plan leaves it
            bottom = task.cube_pose(task.bottom)[0]
            adr = m.jnt_qposadr[task.cube_joint[task.top]]
            d.qpos[adr:adr + 3] = bottom + [0, 0, 2 * CUBE_HALF + PLACE_GAP]
        if carrying:  # held from the side, the cube stays upright, turned square to the (horizontal) jaw axis
            point, rotation = self.ik[side].forward(q)
            yaw = math.atan2(rotation[1, 2], rotation[0, 2])
            adr = m.jnt_qposadr[task.cube_joint[task.top]]
            d.qpos[adr:adr + 7] = [*point, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        mujoco.mj_fwdPosition(m, d)
        others = self.table_geoms | ({task.cube_geom[c] for c in task.cube_geom} - ignore)
        for contact in d.contact[:d.ncon]:
            pair = {contact.geom1, contact.geom2}
            robot = pair & self.robot_geoms
            hit = robot and (pair & others or len(robot) == 2)  # the table, a cube, or itself (exclusions aside)
            if carrying and top in pair:
                hit = not robot or not robot <= self.hand_geoms[side]  # the held cube: only its gripper may touch it
            if hit:
                return tuple(sorted(m.body(m.geom_bodyid[g]).name for g in pair))
        return None

    def _line(self, ik: ArmIK, start: np.ndarray, end: np.ndarray, rotation: np.ndarray, seed: np.ndarray):
        """IK solutions every STEP along a straight move of the grasp point; None if the arm cannot follow it."""
        count = max(1, int(math.ceil(np.linalg.norm(end - start) / STEP)))
        points = [start + (end - start) * k / count for k in range(1, count + 1)]
        return self._path(ik, points, [rotation] * count, seed)

    def _path(self, ik: ArmIK, points: list, rotations: list, seed: np.ndarray):
        """IK solutions at each point (with its rotation), each tracked from the previous one so that the arm moves
        continuously (a step it cannot follow is split in two, down to an eighth); None if the arm cannot follow."""
        solutions, q, previous = [], np.asarray(seed, dtype=float), None
        for point, rotation in zip(points, rotations):
            solution = ik.track(point, rotation, q)
            if not solution.ok and previous is not None:
                for parts in (2, 4, 8):
                    q_try = q
                    for k in range(1, parts + 1):
                        mid_point = previous[0] + (np.asarray(point) - previous[0]) * k / parts
                        solution = ik.track(mid_point, rotation, q_try)  # (the rotation of the end point)
                        if not solution.ok:
                            break
                        q_try = solution.q
                    if solution.ok:
                        break
            if not solution.ok:
                return None
            solutions.append(solution)
            q, previous = solution.q, (np.asarray(point), rotation)
        return solutions

    def _move_clear(self, side: str, q_from, q_to, opening: float, stage: str, stacked: bool = False,
                    samples: int = 12) -> bool:
        """Whether a joint-space move touches nothing on the way (sampled); records why not in self.rejected."""
        for k in range(1, samples + 1):
            q = q_from + (q_to - q_from) * k / samples
            touching = self._collides(side, q, opening, set(), stacked=stacked)
            if touching:
                self.rejected.append((side, stage, touching))
                return False
        return True

    def _clear(self, side: str, solutions, opening: float, ignore: set, stage: str, carrying: bool = False,
               stacked: bool = False) -> bool:
        """Whether every solution keeps MIN_MARGIN and touches nothing; records why not in self.rejected."""
        for solution in solutions:
            if solution.margin < MIN_MARGIN:
                self.rejected.append((side, stage, f"margin {solution.margin:.3f}"))
                return False
            touching = self._collides(side, solution.q, opening, ignore, carrying, stacked)
            if touching:
                self.rejected.append((side, stage, touching))
                return False
        return True

    def _swing(self, ik: ArmIK, start, goal_point, goal_rotation, bottom: np.ndarray, samples: int = 12):
        """A joint-space move from `start` (a Solution) to the pose over the place, sampled: each sample becomes a
        Solution (for the clearance checks); None if the goal has no IK or the held cube would pass lower than 3 cm
        above the other cube's top."""
        goal = ik.solve(goal_point, goal_rotation, seeds=[start.q])
        if not goal.ok:
            return None
        solutions = []
        for k in range(1, samples + 1):
            q = start.q + (goal.q - start.q) * k / samples
            point = ik.forward(q)[0]
            over = np.hypot(*(point[:2] - bottom[:2])) < 0.06
            if over and point[2] < bottom[2] + CUBE_HALF + 0.03 + CUBE_HALF:
                return None
            solutions.append(Solution(q, True, ik.margin(q), 0.0, 0.0))
        return solutions

    def _candidate(self, side: str, heading: float, pitch: float):
        task, ik = self.task, self.ik[side]
        top = task.cube_pose(task.top)[0]
        bottom = task.cube_pose(task.bottom)[0]
        top_geom = {task.cube_geom[task.top]}
        grasp_rot = grasp_rotation(heading, pitch)
        approach = grasp_rot[:, 0]
        start = task.data.qpos[ik.qadr].copy()
        pre = ik.solve(top - BACK * approach, grasp_rot, seeds=[start])
        if not pre.ok:
            self.rejected.append((side, "pre", "no IK"))
            return None
        if not self._clear(side, [pre], 1.0, set(), "pre"):
            return None
        approach_path = []  # straight down from a point above the pre-grasp, if the swing to it would hit something
        if not self._move_clear(side, start, pre.q, 1.0, "swing to pre"):
            via = ik.solve(top - BACK * approach + [0, 0, VIA], grasp_rot, seeds=[pre.q, start])
            if not via.ok or not self._move_clear(side, start, via.q, 1.0, "swing to via"):
                return None
            approach_path = self._line(ik, top - BACK * approach + [0, 0, VIA], top - BACK * approach, grasp_rot,
                                       via.q)
            if approach_path is None or not self._clear(side, [via, *approach_path], 1.0, set(), "via"):
                return None
            approach_path = [via, *approach_path]
            pre = approach_path[-1]
        inward = self._line(ik, top - BACK * approach, top, grasp_rot, pre.q)
        if inward is None:
            self.rejected.append((side, "inward", "no IK"))
            return None
        if not self._clear(side, inward, 1.0, top_geom, "inward"):  # the jaws may touch the cube they come for
            return None
        lift = self._line(ik, top, top + [0, 0, LIFT], grasp_rot, inward[-1].q)
        # carrying: the gripper holds the cube 4 cm wide (input INPUT_4CM); the cube itself is not checked
        if lift is None:
            self.rejected.append((side, "lift", "no IK"))
            return None
        # the first 3 cm: the cube still where it was (it would touch the table in the checker); then carried
        if not (self._clear(side, lift[:3], INPUT_4CM, top_geom, "lift")
                and self._clear(side, lift[3:], INPUT_4CM, top_geom, "lift", carrying=True)):
            return None
        best = None
        place = bottom + [0, 0, 2 * CUBE_HALF + PLACE_GAP]
        carry_height = max(top[2] + LIFT, place[2] + ABOVE)
        for turn in np.radians([0, -15, 15, -30, 30, -45, 45]):
            place_rot = grasp_rotation(heading + turn, pitch)
            begin, end = top + [0, 0, LIFT], np.array([place[0], place[1], carry_height])
            count = max(1, int(math.ceil(np.linalg.norm(end[:2] - begin[:2]) / (2 * STEP))))
            points = [begin + (end - begin) * k / count for k in range(1, count + 1)]
            rotations = [grasp_rotation(heading + turn * k / count, pitch) for k in range(1, count + 1)]
            if carry_height > place[2] + ABOVE:
                points.append(place + [0, 0, ABOVE]); rotations.append(place_rot)
            carry = self._path(ik, points, rotations, lift[-1].q)
            if carry is None:  # no straight carry: a joint-space swing to above the place, if it stays high and clear
                carry = self._swing(ik, lift[-1], place + [0, 0, ABOVE], place_rot, bottom)
            if carry is None:
                self.rejected.append((side, "carry", "no IK"))
                continue
            if not self._clear(side, carry, INPUT_4CM, top_geom, "carry", carrying=True):
                continue
            down = self._line(ik, place + [0, 0, ABOVE], place, place_rot, carry[-1].q)
            if down is None:
                self.rejected.append((side, "down", "no IK"))
                continue
            if not self._clear(side, down, INPUT_4CM, top_geom, "down", carrying=True):
                continue
            back = place - BACK * place_rot[:, 0] + [0, 0, 0.03]
            retreat = self._line(ik, place, back, place_rot, down[-1].q)
            if retreat is None:
                self.rejected.append((side, "retreat", "no IK"))
                continue
            if not self._clear(side, retreat, 1.0, top_geom, "retreat", stacked=True):
                continue
            if not self._move_clear(side, retreat[-1].q, start, 1.0, "swing home", stacked=True):
                continue
            solutions = [*approach_path, pre, *inward, *lift, *carry, *down, *retreat]
            margin = min(s.margin for s in solutions)
            if best is None or margin > best[0]:
                best = (margin, dict(side=side, heading=heading, pitch=pitch, turn=float(turn), pre=pre.q,
                                     approach=[s.q for s in approach_path],
                                     inward=[s.q for s in inward], lift=[s.q for s in lift],
                                     carry=[s.q for s in carry], down=[s.q for s in down],
                                     retreat=[s.q for s in retreat], home=start))
        return best

    def _make_plan(self) -> np.ndarray:
        task = self.task
        top_yaw = _yaw(task.cube_pose(task.top)[1])
        faces = [_wrap(top_yaw + k * math.pi / 2) for k in range(4)]
        # in order of preference: square to the cube's faces at the default pitch, other pitches, then turned 20 deg
        # off square (the closing jaws turn the cube square)
        rounds = [([h for h in faces if abs(h) <= MAX_HEADING], [self.pitch])]
        rounds.append((rounds[0][0], [math.radians(20), math.radians(40)]))
        rounds.append(([_wrap(h + s) for h in faces for s in (-OFF_SQUARE, OFF_SQUARE)
                        if abs(_wrap(h + s)) <= MAX_HEADING], [self.pitch, math.radians(20), math.radians(40)]))
        for headings, pitches in rounds:
            options = [found for side in SIDES for heading in headings for pitch in pitches
                       if (found := self._candidate(side, heading, pitch)) is not None]
            if options:
                break
        else:
            raise RuntimeError(f"no plan for seed {task.seed} ({task.layout}); rejected: {self.rejected}")
        margin, choice = max(options, key=lambda option: option[0])
        self.choice = {"side": choice["side"], "heading_deg": math.degrees(choice["heading"]),
                       "pitch_deg": math.degrees(choice["pitch"]), "place_turn_deg": math.degrees(choice["turn"]),
                       "margin": margin}
        return self._actions(choice)

    def _actions(self, plan: dict) -> np.ndarray:
        task, side = self.task, plan["side"]
        hold = {s: task.data.ctrl[task.arm[s]].copy() for s in SIDES}
        rows = []

        def emit(q, torque, seconds=None):
            count = 1 if seconds is None else max(1, round(seconds * CONTROL_HZ))
            for _ in range(count):
                row = np.zeros(16)
                for i, s in enumerate(SIDES):
                    row[8 * i:8 * i + 7] = q if s == side else hold[s]
                    row[8 * i + 7] = torque if s == side else OPEN
                rows.append(row)

        def move(q_from, q_to, torque):
            seconds = max(MIN_MOVE, np.abs(q_to - q_from).max() / JOINT_SPEED)
            count = max(1, round(seconds * CONTROL_HZ))
            for k in range(1, count + 1):
                s = k / count
                emit(q_from + (q_to - q_from) * (3 * s * s - 2 * s * s * s), torque)  # smoothstep

        def straight(points, torque, speed=0.1, spacing=STEP):
            per_point = max(1, round(spacing / speed * CONTROL_HZ))  # 0.1 m/s, 1 cm apart: 3 steps each
            for q in points:
                for _ in range(per_point):
                    emit(q, torque)

        if plan["approach"]:  # via a point above the pre-grasp
            move(plan["home"], plan["approach"][0], OPEN)
            straight(plan["approach"][1:], OPEN)
        else:
            move(plan["home"], plan["pre"], OPEN)
        emit(plan["pre"], OPEN, SETTLE)
        straight(plan["inward"], OPEN)
        emit(plan["inward"][-1], CLOSE, CLOSE_TIME)
        straight(plan["lift"], CLOSE)
        straight(plan["carry"], CLOSE, spacing=2 * STEP)
        emit(plan["carry"][-1], CLOSE, SETTLE)
        straight(plan["down"], CLOSE, speed=0.05)
        emit(plan["down"][-1], OPEN, OPEN_TIME)
        straight(plan["retreat"], OPEN)
        move(plan["retreat"][-1], plan["home"], OPEN)
        return np.array(rows)
