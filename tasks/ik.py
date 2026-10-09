"""Side grasps for TeleAvatar's arms: the grasp frame of each gripper and an inverse kinematics solver for it.

The robot grasps like a humanoid: the gripper comes from the front or the side, pitched down, and its fingers close
horizontally on two side faces of an object; never from above (a level gripper's housing would hit the table).

Grasp frame (derived from the model's finger collision shapes on 2026-10-08; the same for both grippers): in the
gripper base body (lg_base_link / rg_base_link, the body of arm joint 7), +x is the approach axis (wrist to
fingertips), +z the jaw axis (finger A, ?g_link3, on the +z side) and -y the wrist camera's side. The pads are 4 cm
apart at gripper input 0.466 (8.66 cm fully open at 1), and the fingertips then reach x = 0.2389 m. The grasp point,
where a held 4 cm cube's centre sits, is on the jaws' mid-plane (z = -0.0215 m left, +0.0215 m right) 2.5 cm behind the
fingertips (CHOSEN: pitched 30 deg down, the closed fingers then clear the table under the cube by about 5 mm).

  ik = ArmIK(model, "left")                                  # its own MjData, at the model's 'home'
  rotation = grasp_rotation(heading, pitch=math.radians(30))  # gripper base orientation for a side grasp
  solution = ik.solve(position, rotation, seeds=[q_now])      # Solution(q, ok, margin, ...): the best of many starts
  step = ik.track(next_position, rotation, solution.q)         # the next point of a path, without leaving the branch

The solver is damped least squares on the grasp point's position and the gripper's orientation, with the arm's joint
limits enforced (an active set) and a push away from them in the null space; it tries the seeds in turn and keeps
the converged solution farthest from every limit. Only the seven arm joints move; the lift and the other arm stay as
the MjData has them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

SIDES = ("left", "right")
ARM_JOINTS = {side: [f"arm{side[0].upper()}{i}_joint" for i in range(1, 8)] for side in SIDES}
GRIPPER_BASE = {"left": "lg_base_link", "right": "rg_base_link"}
FINGERTIP_X = 0.2389       # fingertips along the approach at a 4 cm gap (m, gripper base frame)
JAW_MID_Z = {"left": -0.0215, "right": 0.0215}
GRASP_DEPTH = 0.025        # CHOSEN: the cube's centre this far behind the fingertips
INPUT_4CM = 0.466          # gripper input (lg_joint1 / rg_joint1, rad) at which the pads are 4 cm apart
INPUT_6CM = 0.689          # ... 6 cm apart (fully open: 1.0, 8.66 cm)


def grasp_point(side: str, depth: float = GRASP_DEPTH) -> np.ndarray:
    """The grasp point in the gripper base frame."""
    return np.array([FINGERTIP_X - depth, 0.0, JAW_MID_Z[side]])


def grasp_rotation(heading: float, pitch: float) -> np.ndarray:
    """World orientation of a gripper base for a side grasp: approaching along `heading` (rad, about world z; 0 is the
    robot's forward, +x) and pitched `pitch` down, jaws horizontal, wrist camera on top. Columns: the base's axes."""
    approach = np.array([math.cos(pitch) * math.cos(heading), math.cos(pitch) * math.sin(heading), -math.sin(pitch)])
    jaw = np.array([-math.sin(heading), math.cos(heading), 0.0])
    return np.column_stack([approach, np.cross(jaw, approach), jaw])


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Axis times angle of a rotation matrix."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation).ravel())
    quat = quat if quat[0] >= 0 else -quat
    sine = np.linalg.norm(quat[1:])
    return 2.0 * quat[1:] if sine < 1e-12 else (2.0 * math.atan2(sine, quat[0]) / sine) * quat[1:]


@dataclass
class Solution:
    q: np.ndarray          # the seven arm joints (rad)
    ok: bool               # converged, within the joint limits
    margin: float          # the smallest distance of any joint to its nearer limit (rad)
    position_error: float  # m
    rotation_error: float  # rad


class ArmIK:
    """IK of one arm's grasp point. data: an MjData to own (default: a new one at 'home')."""

    def __init__(self, model: mujoco.MjModel, side: str, data: mujoco.MjData | None = None,
                 depth: float = GRASP_DEPTH, rotation_weight: float = 0.15) -> None:
        self.model, self.side = model, side
        if data is None:
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        self.data = data
        self.joints = np.array([model.joint(name).id for name in ARM_JOINTS[side]])
        self.qadr = model.jnt_qposadr[self.joints]
        self.low, self.high = model.jnt_range[self.joints].T.copy()
        self.body = model.body(GRIPPER_BASE[side]).id
        self.point = grasp_point(side, depth)
        self.home = model.key("home").qpos[self.qadr].copy()
        self.weight = rotation_weight  # m per rad: how orientation errors count against position errors

    def margin(self, q: np.ndarray) -> float:
        return float(np.minimum(q - self.low, self.high - q).min())

    def forward(self, q: np.ndarray) -> tuple:
        """The grasp point's world position and the gripper base's world orientation at arm angles q."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        rotation = self.data.xmat[self.body].reshape(3, 3).copy()
        return self.data.xpos[self.body] + rotation @ self.point, rotation

    def _project(self, q, position, rotation, iterations=150, damping=0.01, push=0.05, max_step=0.25):
        q = np.clip(np.array(q, dtype=float), self.low, self.high)
        for _ in range(iterations):
            point, current = self.forward(q)
            error_p, error_r = position - point, rotation_vector(rotation @ current.T)
            if np.linalg.norm(error_p) < 1e-4 and np.linalg.norm(error_r) < 1e-3:
                break
            axes, anchors = self.data.xaxis[self.joints], self.data.xanchor[self.joints]
            jacobian = np.vstack([np.cross(axes, point - anchors).T, self.weight * axes.T])
            error = np.concatenate([error_p, self.weight * error_r])
            step, active, fixed = np.zeros(7), np.ones(7, bool), np.zeros(6)
            for _ in range(4):  # joints that would cross a limit are pinned there and the rest re-solved
                part = jacobian[:, active]
                normal = part @ part.T + damping ** 2 * np.eye(6)
                change = part.T @ np.linalg.solve(normal, error - fixed)
                # null-space push toward the middle of the ranges, strongest for the joint nearest a limit
                low, high = q - self.low, self.high - q
                nearest = min(low.min(), high.min())
                pull = (np.exp(-10 * (low - nearest)) - np.exp(-10 * (high - nearest)))[active]
                pull *= push / (np.abs(pull).sum() + 1e-12)
                change += pull - part.T @ np.linalg.solve(normal, part @ pull)
                step[active] = change
                beyond = active & ((q + step < self.low) | (q + step > self.high))
                if not beyond.any():
                    break
                step[beyond] = np.clip(q[beyond] + step[beyond], self.low[beyond], self.high[beyond]) - q[beyond]
                active &= ~beyond
                fixed = jacobian[:, ~active] @ step[~active]
                if not active.any():
                    break
            largest = np.abs(step).max()
            q = np.clip(q + step * min(1.0, max_step / max(largest, 1e-12)), self.low, self.high)
        point, current = self.forward(q)
        error_p = float(np.linalg.norm(position - point))
        error_r = float(np.linalg.norm(rotation_vector(rotation @ current.T)))
        return q, error_p, error_r

    def solve(self, position, rotation, seeds=(), tries: int = 4, rng: np.random.Generator | None = None,
              tolerance=(1e-3, 0.01)) -> Solution:
        """The pose (grasp point at `position`, gripper base at `rotation`) from each seed (then home and random
        perturbations of it): the converged solution with the largest joint margin."""
        rng = rng if rng is not None else np.random.default_rng(0)
        starts = [np.asarray(seed, dtype=float) for seed in seeds] + [self.home]
        starts += [self.home + rng.normal(0.0, 0.5, 7) for _ in range(tries)]
        best = None
        for start in starts:
            q, error_p, error_r = self._project(start, np.asarray(position, float), np.asarray(rotation, float))
            ok = error_p < tolerance[0] and error_r < tolerance[1]
            candidate = Solution(q, ok, self.margin(q), error_p, error_r)
            if best is None or (candidate.ok, candidate.margin) > (best.ok, best.margin):
                best = candidate
            if ok and seeds and start is starts[0] and candidate.margin > 0.2:
                break  # the first seed (usually the previous waypoint) is good: keep the motion continuous
        return best

    def track(self, position, rotation, q, max_jump: float = 0.15, tolerance=(1e-3, 0.01)) -> Solution:
        """The pose solved from q alone, for the next point of a path: not ok if it does not converge or if a joint
        would move more than max_jump (rad), which means the solution left q's branch (the arm would whip)."""
        q = np.asarray(q, dtype=float)
        solution, error_p, error_r = self._project(q, np.asarray(position, float), np.asarray(rotation, float))
        ok = error_p < tolerance[0] and error_r < tolerance[1] and np.abs(solution - q).max() <= max_jump
        return Solution(solution, bool(ok), self.margin(solution), error_p, error_r)
