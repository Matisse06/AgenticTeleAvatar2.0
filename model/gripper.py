"""The real gripper's command, from the vendor's developer docs (§4.4.1).

The robot's /api/{left,right}_gripper/cmd takes a number from 0 to 1 that sets the gripper motor's feedforward torque,
not an opening: 0 opens with the most force (+2.0 N m), 0.1 applies none, and 1 closes with the most force (-1.6 N m),
linearly in between. The model's left_gripper / right_gripper actuators take that torque in N m (positive opens), so
torque(command) turns a robot command into their ctrl:

  data.ctrl[model.actuator("left_gripper").id] = gripper.torque(0.8)  # close firmly

The vendor's openpi code uses the same curve (src/openpi/policies/teleavatar_v2_policy.py). Their deployment opens
the grippers with 0 and grasps with 0.6 to 1; near 0.1 the force is small.
"""
from __future__ import annotations

import numpy as np

OPEN_TORQUE, CLOSE_TORQUE = 2.0, -1.6  # N m at commands 0 and 1
ZERO_COMMAND = 0.1  # the command that applies no torque


def _finite(value, what: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):  # the robot does not check either; the docs forbid NaN and Inf
        raise ValueError(f"gripper {what} must be finite, got {value!r}")
    return array


def _result(array: np.ndarray):
    return float(array) if array.ndim == 0 else array


def torque(command):
    """Motor torque (N m, positive opens) for a robot gripper command, clipped to [0, 1]."""
    c = np.clip(_finite(command, "command"), 0.0, 1.0)
    return _result(np.where(c < ZERO_COMMAND, OPEN_TORQUE * (1.0 - c / ZERO_COMMAND),
                            CLOSE_TORQUE * (c - ZERO_COMMAND) / (1.0 - ZERO_COMMAND)))


def command(torque_nm):
    """The robot command that gives a motor torque (the inverse of torque()), the torque clipped to its range."""
    t = np.clip(_finite(torque_nm, "torque"), CLOSE_TORQUE, OPEN_TORQUE)
    return _result(np.where(t > 0, ZERO_COMMAND * (1.0 - t / OPEN_TORQUE),
                            ZERO_COMMAND + (1.0 - ZERO_COMMAND) * t / CLOSE_TORQUE))
