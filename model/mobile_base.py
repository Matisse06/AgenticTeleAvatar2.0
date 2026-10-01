"""Drive the TeleAvatar base: velocity commands in the robot's own frame, and wheels that roll with it.

The base moves kinematically (BASE_JOINTS in convert.py). World-frame joints base_x, base_y and base_yaw carry it, and
their actuators (same names) take world-frame velocity commands, clipped per axis to 1 m/s and 1.5 rad/s, and hold the
reached pose while the command is 0. The wheels do not touch the floor: roll_wheels, called before each mj_step, spins
them at the speed of an omni wheel that rolls without slipping (sideways, its rollers turn freely).

    base = MobileBase(model)
    base.command(data, forward=0.3, turn=0.2)  # m/s, rad/s in the robot's frame; repeat it every step
    base.roll_wheels(data)
    mujoco.mj_step(model, data)
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

JOINTS = ("base_x", "base_y", "base_yaw")
WHEELS = ("wheel1_wheel_joint", "wheel2_wheel_joint", "wheel3_wheel_joint")
# NOMINAL: command() changes the velocity at most this fast (m/s^2, rad/s^2), like a base controller's ramp. The
# actuators' force limits alone would allow 4 m/s^2.
ACCELERATION, TURN_ACCELERATION = 1.0, 2.0


class MobileBase:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.qadr = np.array([model.joint(name).qposadr[0] for name in JOINTS])
        self.dadr = np.array([model.joint(name).dofadr[0] for name in JOINTS])
        self.actuators = np.array([model.actuator(name).id for name in JOINTS])
        self.actadr = model.actuator_actadr[self.actuators]
        self.wheel_dofs = np.array([model.joint(name).dofadr[0] for name in WHEELS])
        self.radius = float(model.numeric("wheel_radius").data[0])
        self.ramp = model.opt.timestep * np.array([ACCELERATION, ACCELERATION, TURN_ACCELERATION])
        self.velocity = np.zeros(3)  # the command in effect, in the robot's frame
        centres, rolling = [], []
        for name in WHEELS:
            joint = model.joint(name)
            body = joint.bodyid[0]  # a child of base_link, so its pos and quat are in the base frame
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.body_quat[body])
            axle = rotation.reshape(3, 3) @ model.jnt_axis[joint.id]
            centres.append(model.body_pos[body][:2])
            rolling.append(np.cross(axle, [0.0, 0.0, 1.0])[:2])  # where a positive wheel speed rolls the wheel
        self.centres, self.rolling = np.array(centres), np.array(rolling)
        # Wheel speeds from the base velocity (forward, left, turn): a wheel centre moves at v + turn * z x centre, and
        # rolling without slip turns the wheel by that velocity's component along its rolling direction / radius.
        self.speed_matrix = np.column_stack([self.rolling, self.centres[:, 0] * self.rolling[:, 1]
                                             - self.centres[:, 1] * self.rolling[:, 0]]) / self.radius

    def pose(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """x, y (m) and yaw (rad) of base_link in the world."""
        x, y, yaw = data.qpos[self.qadr]
        return float(x), float(y), float(yaw)

    def place(self, data: mujoco.MjData, x: float, y: float, yaw: float) -> None:
        """Move the robot to a pose and make the actuators hold it there (e.g. after mj_resetDataKeyframe)."""
        data.qpos[self.qadr] = data.act[self.actadr] = (x, y, yaw)
        data.qvel[self.dadr] = 0.0
        self.velocity[:] = 0.0

    def command(self, data: mujoco.MjData, forward: float = 0.0, left: float = 0.0, turn: float = 0.0) -> None:
        """Velocity command in the robot's frame: m/s forward and to the left, rad/s counterclockwise.

        Call it once per step: the velocity ramps towards the command (ACCELERATION), and the actuators take
        world-frame velocities, which change as the robot turns.
        """
        self.velocity += np.clip(np.array([forward, left, turn]) - self.velocity, -self.ramp, self.ramp)
        forward, left, turn = self.velocity
        yaw = data.qpos[self.qadr[2]]
        cos, sin = math.cos(yaw), math.sin(yaw)
        data.ctrl[self.actuators] = (cos * forward - sin * left, sin * forward + cos * left, turn)

    def wheel_speeds(self, data: mujoco.MjData) -> np.ndarray:
        """Speeds (rad/s) at which the three wheels roll without slipping at the base's current velocity."""
        vx, vy, turn = data.qvel[self.dadr]
        yaw = data.qpos[self.qadr[2]]
        cos, sin = math.cos(yaw), math.sin(yaw)
        return self.speed_matrix @ (cos * vx + sin * vy, -sin * vx + cos * vy, turn)  # velocity in the base frame

    def roll_wheels(self, data: mujoco.MjData) -> None:
        data.qvel[self.wheel_dofs] = self.wheel_speeds(data)
