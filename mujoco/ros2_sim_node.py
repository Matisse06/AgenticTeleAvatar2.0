#!/usr/bin/env python3
"""Headless-by-default MuJoCo ROS 2 simulator for the dual-arm joint API."""
from __future__ import annotations

import argparse
import json
import math
import signal
from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Int32, String

HERE = Path(__file__).resolve().parent
ARM_SIDES = ("left", "right")
JOINT_NAMES = {
    "left": tuple(f"l_joint{i}" for i in range(1, 8)),
    "right": tuple(f"r_joint{i}" for i in range(1, 8)),
}
ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
PAUSE = 0
SLOW_START = 1
READY = 2
FSM_NAMES = {PAUSE: "PAUSE", SLOW_START: "SLOW_START", READY: "READY"}
CURRENT_MODE = {
    "meta_mode": 1,
    "left_arm_control_mode": 1,
    "right_arm_control_mode": 1,
    "enable_left_arm": True,
    "enable_right_arm": True,
    "enable_chassis": False,
    "enable_lift_servo": False,
    "enable_collision_check": False,
    "current_mode": 1,
    "simulator": "mujoco",
}


def normalize_joint_target(
    positions: Sequence[float],
    names: Sequence[str],
    expected_names: Sequence[str],
    ranges: np.ndarray,
) -> Optional[np.ndarray]:
    """Validate, name-order and ctrlrange-clip a seven-joint command."""
    try:
        target = np.asarray(positions, dtype=float)
    except (TypeError, ValueError):
        return None
    if target.shape != (7,) or not np.all(np.isfinite(target)):
        return None
    if names:
        if len(names) != 7 or len(set(names)) != 7 or set(names) != set(expected_names):
            return None
        by_name = dict(zip(names, target))
        target = np.asarray([by_name[name] for name in expected_names], dtype=float)
    return np.clip(target, ranges[:, 0], ranges[:, 1])


def clip_target(values: Sequence[float], ranges: np.ndarray) -> Optional[np.ndarray]:
    """Compatibility helper for unnamed seven-joint targets."""
    return normalize_joint_target(values, (), tuple(str(i) for i in range(7)), ranges)


def slow_start_step(current: np.ndarray, target: np.ndarray, max_delta: np.ndarray) -> np.ndarray:
    """Move a control target toward the requested target by one bounded step."""
    return current + np.clip(target - current, -max_delta, max_delta)


class DualArmMuJoCoSim(Node):
    """Minimal PAUSE/SLOW_START/READY simulator exposing the public joint API."""

    def __init__(
        self,
        model_path: Path = HERE / "scene.xml",
        *,
        publish_hz: float = 200.0,
        slow_start_speed: float = 0.5,
        heartbeat_timeout: float = 1.0,
        ready_tolerance: float = 0.08,
        ready_stable_cycles: int = 5,
    ) -> None:
        super().__init__("teleavatar_mujoco_sim")
        if min(publish_hz, slow_start_speed, heartbeat_timeout, ready_tolerance) <= 0:
            raise ValueError("rates, speeds, timeout and tolerance must be positive")
        if ready_stable_cycles < 1:
            raise ValueError("ready_stable_cycles must be positive")

        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        if (self.model.nq, self.model.nv, self.model.nu) != (14, 14, 14):
            raise RuntimeError("the simulator requires a 14-DoF, 14-actuator model")
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id < 0:
            raise RuntimeError("model does not contain the required 'home' keyframe")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)

        self.ctrlrange = np.asarray(self.model.actuator_ctrlrange, dtype=float).copy()
        self.requested_ctrl = np.asarray(self.data.ctrl, dtype=float).copy()
        self.hold_ctrl = np.asarray(self.data.qpos, dtype=float).copy()
        self.command_seen = {side: False for side in ARM_SIDES}
        self.state = PAUSE
        self.last_heartbeat_ns: Optional[int] = None
        self.heartbeat_timeout_ns = int(heartbeat_timeout * 1e9)
        self.ready_tolerance = ready_tolerance
        self.ready_stable_cycles = ready_stable_cycles
        self.ready_count = 0
        self.slow_max_delta = np.full(14, slow_start_speed / publish_hz)
        self.physics_period = 1.0 / publish_hz
        self.step_remainder = 0.0

        self.state_pubs = {
            side: self.create_publisher(JointState, f"/{side}_arm/joint_states", 10)
            for side in ARM_SIDES
        }
        self.fsm_pub = self.create_publisher(Int32, "/fsm_state", 10)
        self.mode_pub = self.create_publisher(String, "/api/current_mode", 10)
        self.create_subscription(Float32, "/api/fsm/enable", self._enable_callback, 10)
        self.command_subs = [
            self.create_subscription(
                JointState,
                f"/api/{side}_arm/joint_cmd",
                lambda msg, arm=side: self._command_callback(msg, arm),
                10,
            )
            for side in ARM_SIDES
        ]
        self.physics_timer = self.create_timer(1.0 / publish_hz, self._physics_tick)
        self.fsm_timer = self.create_timer(1.0 / 20.0, self._fsm_tick)
        self.mode_timer = self.create_timer(1.0, self._publish_mode)
        self._publish_mode()
        self.get_logger().info(
            f"MuJoCo dual-arm simulator ready: {model_path}; state={FSM_NAMES[self.state]}"
        )

    def _set_state(self, state: int) -> None:
        if state == self.state:
            return
        old = self.state
        self.state = state
        self.ready_count = 0
        if state == SLOW_START:
            # Require fresh commands after enabling. This prevents an enable
            # callback from erasing an earlier command while leaving its seen
            # flag set, which could otherwise produce a false READY state.
            self.command_seen = {side: False for side in ARM_SIDES}
        elif state == PAUSE:
            self.hold_ctrl = np.clip(self.data.qpos.copy(), self.ctrlrange[:, 0], self.ctrlrange[:, 1])
            self.data.ctrl[:] = self.hold_ctrl
            self.command_seen = {side: False for side in ARM_SIDES}
        self.get_logger().info(f"FSM {FSM_NAMES[old]} -> {FSM_NAMES[state]}")

    def _enable_callback(self, msg: Float32) -> None:
        if not math.isfinite(float(msg.data)):
            self.get_logger().warning("ignoring non-finite /api/fsm/enable")
            return
        if msg.data > 0:
            self.last_heartbeat_ns = self.get_clock().now().nanoseconds
            if self.state == PAUSE:
                self.requested_ctrl[:] = np.clip(self.data.qpos, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
                self.data.ctrl[:] = self.requested_ctrl
                self._set_state(SLOW_START)
        else:
            self.last_heartbeat_ns = None
            self._set_state(PAUSE)

    def _command_callback(self, msg: JointState, side: str) -> None:
        arm_slice = ARM_SLICES[side]
        target = normalize_joint_target(
            msg.position, msg.name, JOINT_NAMES[side], self.ctrlrange[arm_slice]
        )
        if target is None:
            self.get_logger().warning(
                f"ignoring invalid {side} joint command: require 7 finite positions and, "
                "when names are present, exactly the expected unique joint names"
            )
            return
        self.requested_ctrl[arm_slice] = target
        self.command_seen[side] = True

    def _physics_tick(self) -> None:
        if self.state == PAUSE:
            self.data.ctrl[:] = self.hold_ctrl
        elif self.state == SLOW_START:
            self.data.ctrl[:] = slow_start_step(
                np.asarray(self.data.ctrl), self.requested_ctrl, self.slow_max_delta
            )
        else:
            self.data.ctrl[:] = self.requested_ctrl
        self.data.ctrl[:] = np.clip(self.data.ctrl, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        self.step_remainder += self.physics_period
        steps = int(self.step_remainder / self.model.opt.timestep)
        self.step_remainder -= steps * self.model.opt.timestep
        for _ in range(max(1, steps)):
            mujoco.mj_step(self.model, self.data)
        self._publish_joint_states()

    def _fsm_tick(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if self.state != PAUSE and (
            self.last_heartbeat_ns is None
            or now_ns - self.last_heartbeat_ns > self.heartbeat_timeout_ns
        ):
            self.get_logger().warning("enable heartbeat timed out; entering PAUSE")
            self._set_state(PAUSE)
        elif self.state == SLOW_START:
            error = float(np.max(np.abs(self.data.qpos - self.requested_ctrl)))
            if all(self.command_seen.values()) and error <= self.ready_tolerance:
                self.ready_count += 1
                if self.ready_count >= self.ready_stable_cycles:
                    self._set_state(READY)
            else:
                self.ready_count = 0
        self.fsm_pub.publish(Int32(data=self.state))

    def _publish_joint_states(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for side in ARM_SIDES:
            arm_slice = ARM_SLICES[side]
            msg = JointState()
            msg.header.stamp = stamp
            msg.name = list(JOINT_NAMES[side])
            msg.position = self.data.qpos[arm_slice].tolist()
            msg.velocity = self.data.qvel[arm_slice].tolist()
            msg.effort = self.data.qfrc_actuator[arm_slice].tolist()
            self.state_pubs[side].publish(msg)

    def _publish_mode(self) -> None:
        self.mode_pub.publish(String(data=json.dumps(CURRENT_MODE, separators=(",", ":"))))


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=HERE / "scene.xml")
    parser.add_argument("--viewer", action="store_true", help="show the optional MuJoCo viewer")
    parser.add_argument("--publish-hz", type=float, default=200.0)
    parser.add_argument("--slow-start-speed", type=float, default=0.5, help="rad/s per joint")
    parser.add_argument("--heartbeat-timeout", type=float, default=1.0)
    parser.add_argument("--ready-tolerance", type=float, default=0.08, help="actual qpos error in rad")
    parser.add_argument("--ready-stable-cycles", type=int, default=5, help="consecutive 20 Hz checks")
    return parser.parse_args(args)


def main(args: Optional[Sequence[str]] = None) -> None:
    parsed = parse_args(args)
    rclpy.init(args=None)
    node = DualArmMuJoCoSim(
        parsed.model,
        publish_hz=parsed.publish_hz,
        slow_start_speed=parsed.slow_start_speed,
        heartbeat_timeout=parsed.heartbeat_timeout,
        ready_tolerance=parsed.ready_tolerance,
        ready_stable_cycles=parsed.ready_stable_cycles,
    )
    viewer = None
    try:
        if parsed.viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(node.model, node.data)
        while rclpy.ok() and (viewer is None or viewer.is_running()):
            rclpy.spin_once(node, timeout_sec=0.05)
            if viewer is not None:
                viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: signal.raise_signal(signal.SIGINT))
    main()
