#!/usr/bin/env python3
"""Attach to the isolated MuJoCo simulator and run a small joint-control test."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SIDES = ("left", "right")
JOINT_NAMES = {
    "left": tuple(f"l_joint{i}" for i in range(1, 8)),
    "right": tuple(f"r_joint{i}" for i in range(1, 8)),
}
ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
PAUSE = 0
READY = 2
MAX_AMPLITUDE = 0.5


def interpolate(start: Sequence[float], target: Sequence[float], alpha: float) -> np.ndarray:
    """Linearly interpolate, clamping alpha to [0, 1]."""
    alpha = min(1.0, max(0.0, float(alpha)))
    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if start_array.shape != target_array.shape or not np.all(np.isfinite([*start_array, *target_array])):
        raise ValueError("interpolation endpoints must be equally shaped finite arrays")
    return start_array + (target_array - start_array) * alpha


def parse_joint_indices(value: str) -> tuple[int, ...]:
    """Parse unique one-based joint indices and return zero-based indices."""
    try:
        indices = tuple(int(item.strip()) - 1 for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--joints must be comma-separated integers") from exc
    if not indices or len(set(indices)) != len(indices) or any(index < 0 or index >= 7 for index in indices):
        raise argparse.ArgumentTypeError("--joints must contain unique values from 1 through 7")
    return indices


def plan_targets(
    starts: dict[str, Sequence[float]],
    ctrlrange: np.ndarray,
    active_sides: Sequence[str],
    joint_indices: Sequence[int],
    amplitude: float,
) -> dict[str, np.ndarray]:
    """Apply a bounded relative offset and clip it to model control ranges."""
    if not math.isfinite(amplitude) or abs(amplitude) > MAX_AMPLITUDE:
        raise ValueError(f"amplitude must be finite and no greater than {MAX_AMPLITUDE} rad")
    targets: dict[str, np.ndarray] = {}
    for side in SIDES:
        start = np.asarray(starts[side], dtype=float)
        if start.shape != (7,) or not np.all(np.isfinite(start)):
            raise ValueError(f"{side} start state must contain seven finite positions")
        target = start.copy()
        if side in active_sides:
            target[list(joint_indices)] += amplitude
        ranges = ctrlrange[ARM_SLICES[side]]
        targets[side] = np.clip(target, ranges[:, 0], ranges[:, 1])
    return targets


def message_order(
    side: str, positions: Sequence[float], reverse_names: bool
) -> tuple[list[str], list[float]]:
    """Return name-position pairs in normal or reversed transport order."""
    names = list(JOINT_NAMES[side])
    values = list(positions)
    if reverse_names:
        names.reverse()
        values.reverse()
    return names, values


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=HERE / "scene.xml")
    parser.add_argument("--arms", choices=("both", "left", "right"), default="both")
    parser.add_argument("--joints", type=parse_joint_indices, default=parse_joint_indices("1,4"))
    parser.add_argument("--amplitude", type=float, default=0.12, help="relative joint offset in rad")
    parser.add_argument("--duration", type=float, default=3.0, help="seconds per trajectory")
    parser.add_argument("--hold", type=float, default=0.5, help="seconds to hold the test target")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=0.10, help="maximum joint error in rad")
    parser.add_argument("--reordered-names", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without publishing")
    parsed = parser.parse_args(args)
    if parsed.duration <= 0 or parsed.hold < 0 or parsed.timeout <= 0 or parsed.tolerance <= 0:
        parser.error("durations, timeout and tolerance must be positive (hold may be zero)")
    if not math.isfinite(parsed.amplitude) or abs(parsed.amplitude) > MAX_AMPLITUDE:
        parser.error(f"--amplitude must be finite and within ±{MAX_AMPLITUDE} rad")
    return parsed


def configure_ros_environment() -> str:
    domain = os.environ.get("SIM_ROS_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "90"))
    if domain == "29":
        print("test_control.py refuses production ROS domain 29", file=sys.stderr)
        raise SystemExit(2)
    os.environ["ROS_DOMAIN_ID"] = domain
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    return domain


def run(parsed: argparse.Namespace) -> int:
    configure_ros_environment()

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32, Int32, String

    class ControlClient(Node):
        def __init__(self) -> None:
            super().__init__("mujoco_test_control")
            self.states: dict[str, JointState] = {}
            self.fsm_state: int | None = None
            self.mode: dict | None = None
            self.enable_pub = self.create_publisher(Float32, "/api/fsm/enable", 10)
            self.command_pubs = {
                side: self.create_publisher(JointState, f"/api/{side}_arm/joint_cmd", 10)
                for side in SIDES
            }
            for side in SIDES:
                self.create_subscription(
                    JointState,
                    f"/{side}_arm/joint_states",
                    lambda msg, arm=side: self.states.__setitem__(arm, msg),
                    10,
                )
            self.create_subscription(Int32, "/fsm_state", self._fsm_callback, 10)
            self.create_subscription(String, "/api/current_mode", self._mode_callback, 10)

        def _fsm_callback(self, msg: Int32) -> None:
            self.fsm_state = msg.data

        def _mode_callback(self, msg: String) -> None:
            try:
                self.mode = json.loads(msg.data)
            except json.JSONDecodeError:
                self.mode = None

        def publish_enable(self, enabled: bool) -> None:
            self.enable_pub.publish(Float32(data=1.0 if enabled else 0.0))

        def publish_positions(self, positions: dict[str, Sequence[float]]) -> None:
            stamp = self.get_clock().now().to_msg()
            for side in SIDES:
                names, values = message_order(side, positions[side], parsed.reordered_names)
                msg = JointState()
                msg.header.stamp = stamp
                msg.name = names
                msg.position = values
                self.command_pubs[side].publish(msg)

    def wait_until(client: ControlClient, predicate, timeout: float, action=None) -> bool:
        deadline = time.monotonic() + timeout
        while not stop_requested and time.monotonic() < deadline:
            if action is not None:
                action()
            rclpy.spin_once(client, timeout_sec=0.01)
            if predicate():
                return True
        return False

    def max_error(client: ControlClient, targets: dict[str, Sequence[float]]) -> float:
        return max(
            abs(actual - expected)
            for side in SIDES
            for actual, expected in zip(client.states[side].position, targets[side])
        )

    def drive_trajectory(
        client: ControlClient,
        starts: dict[str, Sequence[float]],
        targets: dict[str, Sequence[float]],
        duration: float,
    ) -> None:
        begin = time.monotonic()
        period = 0.01
        next_publish = begin
        while not stop_requested:
            now = time.monotonic()
            alpha = (now - begin) / duration
            commands = {side: interpolate(starts[side], targets[side], alpha) for side in SIDES}
            if now >= next_publish:
                client.publish_enable(True)
                client.publish_positions(commands)
                next_publish += period
            rclpy.spin_once(client, timeout_sec=0.002)
            if alpha >= 1.0:
                return
        raise KeyboardInterrupt

    model = mujoco.MjModel.from_xml_path(str(parsed.model.resolve()))
    if model.nu != 14:
        raise RuntimeError("control test requires a 14-actuator model")
    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=float)
    active_sides = SIDES if parsed.arms == "both" else (parsed.arms,)
    stop_requested = False

    def request_stop(*_args) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    rclpy.init()
    client = ControlClient()
    starts: dict[str, list[float]] | None = None
    moved = False
    try:
        if not wait_until(
            client,
            lambda: len(client.states) == 2 and client.fsm_state is not None and client.mode is not None,
            parsed.timeout,
        ):
            print("MuJoCo simulator topics were not found", file=sys.stderr)
            return 3
        if client.mode.get("simulator") != "mujoco":
            print(f"refusing non-MuJoCo current_mode: {client.mode}", file=sys.stderr)
            return 2
        if client.fsm_state != PAUSE:
            print(f"simulator must start in PAUSE, got {client.fsm_state}", file=sys.stderr)
            return 1

        starts = {side: list(client.states[side].position) for side in SIDES}
        targets = plan_targets(starts, ctrlrange, active_sides, parsed.joints, parsed.amplitude)
        print(f"active arms: {', '.join(active_sides)}")
        print(f"joints: {', '.join(str(index + 1) for index in parsed.joints)}")
        for side in SIDES:
            print(f"{side} start : {np.round(starts[side], 4).tolist()}")
            print(f"{side} target: {np.round(targets[side], 4).tolist()}")
        if parsed.dry_run:
            print("dry-run complete; no commands published")
            return 0

        print("moving to test target...")
        drive_trajectory(client, starts, targets, parsed.duration)
        moved = True
        if not wait_until(
            client,
            lambda: client.fsm_state == READY and max_error(client, targets) <= parsed.tolerance,
            parsed.timeout,
            lambda: (client.publish_enable(True), client.publish_positions(targets)),
        ):
            raise RuntimeError(
                f"target did not converge: fsm={client.fsm_state}, error={max_error(client, targets):.4f} rad"
            )
        print(f"target reached; max error={max_error(client, targets):.4f} rad")

        hold_until = time.monotonic() + parsed.hold
        while time.monotonic() < hold_until and not stop_requested:
            client.publish_enable(True)
            client.publish_positions(targets)
            rclpy.spin_once(client, timeout_sec=0.01)

        print("returning to latched start...")
        current = {side: list(client.states[side].position) for side in SIDES}
        drive_trajectory(client, current, starts, parsed.duration)
        if not wait_until(
            client,
            lambda: max_error(client, starts) <= parsed.tolerance,
            parsed.timeout,
            lambda: (client.publish_enable(True), client.publish_positions(starts)),
        ):
            raise RuntimeError(f"return did not converge: error={max_error(client, starts):.4f} rad")
        print(f"start restored; max error={max_error(client, starts):.4f} rad")
        return 0
    finally:
        if moved and starts is not None and not stop_requested:
            client.publish_positions(starts)
        for _ in range(5):
            client.publish_enable(False)
            rclpy.spin_once(client, timeout_sec=0.02)
        if client.fsm_state == PAUSE:
            print("simulator returned to PAUSE")
        client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(args))
    except KeyboardInterrupt:
        print("control test interrupted; disable sent", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"control test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
