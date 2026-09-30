#!/usr/bin/env python3
"""End-to-end ROS 2 smoke test for the isolated MuJoCo simulator."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DOMAIN = os.environ.get("SIM_ROS_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "90"))
if DOMAIN == "29":
    print("smoke_test.py refuses production ROS domain 29", file=sys.stderr)
    raise SystemExit(2)
os.environ["ROS_DOMAIN_ID"] = DOMAIN
os.environ["ROS_LOCALHOST_ONLY"] = "1"

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Int32, String

ROOT = Path(__file__).resolve().parent
SIDES = ("left", "right")
EXPECTED_MODE = {
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


class Probe(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_smoke_probe")
        self.enable_pub = self.create_publisher(Float32, "/api/fsm/enable", 10)
        self.commands = {
            side: self.create_publisher(JointState, f"/api/{side}_arm/joint_cmd", 10)
            for side in SIDES
        }
        self.states = {}
        self.fsm = []
        self.mode = None
        for side in SIDES:
            self.create_subscription(
                JointState,
                f"/{side}_arm/joint_states",
                lambda msg, arm=side: self.states.__setitem__(arm, msg),
                10,
            )
        self.create_subscription(Int32, "/fsm_state", lambda msg: self.fsm.append(msg.data), 10)
        self.create_subscription(String, "/api/current_mode", self._mode_callback, 10)

    def _mode_callback(self, msg: String) -> None:
        self.mode = json.loads(msg.data)

    def publish_enable(self, enabled: bool) -> None:
        self.enable_pub.publish(Float32(data=1.0 if enabled else 0.0))

    def publish_targets(self, targets, *, reverse_names=False) -> None:
        for side in SIDES:
            names = [f"{'l' if side == 'left' else 'r'}_joint{i}" for i in range(1, 8)]
            positions = list(targets[side])
            if reverse_names:
                names.reverse()
                positions.reverse()
            msg = JointState(name=names, position=positions)
            self.commands[side].publish(msg)


def wait_until(probe, predicate, timeout, action=None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if action is not None:
            action()
        rclpy.spin_once(probe, timeout_sec=0.02)
        if predicate():
            return True
    return False


def check(label: str, condition: bool, detail="") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}{': ' + detail if detail else ''}")
    if not condition:
        raise RuntimeError(label)


def main() -> int:
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = DOMAIN
    env["SIM_ROS_DOMAIN_ID"] = DOMAIN
    env["ROS_LOCALHOST_ONLY"] = "1"
    process = subprocess.Popen(
        [str(ROOT / "run_sim.sh")], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    try:
        rclpy.init()
        probe = Probe()
        check(
            "state streams and current_mode",
            wait_until(probe, lambda: len(probe.states) == 2 and probe.fsm and probe.mode is not None, 8),
        )
        check("current_mode payload", probe.mode == EXPECTED_MODE, repr(probe.mode))
        check("initial PAUSE", probe.fsm[-1] == 0, str(probe.fsm[-1]))

        start = {side: list(probe.states[side].position) for side in SIDES}
        targets = {side: values.copy() for side, values in start.items()}
        targets["left"][0] += 0.12
        targets["right"][0] -= 0.12
        drive = lambda: (probe.publish_enable(True), probe.publish_targets(targets, reverse_names=True))
        check("SLOW_START observed", wait_until(probe, lambda: 1 in probe.fsm, 2, drive))
        check("SLOW_START to READY on actual qpos", wait_until(probe, lambda: probe.fsm[-1] == 2, 8, drive))
        def reordered_targets_converged():
            return all(
                max(
                    abs(actual - expected)
                    for actual, expected in zip(probe.states[side].position, targets[side])
                ) < 0.09
                for side in SIDES
            )

        check(
            "named command reorder",
            wait_until(probe, reordered_targets_converged, 2, drive),
            repr(
                {
                    side: {
                        "actual": list(probe.states[side].position),
                        "target": targets[side],
                    }
                    for side in SIDES
                }
            ),
        )

        clipped = {side: [99.0] * 7 for side in SIDES}
        check(
            "ctrlrange clipping",
            wait_until(
                probe,
                lambda: (
                    probe.states["left"].position[1] > 1.70
                    and probe.states["right"].position[1] > -0.10
                ),
                8,
                lambda: (probe.publish_enable(True), probe.publish_targets(clipped)),
            ),
        )

        probe.publish_enable(False)
        check("disable to PAUSE", wait_until(probe, lambda: probe.fsm[-1] == 0, 2))
        # Keep spinning while the PAUSE hold target settles and while measuring;
        # sleeping here would leave the probe with a stale pre-PAUSE state.
        wait_until(probe, lambda: False, 0.2)
        paused = {side: list(probe.states[side].position) for side in SIDES}
        wait_until(probe, lambda: False, 0.35)
        pause_drift = {
            side: max(abs(a - b) for a, b in zip(paused[side], probe.states[side].position))
            for side in SIDES
        }
        check(
            "PAUSE holds position",
            all(drift < 0.08 for drift in pause_drift.values()),
            repr(pause_drift),
        )
        print("ROS MuJoCo smoke passed")
        return 0
    finally:
        if "probe" in locals():
            probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        check("simulator process cleanup", process.poll() is not None)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, RuntimeError) as exc:
        print(f"ROS MuJoCo smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
