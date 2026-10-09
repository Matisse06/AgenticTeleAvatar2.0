#!/usr/bin/env python3
"""End-to-end ROS 2 smoke test of the new model's simulator: the checks of mujoco/smoke_test.py, run on
model/run_sim.sh, with the clipping check using this model's joint ranges. Starts and stops its own simulator."""
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
os.environ.update(ROS_DOMAIN_ID=DOMAIN, ROS_LOCALHOST_ONLY="1", ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")

import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Int32, String

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ros2_sim_node import CURRENT_MODE, JOINT_NAMES, model_joint  # noqa: E402

SIDES = ("left", "right")


class Probe(Node):
    def __init__(self) -> None:
        super().__init__("teleavatar_smoke_probe")
        self.enable_pub = self.create_publisher(Float32, "/api/fsm/enable", 10)
        self.commands = {side: self.create_publisher(JointState, f"/api/{side}_arm/joint_cmd", 10) for side in SIDES}
        self.states, self.fsm, self.mode = {}, [], None
        for side in SIDES:
            self.create_subscription(JointState, f"/{side}_arm/joint_states",
                                     lambda msg, arm=side: self.states.__setitem__(arm, msg), 10)
        self.create_subscription(Int32, "/fsm_state", lambda msg: self.fsm.append(msg.data), 10)
        self.create_subscription(String, "/api/current_mode", lambda msg: setattr(self, "mode", json.loads(msg.data)), 10)

    def publish_enable(self, enabled: bool) -> None:
        self.enable_pub.publish(Float32(data=1.0 if enabled else 0.0))

    def publish_targets(self, targets, *, reverse_names=False) -> None:
        for side in SIDES:
            names, positions = list(JOINT_NAMES[side]), list(targets[side])
            if reverse_names:
                names.reverse()
                positions.reverse()
            self.commands[side].publish(JointState(name=names, position=positions))


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
    print(f"[{'PASS' if condition else 'FAIL'}] {label}{': ' + detail if detail else ''}", flush=True)
    if not condition:
        raise RuntimeError(label)


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    # Joint 2 swings the arm away from the body toward 0, the left arm's lower limit and the right arm's upper one. (At
    # the other limit, 1.8 either way, the shoulder meets the torso at about 1.72 rad and stops short of it.)
    joint2_limit = {side: model.joint(model_joint(JOINT_NAMES[side][1])).range[0 if side == "left" else 1]
                    for side in SIDES}
    env = dict(os.environ, SIM_ROS_DOMAIN_ID=DOMAIN)
    process = subprocess.Popen([str(HERE / "run_sim.sh")], cwd=HERE, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        rclpy.init()
        probe = Probe()
        check("state streams and current_mode",
              wait_until(probe, lambda: len(probe.states) == 2 and probe.fsm and probe.mode is not None, 20))
        check("current_mode payload", probe.mode == CURRENT_MODE, repr(probe.mode))
        check("initial PAUSE", probe.fsm[-1] == 0, str(probe.fsm[-1]))

        start = {side: list(probe.states[side].position) for side in SIDES}
        targets = {side: values.copy() for side, values in start.items()}
        targets["left"][0] += 0.12
        targets["right"][0] -= 0.12
        drive = lambda: (probe.publish_enable(True), probe.publish_targets(targets, reverse_names=True))
        check("SLOW_START observed", wait_until(probe, lambda: 1 in probe.fsm, 2, drive))
        check("SLOW_START to READY on actual qpos", wait_until(probe, lambda: probe.fsm[-1] == 2, 8, drive))
        converged = lambda: all(max(abs(a - b) for a, b in zip(probe.states[side].position, targets[side])) < 0.09
                                for side in SIDES)
        check("named command reorder", wait_until(probe, converged, 2, drive),
              repr({side: list(probe.states[side].position) for side in SIDES}))

        # An out-of-range joint 2 command is clipped to the control range: the joint settles at its limit, 0 (from home,
        # 1.16 rad for the left arm and 0.92 rad for the right). Waiting until it is at rest also makes the PAUSE check
        # below fair.
        clipped = {side: [*targets[side][:1], -99.0 if side == "left" else 99.0, *targets[side][2:]] for side in SIDES}
        reached = lambda side: (joint2_limit[side] - probe.states[side].position[1]) * (1 if side == "right" else -1)
        at_limit = lambda: all(reached(side) < 0.02 and abs(probe.states[side].velocity[1]) < 0.05 for side in SIDES)
        check("ctrlrange clipping",
              wait_until(probe, at_limit, 8, lambda: (probe.publish_enable(True), probe.publish_targets(clipped))),
              repr({side: (probe.states[side].position[1], float(joint2_limit[side])) for side in SIDES}))

        probe.publish_enable(False)
        check("disable to PAUSE", wait_until(probe, lambda: probe.fsm[-1] == 0, 2))
        wait_until(probe, lambda: False, 0.2)  # keep spinning while the PAUSE hold settles
        paused = {side: list(probe.states[side].position) for side in SIDES}
        wait_until(probe, lambda: False, 0.35)
        drift = {side: max(abs(a - b) for a, b in zip(paused[side], probe.states[side].position)) for side in SIDES}
        check("PAUSE holds position", all(value < 0.08 for value in drift.values()), repr(drift))
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
