#!/usr/bin/env python3
"""ROS 2 simulator of the new TeleAvatar model (model/scene.xml) with the vendor simulator's API.

Same topics, messages, FSM and command validation as mujoco/ros2_sim_node.py (it loads that file's helpers), so
clients written for the vendor simulator, such as mujoco/test_control.py, work unchanged. The API's l_joint1..7 and
r_joint1..7 are this model's armL1..7_joint and armR1..7_joint (same angle conventions). As in the vendor simulator,
the lift and grippers have no API: they hold the home keyframe's targets, which --lift and --gripper change. The base
is fixed (model/scene.xml); with --model model/scene_mobile.xml it is drivable but, having no API either, holds still.

  ./model/run_sim.sh                  # headless, 200 Hz joint states
  ./model/run_sim.sh --viewer         # with the MuJoCo viewer
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import signal
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Int32, String

HERE = Path(__file__).resolve().parent


def load_vendor_module(name: str, path: Path):
    """Import a vendor file without writing bytecode into the vendor folder, which stays as delivered."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


vendor = load_vendor_module("vendor_ros2_sim_node", HERE.parent / "mujoco" / "ros2_sim_node.py")
ARM_SIDES, JOINT_NAMES, CURRENT_MODE = vendor.ARM_SIDES, vendor.JOINT_NAMES, vendor.CURRENT_MODE
PAUSE, SLOW_START, READY, FSM_NAMES = vendor.PAUSE, vendor.SLOW_START, vendor.READY, vendor.FSM_NAMES
normalize_joint_target, slow_start_step = vendor.normalize_joint_target, vendor.slow_start_step


def model_joint(api_name: str) -> str:
    """l_joint3 -> armL3_joint, r_joint7 -> armR7_joint."""
    return f"arm{api_name[0].upper()}{api_name[-1]}_joint"


def actuator_for(model: mujoco.MjModel, joint: str) -> int:
    joint_id = model.joint(joint).id
    for index in range(model.nu):
        if model.actuator_trntype[index] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[index, 0] == joint_id:
            return index
    raise RuntimeError(f"no actuator drives {joint}")


class TeleAvatarSim(Node):
    """PAUSE/SLOW_START/READY simulator exposing the vendor's public joint API on the new model."""

    def __init__(
        self,
        model_path: Path = HERE / "scene.xml",
        *,
        publish_hz: float = 200.0,
        slow_start_speed: float = 0.5,
        heartbeat_timeout: float = 1.0,
        ready_tolerance: float = 0.08,
        ready_stable_cycles: int = 5,
        lift: Optional[float] = None,
        gripper: Optional[float] = None,
    ) -> None:
        super().__init__("teleavatar_mujoco_sim")
        if min(publish_hz, slow_start_speed, heartbeat_timeout, ready_tolerance) <= 0:
            raise ValueError("rates, speeds, timeout and tolerance must be positive")
        if ready_stable_cycles < 1:
            raise ValueError("ready_stable_cycles must be positive")

        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id < 0:
            raise RuntimeError("model does not contain the required 'home' keyframe")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)

        names = [model_joint(name) for side in ARM_SIDES for name in JOINT_NAMES[side]]
        self.qadr = np.array([self.model.joint(name).qposadr[0] for name in names])
        self.dadr = np.array([self.model.joint(name).dofadr[0] for name in names])
        self.act = np.array([actuator_for(self.model, name) for name in names])
        self.arm_slices = {"left": slice(0, 7), "right": slice(7, 14)}
        # Lift and grippers: fixed targets, clipped to their control ranges.
        for value, actuators in ((lift, ["lift"]), (gripper, ["left_gripper", "right_gripper"])):
            for name in actuators:
                if value is not None:
                    index = self.model.actuator(name).id
                    self.data.ctrl[index] = np.clip(value, *self.model.actuator_ctrlrange[index])
        self.other_ctrl = self.data.ctrl.copy()

        self.ctrlrange = np.asarray(self.model.actuator_ctrlrange[self.act], dtype=float).copy()
        self.requested_ctrl = np.asarray(self.data.ctrl[self.act], dtype=float).copy()
        self.hold_ctrl = self._arm_qpos()
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
            side: self.create_publisher(JointState, f"/{side}_arm/joint_states", 10) for side in ARM_SIDES
        }
        self.fsm_pub = self.create_publisher(Int32, "/fsm_state", 10)
        self.mode_pub = self.create_publisher(String, "/api/current_mode", 10)
        self.create_subscription(Float32, "/api/fsm/enable", self._enable_callback, 10)
        self.command_subs = [
            self.create_subscription(JointState, f"/api/{side}_arm/joint_cmd",
                                     lambda msg, arm=side: self._command_callback(msg, arm), 10)
            for side in ARM_SIDES
        ]
        self.physics_timer = self.create_timer(1.0 / publish_hz, self._physics_tick)
        self.fsm_timer = self.create_timer(1.0 / 20.0, self._fsm_tick)
        self.mode_timer = self.create_timer(1.0, self._publish_mode)
        self._publish_mode()
        self.get_logger().info(f"MuJoCo TeleAvatar simulator ready: {model_path}; state={FSM_NAMES[self.state]}")

    def _arm_qpos(self) -> np.ndarray:
        return np.clip(self.data.qpos[self.qadr], self.ctrlrange[:, 0], self.ctrlrange[:, 1])

    def _set_state(self, state: int) -> None:
        if state == self.state:
            return
        old, self.state, self.ready_count = self.state, state, 0
        if state == SLOW_START:
            self.command_seen = {side: False for side in ARM_SIDES}  # require fresh commands after enabling
        elif state == PAUSE:
            self.hold_ctrl = self._arm_qpos()
            self.data.ctrl[self.act] = self.hold_ctrl
            self.command_seen = {side: False for side in ARM_SIDES}
        self.get_logger().info(f"FSM {FSM_NAMES[old]} -> {FSM_NAMES[state]}")

    def _enable_callback(self, msg: Float32) -> None:
        if not math.isfinite(float(msg.data)):
            self.get_logger().warning("ignoring non-finite /api/fsm/enable")
            return
        if msg.data > 0:
            self.last_heartbeat_ns = self.get_clock().now().nanoseconds
            if self.state == PAUSE:
                self.requested_ctrl[:] = self._arm_qpos()
                self.data.ctrl[self.act] = self.requested_ctrl
                self._set_state(SLOW_START)
        else:
            self.last_heartbeat_ns = None
            self._set_state(PAUSE)

    def _command_callback(self, msg: JointState, side: str) -> None:
        arm = self.arm_slices[side]
        target = normalize_joint_target(msg.position, msg.name, JOINT_NAMES[side], self.ctrlrange[arm])
        if target is None:
            self.get_logger().warning(
                f"ignoring invalid {side} joint command: require 7 finite positions and, "
                "when names are present, exactly the expected unique joint names")
            return
        self.requested_ctrl[arm] = target
        self.command_seen[side] = True

    def _physics_tick(self) -> None:
        if self.state == PAUSE:
            arm_ctrl = self.hold_ctrl
        elif self.state == SLOW_START:
            arm_ctrl = slow_start_step(np.asarray(self.data.ctrl[self.act]), self.requested_ctrl, self.slow_max_delta)
        else:
            arm_ctrl = self.requested_ctrl
        self.data.ctrl[:] = self.other_ctrl
        self.data.ctrl[self.act] = np.clip(arm_ctrl, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        self.step_remainder += self.physics_period
        steps = int(self.step_remainder / self.model.opt.timestep)
        self.step_remainder -= steps * self.model.opt.timestep
        for _ in range(max(1, steps)):
            mujoco.mj_step(self.model, self.data)
        self._publish_joint_states()

    def _fsm_tick(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if self.state != PAUSE and (self.last_heartbeat_ns is None
                                    or now_ns - self.last_heartbeat_ns > self.heartbeat_timeout_ns):
            self.get_logger().warning("enable heartbeat timed out; entering PAUSE")
            self._set_state(PAUSE)
        elif self.state == SLOW_START:
            error = float(np.max(np.abs(self.data.qpos[self.qadr] - self.requested_ctrl)))
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
            arm = self.arm_slices[side]
            msg = JointState()
            msg.header.stamp = stamp
            msg.name = list(JOINT_NAMES[side])
            msg.position = self.data.qpos[self.qadr[arm]].tolist()
            msg.velocity = self.data.qvel[self.dadr[arm]].tolist()
            msg.effort = self.data.qfrc_actuator[self.dadr[arm]].tolist()  # includes gravity compensation
            self.state_pubs[side].publish(msg)

    def _publish_mode(self) -> None:
        self.mode_pub.publish(String(data=json.dumps(CURRENT_MODE, separators=(",", ":"))))


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=HERE / "scene.xml")
    parser.add_argument("--viewer", action="store_true", help="show the optional MuJoCo viewer")
    parser.add_argument("--publish-hz", type=float, default=200.0)
    parser.add_argument("--slow-start-speed", type=float, default=0.5, help="rad/s per joint")
    parser.add_argument("--heartbeat-timeout", type=float, default=1.0)
    parser.add_argument("--ready-tolerance", type=float, default=0.08, help="actual qpos error in rad")
    parser.add_argument("--ready-stable-cycles", type=int, default=5, help="consecutive 20 Hz checks")
    parser.add_argument("--lift", type=float, help="lift target in m (0 = top); default: home keyframe")
    parser.add_argument("--gripper", type=float,
                        help="both gripper inputs, 0 (closed) .. 1 rad (open); default: home keyframe (open)")
    return parser.parse_args(args)


def main(args: Optional[Sequence[str]] = None) -> None:
    parsed = parse_args(args)
    rclpy.init(args=None)
    node = TeleAvatarSim(parsed.model, publish_hz=parsed.publish_hz, slow_start_speed=parsed.slow_start_speed,
                         heartbeat_timeout=parsed.heartbeat_timeout, ready_tolerance=parsed.ready_tolerance,
                         ready_stable_cycles=parsed.ready_stable_cycles, lift=parsed.lift, gripper=parsed.gripper)
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
            # The viewer's thread is a daemon: let it finish before the interpreter's exit runs glfw.terminate,
            # which otherwise can crash the exit while that thread is still drawing (segfault 139).
            for thread in threading.enumerate():
                if thread is not threading.current_thread() and thread.daemon and "_launch_internal" in thread.name:
                    thread.join(timeout=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: signal.raise_signal(signal.SIGINT))
    main()
