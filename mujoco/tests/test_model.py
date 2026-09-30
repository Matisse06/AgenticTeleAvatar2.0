import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from convert_urdf import DEFAULT_URDF, EXCLUDES, HOME, JOINTS, build, parse_and_validate
from ros2_sim_node import clip_target, normalize_joint_target, slow_start_step
from test_control import interpolate, message_order, parse_joint_indices, plan_targets


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = HERE / "robot.xml"
        build(DEFAULT_URDF, cls.output)
        cls.robot = mujoco.MjModel.from_xml_path(str(cls.output))
        cls.scene = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
        cls.xml = ET.parse(cls.output).getroot()

    def test_source_paths_topology_and_joint_count(self):
        root, links, joints, mesh_dir = parse_and_validate(DEFAULT_URDF)
        self.assertEqual(len([j for j in joints if j.get("type") == "revolute"]), 14)
        self.assertEqual(mesh_dir, DEFAULT_URDF.parent / "meshes")
        self.assertEqual(set(HOME), set(JOINTS))

    def test_robot_and_scene_compile_dimensions(self):
        for model in (self.robot, self.scene):
            self.assertEqual((model.nq, model.nv, model.nu), (14, 14, 14))

    def test_actuator_ranges_match_joint_ranges_and_home(self):
        for index, name in enumerate(JOINTS):
            joint_range = self.robot.joint(name).range
            self.assertEqual(list(self.robot.actuator(index).ctrlrange), list(joint_range))
            self.assertLessEqual(joint_range[0], HOME[name])
            self.assertLessEqual(HOME[name], joint_range[1])

    def test_home_keyframe(self):
        expected = [HOME[name] for name in JOINTS]
        key = mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.assertGreaterEqual(key, 0)
        self.assertEqual(list(self.robot.key_qpos[key]), expected)
        self.assertEqual(list(self.robot.key_ctrl[key]), expected)

    def test_collision_exclusions(self):
        actual = {(x.get("body1"), x.get("body2")) for x in self.xml.findall("contact/exclude")}
        self.assertEqual(actual, set(EXCLUDES))

    def test_no_package_mesh_paths_and_files_exist(self):
        self.assertTrue(DEFAULT_URDF.is_relative_to(HERE))
        meshes = self.xml.findall("asset/mesh")
        self.assertEqual(len(meshes), 19)
        for mesh in meshes:
            filename = mesh.get("file")
            self.assertNotIn("package://", filename)
            path = (HERE / filename).resolve()
            self.assertTrue(path.is_relative_to(HERE))
            self.assertTrue(path.is_file())

    def test_demo_runs_after_copy_outside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "mujoco"
            shutil.copytree(HERE, copied, ignore=shutil.ignore_patterns("__pycache__"))
            result = subprocess.run(
                [sys.executable, "convert_urdf.py", "--check-only", "--verbose"],
                cwd=copied,
                text=True,
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            model = mujoco.MjModel.from_xml_path(str(copied / "scene.xml"))
            self.assertEqual((model.nq, model.nv, model.nu), (14, 14, 14))

    def test_urdf_origins_are_emitted_as_equivalent_quaternions(self):
        """Guard the URDF Rz(yaw) Ry(pitch) Rx(roll) convention at every origin."""
        source = ET.parse(DEFAULT_URDF).getroot()
        bodies = {body.get("name"): body for body in self.xml.findall(".//body")}

        def assert_rotation(origin, target):
            roll, pitch, yaw = map(float, origin.get("rpy", "0 0 0").split())
            cr, sr = math.cos(roll), math.sin(roll)
            cp, sp = math.cos(pitch), math.sin(pitch)
            cy, sy = math.cos(yaw), math.sin(yaw)
            expected = (
                cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                -sp, cp * sr, cp * cr,
            )
            self.assertIsNone(target.get("euler"))
            w, x, y, z = [float(value) for value in target.get("quat").split()]
            actual = (
                1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
            )
            for actual_value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(actual_value, expected_value, places=12)

        for joint in source.findall("joint"):
            origin = joint.find("origin")
            if origin is not None:
                child = joint.find("child").get("link")
                assert_rotation(origin, bodies[child])

        for link in source.findall("link"):
            body = bodies[link.get("name")]
            inertial = link.find("inertial")
            if inertial is not None and inertial.find("origin") is not None:
                origin = inertial.find("origin")
                if any(abs(float(value)) > 1e-15
                       for value in origin.get("rpy", "0 0 0").split()):
                    # fullinertia cannot carry a quat in MJCF; the converter
                    # rotates such tensors into the link frame instead.
                    self.assertIsNone(body.find("inertial").get("euler"))
            for tag in ("visual", "collision"):
                for index, item in enumerate(link.findall(tag)):
                    origin = item.find("origin")
                    if origin is not None and item.find("geometry/mesh") is not None:
                        name = f"{link.get('name')}_{tag}_{index}"
                        assert_rotation(origin, body.find(f"geom[@name='{name}']"))

    def test_generated_model_never_uses_mjcf_euler_for_urdf_origins(self):
        self.assertFalse(any(element.get("euler") is not None for element in self.xml.iter()))

    def test_finite_stepping_from_home(self):
        data = mujoco.MjData(self.scene)
        key = mujoco.mj_name2id(self.scene, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.scene, data, key)
        for _ in range(200):
            mujoco.mj_step(self.scene, data)
        self.assertTrue(all(math.isfinite(float(value)) for value in data.qpos))
        self.assertTrue(all(math.isfinite(float(value)) for value in data.qvel))

    def test_simulator_command_helpers_validate_and_limit(self):
        ranges = np.asarray(self.robot.actuator_ctrlrange[:7])
        self.assertEqual(clip_target([99] * 7, ranges).tolist(), ranges[:, 1].tolist())
        self.assertIsNone(clip_target([0] * 6, ranges))
        self.assertIsNone(clip_target([float("nan")] * 7, ranges))
        result = slow_start_step(np.zeros(7), np.ones(7), np.full(7, 0.2))
        self.assertTrue(np.allclose(result, 0.2))

    def test_named_joint_commands_reorder_and_reject_bad_names(self):
        ranges = np.asarray(self.robot.actuator_ctrlrange[:7])
        names = list(JOINTS[:7])
        values = list(range(7))
        expected = np.clip(values, ranges[:, 0], ranges[:, 1])
        actual = normalize_joint_target(values[::-1], names[::-1], names, ranges)
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertIsNone(normalize_joint_target(values, names[:-1], names, ranges))
        self.assertIsNone(normalize_joint_target(values, names[:-1] + names[-2:-1], names, ranges))
        self.assertIsNone(normalize_joint_target(values, names[:-1] + ["unknown"], names, ranges))
    def test_control_client_plans_bounded_interpolated_targets(self):
        starts = {
            "left": np.asarray([HOME[name] for name in JOINTS[:7]], dtype=float),
            "right": np.asarray([HOME[name] for name in JOINTS[7:]], dtype=float),
        }
        targets = plan_targets(
            starts,
            np.asarray(self.robot.actuator_ctrlrange),
            ("left",),
            parse_joint_indices("1,4"),
            0.12,
        )
        self.assertAlmostEqual(targets["left"][0], starts["left"][0] + 0.12)
        self.assertAlmostEqual(targets["left"][3], starts["left"][3] + 0.12)
        self.assertTrue(np.allclose(targets["right"], starts["right"]))
        self.assertTrue(np.allclose(interpolate(starts["left"], targets["left"], 0), starts["left"]))
        self.assertTrue(np.allclose(interpolate(starts["left"], targets["left"], 1), targets["left"]))
        with self.assertRaises(ValueError):
            plan_targets(starts, np.asarray(self.robot.actuator_ctrlrange), ("left",), (0,), 0.6)

    def test_control_client_reversed_transport_preserves_name_value_pairs(self):
        values = list(range(7))
        names, reordered = message_order("left", values, True)
        normalized = normalize_joint_target(
            reordered,
            names,
            JOINTS[:7],
            np.asarray(self.robot.actuator_ctrlrange[:7]),
        )
        expected = np.clip(values, self.robot.actuator_ctrlrange[:7, 0], self.robot.actuator_ctrlrange[:7, 1])
        self.assertTrue(np.allclose(normalized, expected))


if __name__ == "__main__":
    unittest.main()
