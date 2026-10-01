"""Tests of the new TeleAvatar model: the default with the base fixed (model/robot.xml, scene.xml) and the drivable-base
variant (robot_mobile.xml, scene_mobile.xml).

  python3 -m unittest discover -s model/tests -v

Like the vendor's tests, setUpClass rebuilds the generated files in place; test_generated_files_are_up_to_date fails if
that changed them (commit the regenerated files). Needs the unpacked vendor meshes (setup/unpack_assets.py).
"""
import importlib.util
import math
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import convert  # noqa: E402
import mobile_base  # noqa: E402

BASE = [name for name, _, _ in convert.BASE_JOINTS]
VENDOR_URDF = ROOT / "mujoco" / "urdf" / "urdf20260625" / "urdf20260625.urdf"
VENDOR_CONVERTER = ROOT / "mujoco" / "convert_urdf.py"


def urdf_fk(path: Path, q: dict) -> tuple[dict, dict]:
    """Independent forward kinematics of a URDF: world transforms of links, and world (origin, axis) of joints."""
    root = ET.parse(path).getroot()
    joints = root.findall("joint")
    children = {}
    for joint in joints:
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    child_links = {joint.find("child").get("link") for joint in joints}
    links, axes = {}, {}

    def origin(element):
        node = element.find("origin")
        transform = np.eye(4)
        if node is not None:
            roll, pitch, yaw = (float(v) for v in node.get("rpy", "0 0 0").split())
            rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
            ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
            rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            transform[:3, :3] = rz @ ry @ rx
            transform[:3, 3] = [float(v) for v in node.get("xyz", "0 0 0").split()]
        return transform

    def visit(name, transform):
        links[name] = transform
        for joint in children.get(name, []):
            frame = transform @ origin(joint)
            axis_node = joint.find("axis")
            if axis_node is not None and joint.get("type") != "fixed":
                axis = np.array([float(v) for v in axis_node.get("xyz").split()])
                axis /= np.linalg.norm(axis)
                axes[joint.get("name")] = (frame[:3, 3], frame[:3, :3] @ axis)
                value = q.get(joint.get("name"), 0.0)
                motion = np.eye(4)
                if joint.get("type") == "prismatic":
                    motion[:3, 3] = axis * value
                else:
                    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                    motion[:3, :3] = np.eye(3) + math.sin(value) * k + (1 - math.cos(value)) * k @ k
                frame = frame @ motion
            visit(joint.find("child").get("link"), frame)

    for name in [link.get("name") for link in root.findall("link")]:
        if name not in child_links:
            visit(name, np.eye(4))
    return links, axes


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generated = [convert.DEFAULT_OUTPUT, convert.MOBILE_OUTPUT, convert.MOBILE_SCENE]
        cls.before = {path: path.read_bytes() if path.is_file() else b"" for path in cls.generated}
        convert.build_all()
        cls.after = {path: path.read_bytes() for path in cls.generated}
        # Three loaded copies at most (about 831 MB each), to fit the cluster's 8 GB jobs.
        cls.robot = mujoco.MjModel.from_xml_path(str(convert.DEFAULT_OUTPUT))
        cls.model = mujoco.MjModel.from_xml_path(str(convert.SCENE))          # the default: the base fixed
        cls.mobile = mujoco.MjModel.from_xml_path(str(convert.MOBILE_SCENE))  # the drivable base
        cls.xml = ET.fromstring(cls.after[convert.DEFAULT_OUTPUT])
        cls.urdf = ET.parse(convert.DEFAULT_URDF).getroot()
        cls.limits = {j.get("name"): j.find("limit") for j in cls.urdf.findall("joint") if j.find("limit") is not None}

    def home_data(self, model=None):
        model = model or self.model
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        return data

    def qpos(self, data, joint, model=None):
        return float(data.qpos[(model or self.model).joint(joint).qposadr[0]])

    def test_generated_files_are_up_to_date(self):
        for path in self.generated:
            self.assertEqual(self.before[path], self.after[path],
                             f"{path.name} was stale: convert.py rewrote it; commit the new file")

    def test_dimensions(self):
        for model in (self.robot, self.model):
            self.assertEqual((model.nq, model.nv, model.nu, model.na, model.neq), (33, 33, 17, 0, 16))
        mobile = self.mobile
        self.assertEqual((mobile.nq, mobile.nv, mobile.nu, mobile.na, mobile.neq), (39, 39, 20, 3, 16))
        self.assertEqual([self.robot.camera(i).name for i in range(self.robot.ncam)],
                         ["head", "left_wrist", "right_wrist"])
        for model in (self.robot, self.mobile):
            self.assertAlmostEqual(float(model.body_mass.sum()), 85.771, places=2)

    @unittest.skipUnless(VENDOR_URDF.is_file(), "vendor model not present")
    def test_api_joint_angles_mean_the_same_as_in_the_vendor_model(self):
        """l_jointN/r_jointN (vendor URDF, ROS API) and armLN/armRN_joint: same axis directions at the zero pose."""
        _, old = urdf_fk(VENDOR_URDF, {})
        _, new = urdf_fk(convert.DEFAULT_URDF, {})
        # The old root shoulder_base and the new base_link are both x forward, y left, z up.
        for api, joint in convert.API_TO_MODEL.items():
            self.assertGreater(float(old[api][1] @ new[joint][1]), 0.999, f"{api} vs {joint}")

    @unittest.skipUnless(VENDOR_CONVERTER.is_file(), "vendor model not present")
    def test_home_is_the_vendor_home(self):
        spec = importlib.util.spec_from_file_location("vendor_convert_urdf", VENDOR_CONVERTER)
        vendor = importlib.util.module_from_spec(spec)
        previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # keep the vendor folder as delivered
        try:
            spec.loader.exec_module(vendor)
        finally:
            sys.dont_write_bytecode = previous
        self.assertEqual(convert.API_HOME, vendor.HOME)
        data = self.home_data()
        for api, joint in convert.API_TO_MODEL.items():
            self.assertAlmostEqual(self.qpos(data, joint), vendor.HOME[api], places=9)
        self.assertAlmostEqual(self.qpos(data, convert.LIFT), convert.LIFT_HOME, places=9)

    def test_home_satisfies_the_gripper_constraints(self):
        data = self.home_data()
        for prefix in convert.GRIPPERS.values():
            self.assertAlmostEqual(self.qpos(data, f"{prefix}_joint1"), convert.GRIPPER_HOME)
        rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        self.assertLess(float(np.abs(data.efc_pos[rows]).max()), 1e-9)

    def test_actuators_match_the_urdf_limits(self):
        for index in range(self.model.nu):  # in the default model every actuator drives a URDF joint
            joint = self.model.joint(self.model.actuator_trnid[index, 0]).name
            limit = self.limits[joint]
            self.assertEqual(list(self.model.actuator_ctrlrange[index]), [float(limit.get("lower")),
                                                                          float(limit.get("upper"))])
            effort = float(limit.get("effort"))
            self.assertEqual(list(self.model.actuator_forcerange[index]), [-effort, effort])
            self.assertEqual(list(self.model.jnt_actfrcrange[self.model.joint(joint).id]), [-effort, effort])
        names = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.assertEqual(names[:14], [name.replace("_joint", "") for name in convert.ARM_JOINTS])
        # The drivable variant adds the base's three after them; see test_base_limits_follow_the_wheel_motors.
        self.assertEqual([self.mobile.actuator(i).name for i in range(self.mobile.nu)], names + BASE)

    def test_the_default_base_is_fixed(self):
        """As in the vendor's simulator: base_link hangs on the world with no joint, and the wheels are welded."""
        for body in ("base_link", "wheel1_wheel_link", "wheel2_wheel_link", "wheel3_wheel_link"):
            self.assertEqual(int(self.model.body(body).jntnum[0]), 0, body)
            self.assertGreater(int(self.mobile.body(body).jntnum[0]), 0, body)
        with self.assertRaisesRegex(ValueError, "scene_mobile.xml"):
            mobile_base.MobileBase(self.model)

    def test_scene_mobile_is_scene_with_the_drivable_robot(self):
        scene = convert.SCENE.read_text().splitlines()
        mobile = convert.MOBILE_SCENE.read_text().splitlines()
        self.assertEqual(mobile[3], '  <include file="robot_mobile.xml"/>')
        self.assertEqual(scene[2], '  <include file="robot.xml"/>')
        self.assertEqual(mobile[2:3] + mobile[4:], scene[1:2] + scene[3:])  # all else alike, the header aside

    def test_joint_damping_and_friction_come_from_the_damping_document(self):
        for side in "LR":
            for i, (damping, friction) in enumerate(convert.DAMPING_TABLES["measured"], start=1):
                dof = self.model.joint(f"arm{side}{i}_joint").dofadr[0]
                self.assertAlmostEqual(float(self.model.dof_damping[dof]), damping)
                self.assertAlmostEqual(float(self.model.dof_frictionloss[dof]), friction)
        self.assertEqual(convert.DAMPING_TABLES["measured"][0], (0.165, 0.562))  # docx table 2, joint 1
        for prefix in convert.GRIPPERS.values():
            dof = self.model.joint(f"{prefix}_joint1").dofadr[0]
            self.assertEqual((float(self.model.dof_damping[dof]), float(self.model.dof_frictionloss[dof])), (0.12, 0.08))

    def test_reference_frames_match_urdf_kinematics(self):
        data = self.home_data()
        q = {self.model.joint(j).name: float(data.qpos[self.model.jnt_qposadr[j]]) for j in range(self.model.njnt)}
        links, _ = urdf_fk(convert.DEFAULT_URDF, q)
        offset = np.array([0, 0, convert.BASE_HEIGHT])
        for site in ("virtual_base", "left_shoulder_base", "right_shoulder_base", "left_base_virtual",
                     "right_base_virtual", "left_ee", "right_ee"):
            np.testing.assert_allclose(data.site(site).xpos, links[site][:3, 3] + offset, atol=1e-9, err_msg=site)
            np.testing.assert_allclose(data.site(site).xmat.reshape(3, 3), links[site][:3, :3], atol=1e-9,
                                       err_msg=site)
        for body in ("eye_Link", "lg_link3", "rg_link6", "lg_link8"):
            np.testing.assert_allclose(data.body(body).xpos, links[body][:3, 3] + offset, atol=1e-9, err_msg=body)

    def test_robot_stands_on_the_floor_without_contacts_at_home(self):
        for model in (self.model, self.mobile):
            data = self.home_data(model)
            self.assertEqual(data.ncon, 0)
            lowest = {}
            for geom in range(model.ngeom):
                mesh = model.geom_dataid[geom]
                if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T
                name = model.body(model.geom_bodyid[geom]).name
                lowest[name] = min(lowest.get(name, np.inf), float((vertices[:, 2] + data.geom_xpos[geom][2]).min()))
            self.assertLess(abs(min(lowest[f"wheel{i}_wheel_link"] for i in (1, 2, 3))), 0.001)
            self.assertGreater(min(lowest.values()), -0.001)

    def test_arms_hold_home(self):
        data = self.home_data()
        for _ in range(1000):
            mujoco.mj_step(self.model, data)
        error = max(abs(self.qpos(data, joint) - self.model.key("home").qpos[self.model.joint(joint).qposadr[0]])
                    for joint in convert.ARM_JOINTS)
        self.assertLess(error, 0.005)
        self.assertTrue(np.isfinite(data.qpos).all())

    # The drivable base (scene_mobile.xml)

    def test_base_limits_follow_the_wheel_motors(self):
        """Force limits: the wheel motors' URDF effort along their rolling directions. Speeds: within the motors'."""
        base = mobile_base.MobileBase(self.mobile)
        wrench = np.array([[*rolling, centre[0] * rolling[1] - centre[1] * rolling[0]]
                           for centre, rolling in zip(base.centres, base.rolling)]).T  # wheel forces -> Fx, Fy, Mz
        wheel = self.limits[convert.WHEELS[0]]
        push = float(wheel.get("effort")) / base.radius
        force_x, force_y, torque = push / np.abs(np.linalg.inv(wrench)).max(axis=0)
        self.assertAlmostEqual(min(force_x, force_y), convert.BASE_FORCE, delta=1.0)
        self.assertAlmostEqual(torque, convert.BASE_TORQUE, delta=1.0)
        for name, force in zip(BASE, (convert.BASE_FORCE, convert.BASE_FORCE, convert.BASE_TORQUE)):
            self.assertEqual(list(self.mobile.actuator(name).forcerange), [-force, force])
        limits = np.array([convert.BASE_SPEED, convert.BASE_SPEED, convert.BASE_TURN_RATE])
        fastest = np.abs(wrench.T).max(axis=0) * limits / base.radius  # wheel speed at each full-speed command
        self.assertLess(float(fastest.max()), float(wheel.get("velocity")))

    def test_base_drives_and_holds_its_pose(self):
        data = self.home_data(self.mobile)
        base = mobile_base.MobileBase(self.mobile)

        def drive(seconds, **command):
            for _ in range(round(seconds / self.mobile.opt.timestep)):
                base.command(data, **command)
                base.roll_wheels(data)
                mujoco.mj_step(self.mobile, data)

        drive(2.0, forward=0.3)
        # Straight ahead the front wheels roll in opposite directions and the rear wheel only slides on its rollers.
        np.testing.assert_allclose(base.wheel_speeds(data), [2.896, -2.896, 0.0], atol=0.01)
        drive(1.0)
        np.testing.assert_allclose(base.pose(data), (0.6, 0.0, 0.0), atol=1e-3)
        drive(2.0, turn=0.5)  # about base_link's origin
        np.testing.assert_allclose(base.wheel_speeds(data), [1.615, 1.615, 1.988], atol=0.01)
        drive(1.0)
        np.testing.assert_allclose(base.pose(data), (0.6, 0.0, 1.0), atol=1e-3)
        drive(2.0, left=0.2)  # 0.4 m to the robot's left, which now points 1 rad from +y
        drive(1.0)
        np.testing.assert_allclose(base.pose(data), (0.6 - 0.4 * math.sin(1.0), 0.4 * math.cos(1.0), 1.0), atol=1e-3)
        for joint in convert.ARM_JOINTS:  # back at home, within the dead band that joint friction leaves the servo
            home = self.mobile.key("home").qpos[self.mobile.joint(joint).qposadr[0]]
            error = self.qpos(data, joint, self.mobile) - home
            friction = float(self.mobile.dof_frictionloss[self.mobile.joint(joint).dofadr[0]])
            self.assertLess(abs(error), 1.1 * friction / convert.ARM_KP, joint)
        self.assertEqual(data.ncon, 0)
        self.assertTrue(np.isfinite(data.qpos).all())

    def test_wheels_roll_without_slipping(self):
        """At any base velocity, each wheel's lowest point has no velocity along the wheel's rolling direction."""
        data = self.home_data(self.mobile)
        base = mobile_base.MobileBase(self.mobile)
        data.qpos[base.qadr[2]] = 0.7  # turned, so that the world and base frames differ
        data.qvel[base.dadr] = (0.3, -0.2, 0.5)
        base.roll_wheels(data)
        mujoco.mj_forward(self.mobile, data)
        velocity = np.zeros(6)
        for wheel in convert.WHEELS:
            joint = self.mobile.joint(wheel)
            # [rot, lin] at the body frame's origin, the wheel centre (mjOBJ_BODY would give it at the centre of mass)
            mujoco.mj_objectVelocity(self.mobile, data, mujoco.mjtObj.mjOBJ_XBODY, joint.bodyid[0], velocity, 0)
            bottom = velocity[3:] + np.cross(velocity[:3], [0.0, 0.0, -base.radius])
            rolling = np.cross(data.xaxis[joint.id], [0.0, 0.0, 1.0])
            self.assertLess(abs(float(bottom @ rolling)), 1e-9, wheel)
        self.assertGreater(float(np.abs(data.qvel[base.wheel_dofs]).min()), 0.5)

    def test_base_holds_its_pose_while_the_arms_move(self):
        data = self.home_data(self.mobile)
        base = mobile_base.MobileBase(self.mobile)
        arms = np.array([self.mobile.actuator(name.replace("_joint", "")).id for name in convert.ARM_JOINTS])
        swing = data.ctrl[arms] + np.tile([0.4, 0.4, 0, 0, 0, 0, 0], 2)
        drift = 0.0
        for target in (np.clip(swing, *self.mobile.actuator_ctrlrange[arms].T), data.ctrl[arms].copy()):
            data.ctrl[arms] = target
            for _ in range(500):
                base.roll_wheels(data)
                mujoco.mj_step(self.mobile, data)
                drift = max(drift, float(np.hypot(*base.pose(data)[:2])))
        self.assertLess(drift, 0.002)
        self.assertLess(abs(base.pose(data)[2]), 0.002)

    def test_gripper_input_drives_the_fingers(self):
        data = self.home_data()
        for target in (0.0, 1.0, 0.4):
            data.ctrl[self.model.actuator("left_gripper").id] = target
            data.ctrl[self.model.actuator("right_gripper").id] = target
            for _ in range(750):
                mujoco.mj_step(self.model, data)
            for prefix in convert.GRIPPERS.values():
                value = self.qpos(data, f"{prefix}_joint1")
                self.assertAlmostEqual(value, target, delta=0.01)
                self.assertAlmostEqual(self.qpos(data, f"{prefix}_joint2"), convert.GRIPPER_COUPLING * value, delta=1e-3)
                self.assertAlmostEqual(self.qpos(data, f"{prefix}_joint3"), -self.qpos(data, f"{prefix}_joint2"),
                                       delta=1e-3)  # mimic multiplier -1

    def test_grasp_a_box_and_raise_the_lift(self):
        data = self.home_data()
        middle = 0.5 * (data.body("lg_link3").xipos + data.body("lg_link6").xipos)
        spec = mujoco.MjSpec.from_file(str(HERE / "scene.xml"))
        box = spec.worldbody.add_body(name="box", pos=middle)
        box.add_freejoint()
        box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], mass=0.1)
        model = spec.compile()
        data = mujoco.MjData(model)
        data.qpos[:self.model.nq] = self.model.key("home").qpos
        data.qpos[self.model.nq:self.model.nq + 4] = [*middle, 1]
        data.ctrl[:] = self.model.key("home").ctrl
        box_id = model.body("box").id
        model.body_gravcomp[box_id] = 1  # float the box while the gripper closes on it
        data.ctrl[model.actuator("left_gripper").id] = 0.0
        for _ in range(500):
            mujoco.mj_step(model, data)
        model.body_gravcomp[box_id] = 0
        for _ in range(500):
            mujoco.mj_step(model, data)
        start = float(data.body("box").xpos[2])
        data.ctrl[model.actuator("lift").id] -= 0.10  # 10 cm up (positive lift lowers the torso)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        self.assertGreater(float(data.body("box").xpos[2]) - start, 0.09)
        self.assertGreater(float(data.qpos[model.joint("lg_joint1").qposadr[0]]), 0.2)  # stopped by the box
        self.assertTrue(np.isfinite(data.qpos).all())

    def test_cameras_look_where_expected(self):
        data = self.home_data()
        forward = -data.cam_xmat[self.model.camera("head").id].reshape(3, 3)[:, 2]
        self.assertAlmostEqual(math.degrees(math.asin(-forward[2])), 26.0, delta=0.5)  # 26 deg down
        self.assertGreater(forward[0], 0.85)                                           # robot's front is +x
        for camera, prefix in (("left_wrist", "lg"), ("right_wrist", "rg")):
            frame = data.cam_xmat[self.model.camera(camera).id].reshape(3, 3)
            tips = 0.5 * (data.body(f"{prefix}_link3").xipos + data.body(f"{prefix}_link6").xipos)
            to_tips = tips - data.cam_xpos[self.model.camera(camera).id]
            self.assertGreater(float(-frame[:, 2] @ to_tips / np.linalg.norm(to_tips)), 0.9, camera)

    def test_assets(self):
        meshes = self.xml.findall("asset/mesh")
        self.assertEqual(len(meshes), 41 + 323)
        for mesh in meshes:
            path = (HERE / mesh.get("file")).resolve()
            self.assertTrue(path.is_file(), path)
            self.assertNotIn("vendor/collision", mesh.get("file"))  # raw STLs are too big for MuJoCo
        for texture in self.xml.findall("asset/texture"):
            self.assertTrue((HERE / texture.get("file")).is_file())
        self.assertFalse(any(element.get("euler") is not None for element in self.xml.iter()))

    def test_ros_node_maps_the_api_names(self):
        if importlib.util.find_spec("rclpy") is None:
            self.skipTest("ROS 2 not available")
        import ros2_sim_node
        for api, joint in convert.API_TO_MODEL.items():
            self.assertEqual(ros2_sim_node.model_joint(api), joint)
            index = ros2_sim_node.actuator_for(self.model, joint)
            self.assertEqual(self.model.joint(self.model.actuator_trnid[index, 0]).name, joint)


if __name__ == "__main__":
    unittest.main()
