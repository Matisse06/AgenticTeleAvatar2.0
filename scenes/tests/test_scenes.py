"""Tests of the scenes (scenes/*.xml). Every scene passes the checks in SceneChecks: the robot in it is model/robot.xml
unchanged (physics options too), its 'home' keyframe holds the robot's home and puts each object where the scene
places it, nothing moves at home, and the head camera sees every object. Its drivable-base twin
(scenes/<name>_mobile.xml, from scenes/build_mobile.py) passes the same checks against model/robot_mobile.xml, and its
base drives.

  python3 -m unittest discover -s scenes/tests -v

Each scene file needs a test class at the bottom (a test checks that), which can add the scene's own checks; the
twin's class is made from it. Needs the robot model built (model/README.md: unpack the vendor meshes, then run
model/convert.py).
"""
import ctypes
import gc
import os
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering for the camera check; set before importing mujoco

import mujoco
import numpy as np

SCENES = Path(__file__).resolve().parents[1]
MODEL = SCENES.parent / "model"
sys.path[:0] = [str(SCENES), str(MODEL)]
import build_mobile  # noqa: E402
from mobile_base import MobileBase, drivable_variant  # noqa: E402

QPOS_SIZE = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4,
             int(mujoco.mjtJoint.mjJNT_SLIDE): 1, int(mujoco.mjtJoint.mjJNT_HINGE): 1}
ROBOTS = {}  # each robot file loaded once


def without_visuals(path: Path) -> mujoco.MjModel:
    """A robot or scene file compiled without the robot's visual meshes (class 'visual': drawn only, no mass, no
    contacts). Its dynamics are the full model's, field for field, at about 50 MB instead of 0.8 GB. Loading the full
    model peaks about 3 GB higher, and the process keeps most of that, so only the camera check loads it."""
    spec = mujoco.MjSpec.from_file(str(path))
    for geom in [geom for geom in spec.geoms if geom.classname.name == "visual"]:
        spec.delete(geom)
    for kind, used in (("meshes", {geom.meshname for geom in spec.geoms}),
                       ("materials", {geom.material for geom in spec.geoms})):
        for element in [element for element in getattr(spec, kind) if element.name not in used]:
            spec.delete(element)
    used = {name for material in spec.materials for name in material.textures if name}
    for texture in [texture for texture in spec.textures if texture.name not in used]:
        spec.delete(texture)
    return spec.compile()


def release_memory() -> None:
    """Hand freed memory back to the system (glibc keeps it otherwise), so that the next full model loads lower."""
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):  # not glibc
        pass


class SceneChecks:
    """The checks every scene passes. Subclasses set `path`, one per scene file."""
    path: Path
    robot_path = MODEL / "robot.xml"  # the robot the scene attaches; twins: robot_mobile.xml

    @classmethod
    def setUpClass(cls):
        if cls.robot_path not in ROBOTS:
            ROBOTS[cls.robot_path] = without_visuals(cls.robot_path)
        cls.robot = ROBOTS[cls.robot_path]
        cls.model = without_visuals(cls.path)  # the camera check loads the full scene
        cls.objects = [cls.model.jnt_bodyid[j] for j in range(cls.model.njnt)
                       if cls.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        names = {cls.robot.body(b).name for b in range(1, cls.robot.nbody)}
        cls.robot_bodies = {b for b in range(cls.model.nbody) if cls.model.body(b).name in names}

    def home_data(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.model.key("home").id)
        mujoco.mj_forward(self.model, data)
        return data

    def test_physics_options_are_the_robots(self):
        for name in dir(self.robot.opt):
            value = getattr(self.robot.opt, name)
            if not name.startswith("_") and not callable(value):
                np.testing.assert_array_equal(getattr(self.model.opt, name), value, err_msg=name)

    def test_the_robot_is_unchanged(self):
        robot, model = self.robot, self.model
        for j in range(robot.njnt):
            name = robot.joint(j).name
            mine, theirs = model.joint(name), robot.joint(j)
            for field in ("type", "pos", "axis", "range", "limited", "stiffness", "damping", "armature",
                          "frictionloss"):
                np.testing.assert_array_equal(getattr(mine, field), getattr(theirs, field), err_msg=f"{name} {field}")
            self.assertEqual(model.jnt_actgravcomp[mine.id], robot.jnt_actgravcomp[j], name)
        for b in range(1, robot.nbody):
            name = robot.body(b).name
            for field in ("mass", "inertia", "pos", "quat", "ipos", "iquat"):
                np.testing.assert_array_equal(getattr(model.body(name), field), getattr(robot.body(b), field),
                                              err_msg=f"{name} {field}")
            self.assertEqual(model.body_gravcomp[model.body(name).id], robot.body_gravcomp[b], name)
            self.assertEqual(model.body(model.body(name).parentid[0]).name, robot.body(robot.body(b).parentid[0]).name)
        self.assertEqual(model.nu, robot.nu)  # scenes add objects, not actuators
        for a in range(robot.nu):
            mine, theirs = model.actuator(a), robot.actuator(a)
            self.assertEqual(mine.name, theirs.name)
            self.assertEqual(model.joint(mine.trnid[0]).name, robot.joint(theirs.trnid[0]).name)
            for field in ("gainprm", "biasprm", "ctrlrange", "forcerange", "gear", "dyntype", "gaintype", "biastype"):
                np.testing.assert_array_equal(getattr(mine, field), getattr(theirs, field),
                                              err_msg=f"{mine.name} {field}")
        for e in range(robot.neq):
            mine, theirs = model.equality(robot.equality(e).name), robot.equality(e)
            for field in ("type", "data", "solref", "solimp"):
                np.testing.assert_array_equal(getattr(mine, field), getattr(theirs, field), err_msg=theirs.name)
        pairs = lambda m: {(m.body(b1).name, m.body(b2).name) for b1, b2 in zip(m.exclude_signature >> 16,
                                                                                 m.exclude_signature & 0xFFFF)}
        self.assertEqual(pairs(model), pairs(robot))
        for g in range(robot.ngeom):  # the collision geoms; the camera check looks for the visual ones
            mine, theirs = model.geom(robot.geom(g).name), robot.geom(g)
            for field in ("type", "size", "pos", "quat", "contype", "conaffinity", "condim", "friction", "solref",
                          "solimp", "margin", "group"):
                np.testing.assert_array_equal(getattr(mine, field), getattr(theirs, field),
                                              err_msg=f"{theirs.name} {field}")

    def test_home_is_the_robots_home_with_the_objects_in_place(self):
        key, robot_key = self.model.key("home"), self.robot.key("home")
        for j in range(self.robot.njnt):
            start, size = self.robot.jnt_qposadr[j], QPOS_SIZE[int(self.robot.jnt_type[j])]
            mine = self.model.joint(self.robot.joint(j).name).qposadr[0]
            np.testing.assert_array_equal(key.qpos[mine:mine + size], robot_key.qpos[start:start + size],
                                          err_msg=self.robot.joint(j).name)
        np.testing.assert_array_equal(key.ctrl, robot_key.ctrl)
        for body in self.objects:
            start = self.model.jnt_qposadr[self.model.body_jntadr[body]]
            # where the scene puts the body (qpos0), not the world origin that an <include>d robot's 'home' gives
            np.testing.assert_array_equal(key.qpos[start:start + 7], self.model.qpos0[start:start + 7],
                                          err_msg=self.model.body(body).name)

    def test_nothing_moves_at_home(self):
        data = self.home_data()
        for i in range(data.ncon):  # the objects rest on their supports, the robot touches nothing
            bodies = {self.model.geom_bodyid[data.contact[i].geom1], self.model.geom_bodyid[data.contact[i].geom2]}
            self.assertFalse(bodies & self.robot_bodies, [self.model.body(b).name for b in bodies])
        objects = data.xpos[self.objects].copy()
        robot_qpos = [self.model.jnt_qposadr[j] for j in range(self.model.njnt)
                      if self.model.jnt_bodyid[j] in self.robot_bodies]
        home = data.qpos[robot_qpos].copy()
        for _ in range(500):  # 1 s
            mujoco.mj_step(self.model, data)
        self.assertLess(float(np.abs(data.xpos[self.objects] - objects).max()), 0.001)
        self.assertLess(float(np.abs(data.qpos[robot_qpos] - home).max()), 0.005)
        self.assertTrue(np.isfinite(data.qpos).all())

    def test_the_head_camera_sees_every_object(self):
        pixels = self.head_camera_pixels()
        release_memory()  # before the next scene's full load
        for name, count in pixels.items():
            self.assertGreater(count, 20, name)

    def head_camera_pixels(self) -> dict:
        """Each object's pixels in a 480 x 480 head image at home (square, as the robot's 960 x 960 head images: 120 deg
        either way). This needs the full scene, since the arms' visual meshes could hide an object, so it also checks
        that the scene has every visual geom of the robot file."""
        model = mujoco.MjModel.from_xml_path(str(self.path))
        names = {geom.get("name") for geom in ET.parse(self.robot_path).find("worldbody").iter("geom")}
        self.assertFalse({name for name in names if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0})
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, 480, 480) as renderer:
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera="head")
            ids, types = np.moveaxis(renderer.render(), -1, 0)
        geoms = ids[types == mujoco.mjtObj.mjOBJ_GEOM]
        objects = [model.body(self.model.body(body).name).id for body in self.objects]
        return {model.body(body).name: int(np.isin(geoms, np.flatnonzero(model.geom_bodyid == body)).sum())
                for body in objects}

    def test_views_for_people(self):
        for camera in ("overview", "front", "left_side"):  # overview: render_video.py's default
            self.assertEqual(self.model.cam_bodyid[self.model.camera(camera).id], 0, camera)


class TableCubes(SceneChecks, unittest.TestCase):
    path = SCENES / "table_cubes.xml"

    def test_cubes_rest_on_the_table_where_both_arms_reach_them(self):
        data = self.home_data()
        top = data.geom("table_top").xpos[2] + self.model.geom("table_top").size[2]
        self.assertAlmostEqual(top, 0.75, places=9)
        for name, colour in (("red_cube", 0), ("green_cube", 1)):
            cube = self.model.geom(name)
            np.testing.assert_array_equal(cube.size, [0.02, 0.02, 0.02])
            self.assertAlmostEqual(float(self.model.body(name).mass[0]), 0.064)
            self.assertEqual(int(np.argmax(cube.rgba[:3])), colour, name)
            self.assertAlmostEqual(data.geom(name).xpos[2] - cube.size[2], top, places=9)  # resting on the top
            x, y = data.body(name).xpos[:2]
            # the strip where both arms grasp from the front or the side with joint margin to spare (README)
            self.assertTrue(0.35 <= x <= 0.45 and abs(y) <= 0.05, f"{name} at {x:.3f}, {y:.3f}")


class DrivableChecks:
    """What a scene's drivable-base twin passes on top of its scene's checks."""

    def test_the_base_drives(self):
        data = self.home_data()
        base = MobileBase(self.model)
        objects = data.xpos[self.objects].copy()
        for step in range(1000):  # 0.2 m back: 0.2 m/s for 1 s (ramped), then stop
            base.command(data, forward=-0.2 if step < 500 else 0.0)
            base.roll_wheels(data)
            mujoco.mj_step(self.model, data)
        np.testing.assert_allclose(base.pose(data), (-0.2, 0.0, 0.0), atol=0.002)
        self.assertLess(float(np.abs(data.xpos[self.objects] - objects).max()), 0.001)


def twin_tests(scene_tests: type) -> type:
    """Tests of a scene's drivable-base twin: the scene's checks, against robot_mobile.xml, and DrivableChecks."""
    return type(f"{scene_tests.__name__}Mobile", (DrivableChecks, scene_tests),
                {"path": drivable_variant(scene_tests.path), "robot_path": MODEL / "robot_mobile.xml"})


globals().update({tests.__name__: tests for tests in map(twin_tests, SceneChecks.__subclasses__())})


class EveryScene(unittest.TestCase):
    def test_every_scene_file_has_a_test_class(self):
        tested = {cls.path.name for cls in SceneChecks.__subclasses__()}
        self.assertEqual(tested, {path.name for path in build_mobile.sources()})

    def test_every_scene_has_an_up_to_date_drivable_twin(self):
        sources = build_mobile.sources()
        twins = {path.name for path in SCENES.glob("*.xml")} - {path.name for path in sources}
        self.assertEqual(twins, {drivable_variant(path).name for path in sources}, "run python3 scenes/build_mobile.py")
        for source in sources:
            self.assertEqual(drivable_variant(source).read_text(encoding="utf-8"), build_mobile.twin(source),
                             f"{drivable_variant(source).name} is out of date: run python3 scenes/build_mobile.py")


if __name__ == "__main__":
    unittest.main()
