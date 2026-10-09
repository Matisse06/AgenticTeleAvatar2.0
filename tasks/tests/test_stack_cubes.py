"""Tests of the stacking task (tasks/): layouts, stepping, the success check, the scripted expert and the vendor's view.

  python3 -m unittest discover -s tasks/tests -v

Needs the robot model built (model/README.md). Physics runs on the scene compiled without the robot's visual meshes
(the same dynamics, a fraction of the memory); only the vendor-view test loads the full scene, to render the eyes.
"""
import math
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; set before importing mujoco

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tasks import stack_cubes  # noqa: E402
from tasks.expert import StackExpert  # noqa: E402
from tasks.stack_cubes import CONTROL_HZ, StackCubes, sample_layout  # noqa: E402
from sweep_contacts import compile_without_visuals  # noqa: E402  (model/sweep_contacts.py)

MODEL = compile_without_visuals(stack_cubes.SCENE)


def home_action(task: StackCubes) -> np.ndarray:
    """Hold home: the arms' home targets, grippers open."""
    action = np.zeros(16)
    for i, side in enumerate(stack_cubes.SIDES):
        action[8 * i:8 * i + 7] = task.model.key("home").ctrl[task.arm[side]]
        action[8 * i + 7] = 2.0
    return action


class Layouts(unittest.TestCase):
    def test_layouts_are_seeded_and_inside_the_reach_band(self):
        self.assertEqual(sample_layout(7), sample_layout(7))
        self.assertNotEqual(sample_layout(7), sample_layout(8))
        for seed in range(300):
            layout = sample_layout(seed)
            (x1, y1, yaw1), (x2, y2, yaw2) = layout["red_cube"], layout["green_cube"]
            for x, y, yaw in layout.values():
                self.assertTrue(stack_cubes.LAYOUT_X[0] <= x <= stack_cubes.LAYOUT_X[1], seed)
                self.assertLessEqual(abs(y), stack_cubes.band_half_width(x), seed)
                self.assertTrue(0 <= yaw < 2 * math.pi, seed)
            self.assertGreaterEqual(math.hypot(x1 - x2, y1 - y2), stack_cubes.MIN_SEPARATION, seed)


class Stepping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = StackCubes(MODEL)

    def test_reset_leaves_the_robot_at_home_and_the_cubes_at_rest(self):
        task = self.task
        observation = task.reset(3)
        self.assertEqual((task.steps, task.data.time), (0, 0.0))
        home = task.model.key("home")
        for side in stack_cubes.SIDES:
            np.testing.assert_allclose(observation[f"{side}_arm"], home.ctrl[task.arm[side]], atol=1e-3)
        for cube, (x, y, yaw) in task.layout.items():
            position, quat, speed, turn = task.cube_pose(cube)
            np.testing.assert_allclose(position, [x, y, 0.77], atol=5e-4)
            self.assertLess(float(np.linalg.norm(speed)), 1e-3)
            self.assertAlmostEqual(2 * math.atan2(quat[3], quat[0]) % (2 * math.pi), yaw % (2 * math.pi), places=3)
        self.assertNotIn("red_cube", str(observation))  # the observation holds the robot's state only

    def test_control_runs_at_30_hz(self):
        task = self.task
        task.reset(0)
        counts, before = [], task.data.time
        for _ in range(CONTROL_HZ):
            task.step(home_action(task))
            counts.append(round((task.data.time - before) / task.model.opt.timestep))
            before = task.data.time
        self.assertAlmostEqual(task.data.time, 1.0, places=9)
        self.assertEqual(set(counts), {16, 17})  # 2 ms physics steps: 16 2/3 per 1/30 s

    def test_actions_are_clipped_to_the_ranges(self):
        task = self.task
        task.reset(0)
        action = home_action(task)
        action[0], action[7] = 9.0, -9.0  # left joint 1 far past its range; the left gripper past full closing torque
        task.step(action)
        self.assertEqual(task.data.ctrl[task.arm["left"][0]], task.model.actuator_ctrlrange[task.arm["left"][0], 1])
        self.assertEqual(task.data.ctrl[task.gripper["left"]], -1.6)
        with self.assertRaises(ValueError):
            task.step(np.zeros(15))
        with self.assertRaises(ValueError):
            task.step(np.full(16, np.nan))


class Success(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = StackCubes(MODEL)

    def place_top(self, offset, steps=200):
        """The top cube put on the bottom one (offset from exactly on top), then left to settle."""
        task = self.task
        task.reset(0)
        bottom = task.cube_pose(task.bottom)[0]
        adr = task.model.jnt_qposadr[task.cube_joint[task.top]]
        task.data.qpos[adr:adr + 7] = [*(bottom + [0, 0, 0.04] + offset), 1, 0, 0, 0]
        task.data.qvel[task.model.jnt_dofadr[task.cube_joint[task.top]]:][:6] = 0
        info = None
        for _ in range(steps // 17 + 1):
            _, info = task.step(home_action(task))
        return info

    def test_a_cube_resting_on_the_other_succeeds(self):
        info = self.place_top(np.array([0.0, 0.0, 0.001]))
        self.assertTrue(info["on"] and info["static"] and not info["grasped"] and info["success"], info)

    def test_a_cube_beside_the_other_does_not(self):
        info = self.place_top(np.array([0.05, 0.0, -0.04]))  # on the table, 5 cm away
        self.assertFalse(info["on"] or info["success"], info)

    def test_instruction_names_the_cubes(self):
        self.assertEqual(self.task.instruction, "stack the red cube on the green cube")
        self.assertEqual(StackCubes(MODEL, top="green_cube").instruction, "stack the green cube on the red cube")


class Expert(unittest.TestCase):
    """The scripted expert stacks the cubes, with either arm (seeds where it picks the left, then the right)."""

    def test_the_expert_stacks_the_cubes(self):
        task = StackCubes(MODEL)
        expert = StackExpert(task)
        arms, held = set(), False
        for seed in (3, 4):
            task.reset(seed)
            expert.reset()
            info = {"success": False}
            while task.steps < task.max_steps and not info["success"]:
                for action in expert.infer(task.observation(), task.steps)[:16]:
                    _, info = task.step(action)
                    held |= info["grasped"]
                    if info["success"]:
                        break
            self.assertTrue(info["success"], (seed, expert.choice))
            arms.add(expert.choice["side"])
        self.assertTrue(held)  # the grasp check saw the cube held on the way
        self.assertEqual(arms, {"left", "right"})


class VendorView(unittest.TestCase):
    def test_the_vendor_observation(self):
        from tasks.openpi import OpenpiView
        task = StackCubes()  # the full scene: the eyes are rendered
        view = OpenpiView(task)
        try:
            observation = task.reset(5)
            request = view.observation()
            state = request["observation/state"]
            self.assertEqual((state.shape, state.dtype), ((48,), np.float32))
            np.testing.assert_allclose(state[0:7], observation["left_arm"], atol=1e-6)
            np.testing.assert_allclose(state[8:15], observation["right_arm"], atol=1e-6)
            self.assertEqual(float(np.abs(state[[7, 15, 23, 31, 39, 47]]).max()), 0.0)  # as the vendor's client
            for key, shape in (("head_camera", (960, 960, 3)), ("left_color", (400, 640, 3)),
                               ("right_color", (400, 640, 3))):
                image = request[f"observation/images/{key}"]
                self.assertEqual((image.shape, image.dtype), (shape, np.uint8), key)
                self.assertGreater(float(image.std()), 5.0, key)  # not blank
            self.assertEqual(request["prompt"], "stack the red cube on the green cube")
        finally:
            view.close()


if __name__ == "__main__":
    unittest.main()
