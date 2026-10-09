"""The vendor's pi0.5 interface for TeleAvatar v2 (github.com/dexteleop/openpi at c7616ac, examples/teleavatar_v2 and
src/openpi/policies/teleavatar_v2_policy.py), on top of a task: what their client sends the policy server, and what
it does with the answer. A policy served with their configs runs on the simulator unchanged.

  view = OpenpiView(task)                        # renders the three eyes the policy uses
  request = view.observation()                   # the dict their client sends (below)
  chunk = server.infer(request)["actions"]       # (30, 16); execute the first 16 with task.step, then ask again

The request (their env.py and ros2_interface.py):
- "observation/state": float32[48]. [0:7] the left arm's joint angles (rad), [8:15] the right arm's, [16:23] and
  [24:31] their velocities (rad/s), [32:39] and [40:47] their torques (N m). The gripper slots (7, 15, 23, 31, 39, 47)
  stay 0, as on the robot: their client never reads the grippers. The policy uses [0:7] and [8:15] only.
- "observation/images/head_camera": uint8[960, 960, 3] RGB, the head's left eye
- "observation/images/left_color": uint8[400, 640, 3], the left wrist's right eye (the inner one)
- "observation/images/right_color": uint8[400, 640, 3], the right wrist's left eye
- "prompt": the task's instruction (case matters: their tokenizer does not lowercase)

The answer is a chunk of 30 actions in the task's own layout: per arm (left first) seven absolute joint targets (rad)
and the gripper's motor torque (N m). Their client steps them at 30 Hz, 16 per query, which StackCubes.step does too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from cameras import FACE_SIZE, POLICY_EYES, FisheyeCamera  # noqa: E402  (model/cameras.py)

CHUNK, EXECUTED = 30, 16  # actions per answer, executed per query (their main.py: action_horizon, open_loop_horizon)


class OpenpiView:
    """Renders the policy's three eyes through their lenses (one shared renderer for their cube faces)."""

    def __init__(self, task) -> None:
        self.task = task
        self.renderer = mujoco.Renderer(task.model, FACE_SIZE, FACE_SIZE)
        self.cameras = {key: FisheyeCamera(task.model, eye, renderer=self.renderer)
                        for key, eye in POLICY_EYES.items()}

    def images(self) -> dict:
        """{key: image} of the policy's eyes: head_camera (960 x 960), left_color and right_color (400 x 640)."""
        return {key: camera.render(self.task.data) for key, camera in self.cameras.items()}

    def observation(self, images: dict | None = None) -> dict:
        obs = self.task.observation()
        state = np.zeros(48, np.float32)
        for offset, side in ((0, "left"), (8, "right")):
            state[offset:offset + 7] = obs[f"{side}_arm"]
            state[16 + offset:16 + offset + 7] = obs[f"{side}_arm_velocity"]
            state[32 + offset:32 + offset + 7] = obs[f"{side}_arm_torque"]
        images = self.images() if images is None else images
        return {"observation/state": state, **{f"observation/images/{key}": image for key, image in images.items()},
                "prompt": self.task.instruction}

    def close(self) -> None:
        for camera in self.cameras.values():
            camera.close()
        self.renderer.close()
