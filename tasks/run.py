#!/usr/bin/env python3
"""Run a policy on the stacking task and record how it does.

  python3 tasks/run.py --policy expert --seeds 0 1 2 --out outputs/tasks/expert            # the scripted expert
  python3 tasks/run.py --policy expert --seeds 0 --out outputs/tasks/expert --video         # and an MP4 per episode
  python3 tasks/run.py --policy openpi --port 8000 --seeds 100000 100001 --out outputs/tasks/pi05

The loop is the one the vendor's client runs on the robot: ask the policy for a chunk of 30 actions, execute the
first 16 at 30 Hz, ask again (the user's ManiSkill evaluation ran the same loop with chunks of 10, 5 executed, at
20 Hz). The simulation waits while the policy computes, as it did there, so latency never changes the outcome. An
episode ends at the first step that succeeds (StackCubes.evaluate) or after 15 s. Policies:
- expert: tasks/expert.py, scripted side grasps from the true cube poses (privileged)
- openpi: a pi0/pi0.5 policy server (scripts/serve_policy.py in the vendor's openpi), through their websocket client
  (pip install openpi-client); it gets the vendor's observation dict (tasks/openpi.py) with the three eye images

Outputs in --out: results.json (every episode and a summary, rewritten after each), episodes/seed<N>.jsonl (each
step: the action, the cubes' true poses and the success checks) and, with --video, videos/seed<N>.mp4 (the overview
camera and the three eyes the policy sees, at 30 fps; rendering needs a GPU to be quick: each frame is about 12
renders). Held-out evaluation seeds start at 100000.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks.openpi import EXECUTED, OpenpiView  # noqa: E402
from tasks.stack_cubes import CONTROL_HZ, CUBES, StackCubes  # noqa: E402


class ExpertPolicy:
    """The scripted expert (tasks/expert.py); it reads the true cube poses, not the observation."""
    needs_images = False

    def __init__(self, task: StackCubes) -> None:
        from tasks.expert import StackExpert
        self.expert = StackExpert(task)

    def reset(self) -> None:
        self.expert.reset()

    def infer(self, observation: dict, step: int) -> np.ndarray:
        return self.expert.infer(observation, step)

    def describe(self) -> dict:
        return {"policy": "expert", "choice": self.expert.choice}


class OpenpiPolicy:
    """A pi0/pi0.5 policy server, through the vendor's websocket client."""
    needs_images = True

    def __init__(self, host: str, port: int) -> None:
        try:
            from openpi_client import websocket_client_policy
        except ImportError as error:
            raise SystemExit("--policy openpi needs the openpi client: pip install openpi-client") from error
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)  # waits for the server
        self.metadata = self.client.get_server_metadata()

    def reset(self) -> None:
        pass

    def infer(self, observation: dict, step: int) -> np.ndarray:
        return np.asarray(self.client.infer(observation)["actions"], dtype=float)

    def describe(self) -> dict:
        return {"policy": "openpi", "server_metadata": self.metadata}


def video_frame(task: StackCubes, view: OpenpiView, scene_renderer, eyes: dict | None = None) -> np.ndarray:
    """The overview camera (640 x 480) beside the head eye (480 x 480), the wrist eyes (384 x 240 each) below."""
    from PIL import Image
    eyes = view.images() if eyes is None else eyes
    scene_renderer.update_scene(task.data, camera="overview")
    small = lambda image, size: np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))
    frame = np.zeros((480 + 240, 1120, 3), np.uint8)
    frame[:480, :640] = scene_renderer.render()
    frame[:480, 640:] = small(eyes["head_camera"], (480, 480))
    frame[480:, :384] = small(eyes["left_color"], (384, 240))
    frame[480:, 384:768] = small(eyes["right_color"], (384, 240))
    return frame


def run_episode(task: StackCubes, policy, seed: int, view: OpenpiView | None, out: Path, scene_renderer=None) -> dict:
    started = time.time()
    observation = task.reset(seed)
    policy.reset()
    queue, latencies, frames, info = [], [], [], {"success": False}
    log = (out / "episodes" / f"seed{seed}.jsonl").open("w")
    try:
        while task.steps < task.max_steps:
            images = None
            if not queue:
                if policy.needs_images or scene_renderer is not None:
                    images = view.images()
                request = view.observation(images) if policy.needs_images else observation
                begin = time.time()
                chunk = np.asarray(policy.infer(request, task.steps), dtype=float)
                latencies.append(time.time() - begin)
                if chunk.ndim != 2 or chunk.shape[1] != 16 or len(chunk) == 0:
                    raise ValueError(f"the policy returned actions of shape {chunk.shape}, not (n, 16)")
                queue = list(chunk[:EXECUTED])
            if scene_renderer is not None:
                frames.append(video_frame(task, view, scene_renderer, images))
            action = queue.pop(0)
            observation, info = task.step(action)
            poses = {cube: np.round(np.concatenate(task.cube_pose(cube)[:2]), 5).tolist() for cube in CUBES}
            log.write(json.dumps({"step": task.steps, "action": np.round(action, 5).tolist(), **poses,
                                  **{k: info[k] for k in ("success", "on", "static", "grasped")}}) + "\n")
            if info["success"]:
                break
    finally:
        log.close()
    if frames:
        import imageio
        (out / "videos").mkdir(exist_ok=True)
        imageio.mimwrite(out / "videos" / f"seed{seed}.mp4", frames, fps=CONTROL_HZ)
    return {"seed": seed, "success": bool(info["success"]), "steps": task.steps, "sim_time_s": task.steps / CONTROL_HZ,
            "queries": len(latencies), "query_s": [round(t, 4) for t in latencies],
            "wall_time_s": round(time.time() - started, 2), "layout": task.layout, **policy.describe()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=("expert", "openpi"), default="expert")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", default="red_cube", help="the cube to put on the other one")
    parser.add_argument("--video", action="store_true", help="an MP4 per episode (videos/)")
    parser.add_argument("--host", default="localhost", help="openpi policy server")
    parser.add_argument("--port", type=int, default=8000, help="openpi policy server")
    args = parser.parse_args()
    (args.out / "episodes").mkdir(parents=True, exist_ok=True)
    task = StackCubes(top=args.top)
    policy = ExpertPolicy(task) if args.policy == "expert" else OpenpiPolicy(args.host, args.port)
    view = OpenpiView(task) if policy.needs_images or args.video else None
    scene_renderer = None
    if args.video:
        import mujoco
        scene_renderer = mujoco.Renderer(task.model, 480, 640)
    episodes = []
    try:
        for seed in args.seeds:
            episode = run_episode(task, policy, seed, view, args.out, scene_renderer)
            episodes.append(episode)
            done = [e for e in episodes if e["success"]]
            summary = {"policy": args.policy, "instruction": task.instruction, "episodes": len(episodes),
                       "successes": len(done), "success_rate": len(done) / len(episodes),
                       "mean_sim_time_when_successful_s":
                           float(np.mean([e["sim_time_s"] for e in done])) if done else None}
            (args.out / "results.json").write_text(json.dumps({"summary": summary, "episodes": episodes}, indent=1))
            print(f"seed {seed}: {'success' if episode['success'] else 'failure'} after {episode['steps']} steps "
                  f"({episode['sim_time_s']:.1f} s), {episode['queries']} queries, {episode['wall_time_s']:.0f} s"
                  f" wall | {len(done)}/{len(episodes)}", flush=True)
    finally:  # also when a run stops early: unclosed renderers print EGL errors over the real one at exit
        if view is not None:
            view.close()
        if scene_renderer is not None:
            scene_renderer.close()


if __name__ == "__main__":
    main()
