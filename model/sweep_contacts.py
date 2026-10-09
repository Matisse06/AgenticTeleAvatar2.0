#!/usr/bin/env python3
"""Which body pairs of the robot touch: in random poses (the check behind the contact exclusions in convert.py,
EXCLUDES and the gripper's inner pairs), or in one pose.

  python3 model/sweep_contacts.py                                   # 3000 random poses of model/robot.xml
  python3 model/sweep_contacts.py --poses 10000 --seed 7 --cad      # and each pair's CAD gap at its deepest overlap
  python3 model/sweep_contacts.py --poses 10000 --seed 7 --cad --cad-poses 1000 --pair lg_link7 base_link
  python3 model/sweep_contacts.py --at armL2_joint=1.75 --cad --pair armL2_link body_link   # 'home', joint 2 moved

Random poses draw every arm joint uniformly within its range, the lift within its range and each gripper input within
[0, 1] (the finger linkage follows through its equality constraints); --at takes 'home' and changes the joints named.
MuJoCo's collision detection then runs on the robot alone, without a floor. It never collides a body with its parent,
and the model's exclusions are skipped, so every pair listed can touch in the simulation. It is either a real
self-collision (an arm against the torso, the column or the other arm; the forearm against the upper arm), which
stays, or an overlap of the convex collision pieces where the robot's real parts do not touch, which needs an
exclusion. --cad tells them apart: it measures the gap between the two links' CAD surfaces (their visual meshes:
every vertex plus about one random point per mm^2). Gaps within the sampling, 1 to 2 mm, mean that the CAD parts
touch or overlap there too. --cad needs scipy and trimesh.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent


def compile_without_visuals(path: Path) -> mujoco.MjModel:
    """The model with its visual meshes removed (drawn only: no mass, no contacts), which leaves the dynamics and
    collisions unchanged and loads in a fraction of the memory."""
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


def _joint_equalities(model: mujoco.MjModel) -> list:
    # int() on both sides: `numpy integer == enum` works, but `numpy integer in (enum, ...)` is always False
    return [e for e in range(model.neq) if int(model.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT)]


def couple(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Sets the gripper linkage joints from their equality constraints (a few passes settle chains of them)."""
    for _ in range(4):
        for e in _joint_equalities(model):
            first, second = model.jnt_qposadr[model.eq_obj1id[e]], model.jnt_qposadr[model.eq_obj2id[e]]
            c = model.eq_data[e]
            qpos[first] = c[0] + c[1] * qpos[second] + c[2] * qpos[second] ** 2
    return qpos


def random_pose(model: mujoco.MjModel, rng: np.random.Generator) -> np.ndarray:
    """qpos with the independent joints drawn uniformly and the coupled gripper joints following their equalities."""
    qpos = model.qpos0.copy()
    coupled = {int(model.eq_obj1id[e]) for e in _joint_equalities(model)}
    movable = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    for joint in range(model.njnt):
        if joint in coupled or int(model.jnt_type[joint]) not in movable:
            continue
        low, high = model.jnt_range[joint] if model.jnt_limited[joint] else (-np.pi, np.pi)
        qpos[model.jnt_qposadr[joint]] = rng.uniform(low, high)
    return couple(model, qpos)


def fixed_pose(model: mujoco.MjModel, changes: dict) -> np.ndarray:
    """'home' with the named joints set to the given values (the gripper linkage follows its input)."""
    qpos = model.key("home").qpos.copy()
    for joint, value in changes.items():
        qpos[model.joint(joint).qposadr[0]] = value
    return couple(model, qpos)


def touching(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> dict:
    """{(body, body): deepest penetration (m, negative)} of the pairs that touch at qpos."""
    data.qpos[:] = qpos
    mujoco.mj_fwdPosition(model, data)
    pairs = {}
    for contact in data.contact[:data.ncon]:
        bodies = tuple(sorted(model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)))
        pairs[bodies] = min(pairs.get(bodies, 0.0), float(contact.dist))
    return pairs


def sweep(model: mujoco.MjModel, poses: int, seed: int) -> dict:
    """{(body, body): [(penetration, qpos), ...]} over the random poses, deepest first."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    pairs = defaultdict(list)
    for _ in range(poses):
        qpos = random_pose(model, rng)
        for bodies, depth in touching(model, data, qpos).items():
            pairs[bodies].append((depth, qpos))
    return {bodies: sorted(hits, key=lambda hit: hit[0]) for bodies, hits in pairs.items()}


class CadSurfaces:
    """The links' CAD surfaces (visual meshes) as dense point clouds in each body's frame: every vertex plus about
    `density` random points per m^2 of surface (seeded), loaded once per body."""

    def __init__(self, model: mujoco.MjModel, path: Path, density: float = 1e6) -> None:
        self.model, self.density, self.cache, self.trees = model, density, {}, {}
        self.spec = mujoco.MjSpec.from_file(str(path))
        self.files = {mesh.name: path.parent / mesh.file for mesh in self.spec.meshes}

    def points(self, body: str) -> np.ndarray:
        if body not in self.cache:
            import trimesh
            clouds = []
            for geom in self.spec.geoms:
                if geom.classname.name == "visual" and geom.parent.name == body:
                    mesh = trimesh.load(self.files[geom.meshname], force="mesh", process=False)
                    samples, _ = trimesh.sample.sample_surface(mesh, int(mesh.area * self.density), seed=0)
                    rotation = np.zeros(9)
                    mujoco.mju_quat2Mat(rotation, np.asarray(geom.quat, dtype=float))
                    clouds.append(np.concatenate([samples, mesh.vertices]) @ rotation.reshape(3, 3).T
                                  + np.asarray(geom.pos))
            self.cache[body] = np.concatenate(clouds) if clouds else np.zeros((0, 3))
        return self.cache[body]

    def tree(self, body: str):
        if body not in self.trees:
            from scipy.spatial import cKDTree
            self.trees[body] = cKDTree(self.points(body))
        return self.trees[body]

    def gap(self, bodies: tuple, qpos: np.ndarray, reach: float = 0.05) -> float:
        """Smallest distance (m) between the two bodies' CAD surfaces at qpos; inf if it exceeds reach or a body has
        no visual mesh. The smaller cloud is moved into the larger body's frame and looked up in its tree."""
        data = mujoco.MjData(self.model)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, data)
        big, small = sorted(bodies, key=lambda body: -len(self.points(body)))
        if not len(self.points(small)):
            return float("inf")
        frame, other = data.body(big), data.body(small)
        moved = (self.points(small) @ other.xmat.reshape(3, 3).T + other.xpos - frame.xpos) @ frame.xmat.reshape(3, 3)
        return float(self.tree(big).query(moved, k=1, distance_upper_bound=reach, workers=-1)[0].min())


def _mm(value: float) -> str:
    return f"{value * 1000:7.1f}mm" if np.isfinite(value) else "  >50 mm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=HERE / "robot.xml")
    parser.add_argument("--poses", type=int, default=3000, help="random poses")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--at", nargs="+", metavar="JOINT=VALUE", help="one pose: 'home' with these joints changed")
    parser.add_argument("--cad", action="store_true", help="measure the CAD gaps (needs scipy and trimesh)")
    parser.add_argument("--cad-poses", type=int, default=1,
                        help="random poses: measure each pair's gap at up to this many of its poses, deepest first")
    parser.add_argument("--pair", nargs=2, metavar="BODY", help="report only this pair (with --at: even if apart)")
    args = parser.parse_args()
    model = compile_without_visuals(args.model)
    cad = CadSurfaces(model, args.model) if args.cad else None
    wanted = tuple(sorted(args.pair)) if args.pair else None

    if args.at:
        changes = {item.split("=")[0]: float(item.split("=")[1]) for item in args.at}
        qpos = fixed_pose(model, changes)
        pairs = touching(model, mujoco.MjData(model), qpos)
        if wanted:
            pairs = {wanted: pairs.get(wanted, 0.0)}
        print(f"{args.model.name} at 'home' with {changes}: {len(pairs)} pair(s)")
        for bodies, depth in sorted(pairs.items()):
            line = f"{bodies[0] + ' / ' + bodies[1]:58s} " + (f"overlap {-depth * 1000:5.1f} mm" if depth < 0
                                                                else "apart (collision shapes)")
            print(line + (f"   CAD gap {_mm(cad.gap(bodies, qpos))}" if cad else ""))
        return

    pairs = sweep(model, args.poses, args.seed)
    if wanted:
        pairs = {wanted: pairs.get(wanted, [])}
    print(f"{args.model.name}: {args.poses} random poses, seed {args.seed}; {len(pairs)} body pairs touch")
    heading = f"{'pair':58s} {'poses':>6s} {'deepest':>9s}"
    if cad and args.cad_poses == 1:
        heading += "   CAD gap"
    elif cad:
        heading += "   CAD gap: deepest, largest; poses measured / apart >1 mm, >5 mm"
    print(heading)
    for bodies, hits in sorted(pairs.items(), key=lambda item: -len(item[1])):
        line = f"{bodies[0] + ' / ' + bodies[1]:58s} {len(hits):6d} " + (_mm(hits[0][0]) if hits else "      -")
        if cad and hits:
            gaps = np.array([cad.gap(bodies, qpos) for _, qpos in hits[:args.cad_poses]])
            if args.cad_poses == 1:
                line += f" {_mm(gaps[0])}"
            else:
                apart = np.where(np.isfinite(gaps), gaps, 0.05)
                line += (f" {_mm(gaps[0])}, {_mm(apart.max())}; {len(gaps)} / {int((apart > 0.001).sum())},"
                         f" {int((apart > 0.005).sum())}")
        print(line)


if __name__ == "__main__":
    main()
