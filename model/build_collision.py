#!/usr/bin/env python3
"""Split each collision mesh of the URDF into convex pieces (CoACD), for MuJoCo contacts.

MuJoCo collides every mesh as its convex hull. The vendor's collision STLs are the raw CAD surfaces (4.6M triangles,
mostly not closed), so their hulls fill concave regions: the gripper base's hull is 4x its volume and covers the gap
between the fingers. This writes model/assets/collision/<mesh>_<k>.obj (the pieces' hulls, a few hundred kB per link)
and index.json; convert.py uses them and falls back to the raw STL for any mesh without pieces.

Needs coacd, trimesh and fast-simplification (pip). One-time; about a minute on many cores:
  python3 model/build_collision.py            # all meshes
  python3 model/build_collision.py lg-link3   # selected meshes (file stems)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE / "vendor" / "urdf" / "teleavatar_urdf_20260928-v2.urdf"  # meshes as in 20260928
OUT = HERE / "assets" / "collision"
PARAMS = {"target_faces": 40000, "threshold": 0.05, "max_convex_hull": 16, "max_ch_vertex": 128,
          "resolution": 2000, "seed": 0,
          # Large concave parts get more pieces: the chassis with its lift column, and the gripper bases (palms).
          "max_convex_hull_for": {"base_link": 32, "lg-base_link": 32, "rg-base_link": 32},
          # Keep a single hull when the pieces fill at least this share of it: splitting gains nothing there.
          "single_hull_above": 0.85}


def collision_meshes(urdf: Path) -> dict[str, Path]:
    meshes = {}
    for mesh in ET.parse(urdf).getroot().findall("link/collision/geometry/mesh"):
        path = (urdf.parent / mesh.get("filename")).resolve()
        meshes[path.stem] = path
    return meshes


def mirror(stem: str) -> str | None:
    """The other side's mesh (armL3_link <-> armR3_link, lg-link2 <-> rg-link2), or None."""
    for left, right in (("armL", "armR"), ("lg-", "rg-")):
        if left in stem:
            return stem.replace(left, right)
        if right in stem:
            return stem.replace(right, left)
    return None


def decompose(job: tuple[str, str]) -> dict:
    import coacd
    import numpy as np
    import trimesh
    coacd.set_log_level("error")
    stem, source = job
    start = time.time()
    mesh = trimesh.load(source, force="mesh")
    faces_in = len(mesh.faces)
    full_hull = mesh.convex_hull
    if faces_in > PARAMS["target_faces"]:
        mesh = mesh.simplify_quadric_decimation(face_count=PARAMS["target_faces"])
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=PARAMS["threshold"],
                            max_convex_hull=PARAMS["max_convex_hull_for"].get(stem, PARAMS["max_convex_hull"]),
                            max_ch_vertex=PARAMS["max_ch_vertex"], preprocess_mode="auto",
                            resolution=PARAMS["resolution"], seed=PARAMS["seed"])
    split = [trimesh.Trimesh(np.asarray(vertices), np.asarray(faces)).convex_hull for vertices, faces in parts]
    return {"mesh": stem, "faces_in": faces_in, "split": [(hull.vertices, hull.faces) for hull in split],
            "hull": (full_hull.vertices, full_hull.faces),
            "split_volume_cm3": round(sum(hull.volume for hull in split) * 1e6, 2),
            "hull_volume_cm3": round(full_hull.volume * 1e6, 2), "seconds": round(time.time() - start, 1)}  # printed only


def write_pieces(stem: str, hulls: list) -> None:
    for old in OUT.glob(f"{stem}_*.obj"):
        old.unlink()
    for index, (vertices, faces) in enumerate(hulls):
        lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
        (OUT / f"{stem}_{index}.obj").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("meshes", nargs="*", help="mesh file stems to redo, with their mirror image (default: all)")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--jobs", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    meshes = collision_meshes(args.urdf.resolve())
    selected = set(args.meshes or meshes)
    unknown = sorted(selected - set(meshes))
    if unknown:
        parser.error(f"not collision meshes of {args.urdf.name}: {unknown}")
    selected |= {mirror(stem) for stem in selected if mirror(stem) in meshes}  # both sides decide together
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {"params": PARAMS, "meshes": {}}
    if index["params"] != PARAMS:
        index = {"params": PARAMS, "meshes": {}}
    results = {}
    jobs = [(stem, str(meshes[stem])) for stem in sorted(selected)]
    with multiprocessing.Pool(min(args.jobs, len(jobs))) as pool:
        for result in pool.imap_unordered(decompose, jobs):
            results[result["mesh"]] = result
            print(f"{len(results):3d}/{len(jobs)} {result['mesh']}: {len(result['split'])} pieces, "
                  f"{result['split_volume_cm3'] / result['hull_volume_cm3']:.2f} of the hull, {result['seconds']} s",
                  flush=True)
    for stem, result in sorted(results.items()):
        # A single hull when the pieces fill most of it anyway; left and right parts decide together, so the two
        # arms and grippers get the same kind of collision geometry.
        ratios = [result["split_volume_cm3"] / result["hull_volume_cm3"]]
        if mirror(stem) in results:
            other = results[mirror(stem)]
            ratios.append(other["split_volume_cm3"] / other["hull_volume_cm3"])
        hulls = [result["hull"]] if sum(ratios) / len(ratios) >= PARAMS["single_hull_above"] else result["split"]
        write_pieces(stem, hulls)
        volume = result["hull_volume_cm3"] if len(hulls) == 1 and hulls[0] is result["hull"] else result["split_volume_cm3"]
        index["meshes"][stem] = {"pieces": len(hulls), "faces_in": result["faces_in"], "pieces_volume_cm3": volume,
                                 "split_volume_cm3": result["split_volume_cm3"],
                                 "hull_volume_cm3": result["hull_volume_cm3"]}
    index["meshes"] = dict(sorted(index["meshes"].items()))
    index_path.write_text(json.dumps(index, indent=1) + "\n")
    print(f"wrote {sum(m['pieces'] for m in index['meshes'].values())} pieces for {len(index['meshes'])} meshes to {OUT}")


if __name__ == "__main__":
    main()
