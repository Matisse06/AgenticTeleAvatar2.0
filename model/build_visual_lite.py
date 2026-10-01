#!/usr/bin/env python3
"""Optional lighter visual meshes, for CPU rendering and many parallel simulations: convert.py --visual-faces N.

The vendor's visual meshes have 3.5M triangles in total. They render in about 2 ms per frame on a GPU, but take about
1.2 s per 960x720 frame on 4 CPU threads (Mesa llvmpipe), and their bounding-volume hierarchies are most of the
compiled model's 830 MB. This decimates every visual mesh above N faces to N, keeping its texture coordinates
(MeshLab's texture-aware quadric edge collapse), into model/assets/visual_<N>/; smaller meshes are used as delivered.
Shapes and textures are unchanged, but shading on smooth surfaces changes slightly, so never mix renders made with and
without it in one dataset (train and evaluate with the same setting).

Needs pymeshlab (pip). About a minute:
  python3 model/build_visual_lite.py 50000 && python3 model/convert.py --visual-faces 50000
"""
import argparse
import multiprocessing
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE / "vendor" / "urdf" / "teleavatar_urdf_20260928.urdf"


def visual_meshes(urdf: Path) -> list[Path]:
    root = ET.parse(urdf).getroot()
    return sorted({(urdf.parent / mesh.get("filename")).resolve() for mesh in root.findall("link/visual/geometry/mesh")})


def decimate(job: tuple[str, int, str]) -> str:
    import pymeshlab
    source, faces, out = job
    meshes = pymeshlab.MeshSet()
    meshes.load_new_mesh(source)
    before = meshes.current_mesh().face_number()
    target = Path(out) / Path(source).name
    if before <= faces:
        target.unlink(missing_ok=True)  # convert.py uses the vendor file
        return f"{Path(source).name}: {before} faces, kept"
    meshes.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=faces, qualitythr=0.3, extratcoordw=1.0, preserveboundary=True, optimalplacement=True,
        preservenormal=True)
    meshes.save_current_mesh(str(target), save_wedge_texcoord=True, save_textures=False, save_vertex_normal=False)
    Path(f"{target}.mtl").unlink(missing_ok=True)  # MeshLab's placeholder material; MuJoCo ignores it
    return f"{Path(source).name}: {before} -> {meshes.current_mesh().face_number()} faces"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("faces", type=int, help="maximum faces per visual mesh, e.g. 50000")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    args = parser.parse_args()
    out = HERE / "assets" / f"visual_{args.faces}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(path), args.faces, str(out)) for path in visual_meshes(args.urdf.resolve())]
    with multiprocessing.Pool(min(len(jobs), multiprocessing.cpu_count())) as pool:
        for line in pool.imap_unordered(decimate, jobs):
            print(line, flush=True)
    print(f"wrote {out}; now run: python3 model/convert.py --visual-faces {args.faces}")


if __name__ == "__main__":
    main()
