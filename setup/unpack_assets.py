#!/usr/bin/env python3
"""Unpack the vendor's textured model archive into model/vendor/ and verify every file against model/vendor/SHA256SUMS.

The archive's meshes and textures (242 files, 590 MB) are too big for git, so git tracks only their checksums. Any of
the three published archives works (urdf_20260825_textured9.14 / 9.17 / 9.21.7z hold byte-identical meshes); only
visual/ and collision/ are unpacked, because the URDFs and documents in model/vendor/ are tracked in git.

  python3 setup/unpack_assets.py urdf_20260825_textured9.21.7z      # needs py7zr (pip install py7zr)

Download the archive from the robot's documentation page: https://www.dexteleop.com/docs/teleavatar-2/resources/urdf

On the cluster: ./sim.sh uv pip install py7zr, then ./sim.sh python3 setup/unpack_assets.py <archive>. To keep the
590 MB off the backed-up home directory, first make model/vendor/visual and model/vendor/collision symlinks to folders
under $HL: the files then go there.
"""
import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "model" / "vendor"
FOLDERS = ("visual", "collision")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> list[str]:
    """Return the problems found (missing, changed or unexpected files); empty means verified."""
    expected = {}
    for line in (VENDOR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest
    problems = [f"missing {name}" for name in expected if not (root / name).is_file()]
    problems += [f"checksum mismatch {name}" for name, digest in expected.items()
                 if (root / name).is_file() and sha256(root / name) != digest]
    present = {path.relative_to(root).as_posix() for folder in FOLDERS if (root / folder).is_dir()
               for path in (root / folder).rglob("*") if path.is_file()}
    problems += [f"unexpected {name}" for name in sorted(present - set(expected))]
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, nargs="?", help="urdf_20260825_textured*.7z")
    parser.add_argument("--verify-only", action="store_true", help="only check the files already in model/vendor/")
    args = parser.parse_args()

    if not args.verify_only:
        if args.archive is None:
            parser.error("give the archive path, or --verify-only")
        import py7zr  # imported here so --verify-only works without it
        visual = VENDOR / "visual"  # unpack next to a symlinked target, so the 590 MB never passes through home
        with tempfile.TemporaryDirectory(dir=visual.resolve().parent if visual.is_symlink() else VENDOR) as scratch:
            with py7zr.SevenZipFile(args.archive) as archive:
                archive.extractall(path=scratch)
            top = Path(scratch)
            entries = list(top.iterdir())
            if len(entries) == 1 and entries[0].is_dir():  # 9.14 wraps everything in urdf_20260825_textured/
                top = entries[0]
            for folder in FOLDERS:
                if not (top / folder).is_dir():
                    print(f"{args.archive} has no {folder}/ folder", file=sys.stderr)
                    return 1
                destination = VENDOR / folder
                if destination.is_symlink():  # e.g. to $HL on the cluster: fill the folder it points to
                    destination = destination.resolve()
                    destination.mkdir(parents=True, exist_ok=True)
                    for old in destination.iterdir():
                        shutil.rmtree(old) if old.is_dir() and not old.is_symlink() else old.unlink()
                    for item in (top / folder).iterdir():
                        shutil.move(str(item), destination / item.name)
                else:
                    shutil.rmtree(destination, ignore_errors=True)
                    shutil.move(str(top / folder), destination)
        for folder in FOLDERS:  # the archive stores world-writable permissions; use 755 / 644
            root = (VENDOR / folder).resolve()
            for path in [root, *root.rglob("*")]:
                path.chmod(0o755 if path.is_dir() else 0o644)
        print(f"unpacked {', '.join(f + '/' for f in FOLDERS)} from {args.archive} into {VENDOR}")

    problems = verify(VENDOR)
    if problems:
        print(f"{len(problems)} problems:", *problems[:20], sep="\n  ", file=sys.stderr)
        return 1
    print(f"verified {len((VENDOR / 'SHA256SUMS').read_text().splitlines())} files against {VENDOR / 'SHA256SUMS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
