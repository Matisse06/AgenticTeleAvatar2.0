#!/usr/bin/env python3
"""Convert the TeleAvatar URDF to a fixed-base MuJoCo dual-arm model."""
from __future__ import annotations

import argparse
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE / "urdf/urdf20260625/urdf20260625.urdf"
DEFAULT_OUTPUT = HERE / "robot.xml"
JOINTS = [*(f"l_joint{i}" for i in range(1, 8)), *(f"r_joint{i}" for i in range(1, 8))]
HOME = {
    "l_joint1": 0.41, "l_joint2": 1.16, "l_joint3": -0.47, "l_joint4": 0.90,
    "l_joint5": 0.23, "l_joint6": -0.15, "l_joint7": 0.60,
    "r_joint1": -0.50, "r_joint2": -0.92, "r_joint3": 0.52, "r_joint4": -1.28,
    "r_joint5": 0.32, "r_joint6": 0.55, "r_joint7": -0.52,
}
EXCLUDES = (("base_link", "Link-L1"), ("base_link", "Link-R1"),
            ("Link-L5", "Link-L7"), ("Link-R5", "Link-R7"))
DAMPING = "0.5"       # nominal simulation value, not hardware calibration
ARMATURE = "0.02"     # nominal simulation value, not hardware calibration
FORCERANGE = "-150 150"  # nominal simulation value, not hardware calibration


def vec(value: str) -> str:
    return " ".join(value.split())


def rpy_to_quat(rpy: str) -> str:
    """Convert URDF fixed-axis RPY to a normalized MJCF wxyz quaternion.

    URDF defines the rotation as Rz(yaw) @ Ry(pitch) @ Rx(roll).  MJCF's
    ``euler`` convention depends on compiler settings, so URDF angles must not
    be copied there directly.
    """
    values = tuple(float(value) for value in rpy.split())
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid URDF rpy: {rpy!r}")
    roll, pitch, yaw = values
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    quat = (cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr)
    norm = math.sqrt(sum(value * value for value in quat))
    quat = tuple(value / norm for value in quat)
    return " ".join(f"{value:.17g}" for value in quat)


def origin(parent: ET.Element) -> tuple[str, str]:
    node = parent.find("origin")
    xyz, rpy = (("0 0 0", "0 0 0") if node is None else (
        vec(node.get("xyz", "0 0 0")), vec(node.get("rpy", "0 0 0"))))
    return xyz, rpy_to_quat(rpy)


def parse_and_validate(urdf_path: Path, verbose: bool = False) -> tuple[ET.Element, dict[str, ET.Element], list[ET.Element], Path]:
    if not urdf_path.is_file():
        raise ValueError(f"URDF does not exist: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    links = {element.get("name"): element for element in root.findall("link")}
    joints = root.findall("joint")
    by_name = {joint.get("name"): joint for joint in joints}
    missing = [name for name in JOINTS if name not in by_name]
    if missing:
        raise ValueError(f"missing arm joints: {missing}")
    if sum(joint.get("type") == "revolute" for joint in joints) != 14:
        raise ValueError("URDF must contain exactly 14 revolute joints")
    children = [joint.find("child").get("link") for joint in joints]
    roots = set(links) - set(children)
    if roots != {"shoulder_base"} or len(children) != len(set(children)):
        raise ValueError("URDF topology must be one tree rooted at shoulder_base")
    adjacency: dict[str, list[str]] = {}
    for joint in joints:
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if parent not in links or child not in links:
            raise ValueError(f"joint {joint.get('name')} references an unknown link")
        adjacency.setdefault(parent, []).append(child)
    reached = set()
    def visit(link_name: str) -> None:
        if link_name in reached:
            raise ValueError(f"cycle detected at {link_name}")
        reached.add(link_name)
        for child in adjacency.get(link_name, []):
            visit(child)
    visit("shoulder_base")
    if reached != set(links):
        raise ValueError(f"disconnected links: {sorted(set(links) - reached)}")
    for name in JOINTS:
        joint = by_name[name]
        limit = joint.find("limit")
        low, high = float(limit.get("lower")), float(limit.get("upper"))
        if not low <= HOME[name] <= high:
            raise ValueError(f"home target {HOME[name]} outside {name} range [{low}, {high}]")
    mesh_dir = urdf_path.parent / "meshes"
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if not filename.startswith("package://urdf20260625/meshes/"):
            raise ValueError(f"unsupported mesh path: {filename}")
        if not (mesh_dir / Path(filename).name).is_file():
            raise ValueError(f"missing mesh: {filename}")
    if verbose:
        print(f"validated {urdf_path}: {len(links)} links, {len(joints)} joints, 14-DOF tree")
        print(f"validated {len(root.findall('.//mesh'))} mesh references and all home ranges")
    return root, links, joints, mesh_dir


def add_inertial(link: ET.Element, body: ET.Element) -> None:
    data = link.find("inertial")
    if data is None:
        return
    mass, inertia = data.find("mass"), data.find("inertia")
    if mass is None or inertia is None:
        return
    xyz, quat = origin(data)
    values = " ".join(inertia.get(k, "0") for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
    attrs = {"pos": xyz, "mass": mass.get("value"), "fullinertia": values}
    origin_node = data.find("origin")
    if origin_node is not None and any(
            abs(float(value)) > 1e-15 for value in origin_node.get("rpy", "0 0 0").split()):
        # MuJoCo forbids fullinertia with an inertial orientation. Preserve the
        # URDF frame by rotating the full inertia tensor into the link frame.
        ixx, iyy, izz, ixy, ixz, iyz = (float(inertia.get(k, "0")) for k in
                                        ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
        source = ((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz))
        w, x, y, z = (float(value) for value in quat.split())
        rotation = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
        rotated = tuple(tuple(sum(rotation[i][k] * source[k][l] * rotation[j][l]
                                  for k in range(3) for l in range(3))
                              for j in range(3)) for i in range(3))
        attrs["fullinertia"] = " ".join(f"{value:.17g}" for value in
            (rotated[0][0], rotated[1][1], rotated[2][2],
             rotated[0][1], rotated[0][2], rotated[1][2]))
    ET.SubElement(body, "inertial", attrs)


def add_geoms(link: ET.Element, body: ET.Element) -> None:
    for tag, contact in (("visual", False), ("collision", True)):
        for index, item in enumerate(link.findall(tag)):
            mesh = item.find("geometry/mesh")
            if mesh is None:
                continue
            xyz, quat = origin(item)
            attrs = {"name": f"{link.get('name')}_{tag}_{index}", "type": "mesh",
                     "mesh": Path(mesh.get("filename")).stem, "pos": xyz, "quat": quat,
                     "contype": "1" if contact else "0", "conaffinity": "1" if contact else "0"}
            if not contact:
                color = item.find("material/color")
                attrs["rgba"] = vec(color.get("rgba")) if color is not None else "0.75 0.75 0.75 1"
                attrs["group"] = "1"
            ET.SubElement(body, "geom", attrs)


def build(urdf_path: Path = DEFAULT_URDF, output: Path = DEFAULT_OUTPUT, verbose: bool = False) -> None:
    urdf_path, output = urdf_path.resolve(), output.resolve()
    root, links, joints, mesh_dir = parse_and_validate(urdf_path, verbose)
    by_name = {joint.get("name"): joint for joint in joints}
    children: dict[str, list[ET.Element]] = {}
    child_names = set()
    for joint in joints:
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
        child_names.add(joint.find("child").get("link"))
    root_link = next(name for name in links if name not in child_names)

    model = ET.Element("mujoco", {"model": "teleavatar_2_sim"})
    model.append(ET.Comment("All damping, armature, force, gain, solver and contact parameters are nominal simulation values; not physical-robot calibration."))
    ET.SubElement(model, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(model, "option", {"timestep": "0.002", "iterations": "50",
                                     "integrator": "implicitfast", "gravity": "0 0 -9.81"})
    asset = ET.SubElement(model, "asset")
    for path in sorted(mesh_dir.glob("*.STL")):
        relative = Path(os.path.relpath(path, output.parent)).as_posix()
        ET.SubElement(asset, "mesh", {"name": path.stem, "file": relative})
    worldbody = ET.SubElement(model, "worldbody")

    def add_body(parent: ET.Element, link_name: str, incoming: ET.Element | None = None) -> None:
        body = ET.SubElement(parent, "body", {"name": link_name})
        if incoming is not None:
            xyz, quat = origin(incoming)
            body.set("pos", xyz); body.set("quat", quat)
            if incoming.get("type") == "revolute":
                limit = incoming.find("limit")
                ET.SubElement(body, "joint", {"name": incoming.get("name"), "type": "hinge",
                    "axis": vec(incoming.find("axis").get("xyz", "0 0 1")), "limited": "true",
                    "range": f"{limit.get('lower')} {limit.get('upper')}",
                    "damping": DAMPING, "armature": ARMATURE})
        add_inertial(links[link_name], body)
        add_geoms(links[link_name], body)
        for child_joint in children.get(link_name, []):
            add_body(body, child_joint.find("child").get("link"), child_joint)

    add_body(worldbody, root_link)
    actuator = ET.SubElement(model, "actuator")
    for name in JOINTS:
        limit = by_name[name].find("limit")
        joint_range = f"{limit.get('lower')} {limit.get('upper')}"
        ET.SubElement(actuator, "position", {"name": f"{name}_actuator", "joint": name,
            "kp": "120", "kv": "8", "ctrllimited": "true", "ctrlrange": joint_range,
            "forcelimited": "true", "forcerange": FORCERANGE})
    contact = ET.SubElement(model, "contact")
    for first, second in EXCLUDES:
        ET.SubElement(contact, "exclude", {"body1": first, "body2": second})
    keyframe = ET.SubElement(model, "keyframe")
    values = " ".join(str(HOME[name]) for name in JOINTS)
    ET.SubElement(keyframe, "key", {"name": "home", "qpos": values, "ctrl": values})
    ET.indent(model, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(ET.tostring(model, encoding="unicode") + "\n", encoding="utf-8")
    if verbose:
        print(f"wrote {output} with 14 position actuators and 4 contact exclusions")


def compile_check(urdf: Path, verbose: bool) -> None:
    import mujoco
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        output = Path(directory) / "robot.xml"
        build(urdf, output, verbose)
        model = mujoco.MjModel.from_xml_path(str(output))
        if (model.nq, model.nv, model.nu) != (14, 14, 14):
            raise ValueError(f"compiled dimensions are nq/nv/nu={model.nq}/{model.nv}/{model.nu}")
        if verbose:
            print("MuJoCo compile check passed: nq=14, nv=14, nu=14")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true", help="validate and compile without writing robot.xml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        compile_check(args.urdf, args.verbose)
    else:
        build(args.urdf, args.output, args.verbose)
        if not args.verbose:
            print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
