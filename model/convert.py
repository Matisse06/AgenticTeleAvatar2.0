#!/usr/bin/env python3
"""Build the MuJoCo model of TeleAvatar 2.0 from the vendor's 20260928 URDF and textured meshes, in two variants:
model/robot.xml, the default, with the base fixed to the floor as in the vendor's simulator, and
model/robot_mobile.xml, with the drivable base we added (BASE_JOINTS below), loaded by model/scene_mobile.xml.

  python3 model/convert.py                       # writes both, scene_mobile.xml, and the resized textures they use
  python3 model/convert.py --texture-size 2048   # full-resolution textures (about 4x the memory)
  python3 model/convert.py --check-only          # build both in temporary files and compile them

Needs mujoco, numpy and pillow, the vendor meshes (setup/unpack_assets.py) and the convex collision pieces in
model/assets/collision/ (tracked in git; model/build_collision.py regenerates them). Values that do not come from the
vendor's files are marked NOMINAL or ASSUMED here and in CLAUDE.md.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

from gripper import CLOSE_TORQUE, OPEN_TORQUE, torque as gripper_torque

HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE / "vendor" / "urdf" / "teleavatar_urdf_20260928.urdf"
DEFAULT_OUTPUT = HERE / "robot.xml"
MOBILE_OUTPUT = HERE / "robot_mobile.xml"
SCENE, MOBILE_SCENE = HERE / "scene.xml", HERE / "scene_mobile.xml"
COLLISION_DIR = HERE / "assets" / "collision"
TEXTURE_DIR = HERE / "assets"  # textures_<size>/ is generated next to collision/

# The ROS API (vendor simulator and real robot) names the arm joints l_joint1..7 / r_joint1..7. Their angles mean the
# same as armL1..7_joint / armR1..7_joint in the 20260928 URDF: same zero pose, same axis directions (checked against
# mujoco/urdf/urdf20260625 on 2026-10-01). The 20260922 URDF and the archive's own URDF have seven of them flipped.
API_TO_MODEL = {f"{side.lower()}_joint{i}": f"arm{side}{i}_joint" for side in "LR" for i in range(1, 8)}
ARM_JOINTS = list(API_TO_MODEL.values())
# Vendor home pose (mujoco/convert_urdf.py HOME, in API names).
API_HOME = {
    "l_joint1": 0.41, "l_joint2": 1.16, "l_joint3": -0.47, "l_joint4": 0.90,
    "l_joint5": 0.23, "l_joint6": -0.15, "l_joint7": 0.60,
    "r_joint1": -0.50, "r_joint2": -0.92, "r_joint3": 0.52, "r_joint4": -1.28,
    "r_joint5": 0.32, "r_joint6": 0.55, "r_joint7": -0.52,
}
LIFT = "lift_carriage_joint"
# NOMINAL: 0.136 m down from the top puts the shoulder joints 1.27 m above the floor, as in the old model.
LIFT_HOME = 0.136
GRIPPERS = {"left": "lg", "right": "rg"}
# manifest.json: "gripper_input_range_radians": [0, 1] = lg_joint1 / rg_joint1, the joint the damping document lists.
# ASSUMED: the URDF leaves the input joint and the finger linkage (lg_joint2, which 7 joints mimic) independent; their
# ranges, [0, 1] and [-0.98, 0], and the CAD pose (both 0) suggest finger angle = -0.98 * input. With it, input 0 is
# closed (fingertips touch) and 1 is fully open, the parallel fingers moving symmetrically. Check on the real gripper.
GRIPPER_COUPLING = -0.98
GRIPPER_HOME = 1.0  # open
# Drive (developer docs §4.4.1): the gripper motor takes a feedforward torque, which the robot's 0..1 command sets
# through the curve in gripper.py; the actuators take that torque in N m (-1.6 closing .. +2.0 opening). ASSUMED:
# lg_joint1 / rg_joint1 is the motor's output angle (the docs' q, §4.4.2: 0 closed, opening positive), so the torque
# acts on it directly. The home command is 0 (open), as in the vendor's deployment.
GRIPPER_HOME_COMMAND = 0.0
# Gripper linkage, per side (lg / rg): base, input crank (link1), finger A = links 2, 3, 4, 9 (moved by joint 2), finger
# B = links 5, 6, 7, 10, and link8, the wrist camera. A finger's parallelogram bars overlap each other and the crank sits
# inside the housing, so within a gripper only finger A against finger B can touch (and the camera, a fixed part).
FINGERS = ((2, 3, 4, 9), (5, 6, 7, 10))
# Floor contact: the wheel collision meshes are 179.4 mm across, so the axles (base_link origin) are 89.7 mm up.
BASE_HEIGHT = 0.0897
WHEEL_RADIUS = BASE_HEIGHT
# By default (robot.xml) the base is fixed to the world and the three omni wheels (the URDF's continuous joints) are
# welded, as in the vendor's simulator. ASSUMED, in robot_mobile.xml only: the base drives kinematically. World-frame
# joints carry base_link in the floor plane (slides base_x and base_y, hinge base_yaw about base_link's z), so it
# neither tips nor slips; the wheels roll with it above the floor, at the speeds model/mobile_base.py gives them. The
# actuators take velocity commands (world frame, m/s and rad/s) and hold the reached pose while the command is 0.
BASE_JOINTS = (("base_x", "slide", "1 0 0"), ("base_y", "slide", "0 1 0"), ("base_yaw", "hinge", "0 0 1"))
WHEELS = ("wheel1_wheel_joint", "wheel2_wheel_joint", "wheel3_wheel_joint")
BASE_SPEED, BASE_TURN_RATE = 1.0, 1.5  # NOMINAL command limits (the wheel motors would allow 1.7 m/s, 4.7 rad/s)
# The wheel motors' 20 N m (URDF effort) at the wheel radius push 223 N each along their rolling direction: for this
# wheel layout at most 360 N along any base axis and 209 N m about z, used as the actuators' force limits.
BASE_FORCE, BASE_TORQUE = 360.0, 209.0
BASE_TRAVEL = 4.5  # m from the start, either way: keeps the robot on scene.xml's 10 x 10 m floor
# NOMINAL position hold: critically damped at 10 Hz for the 85.8 kg robot (yaw: its 4.3 kg m^2 at home), so the arms'
# reactions barely move it, as wheel motors holding still would.
BASE_KP, BASE_KV = 340000.0, 10800.0
BASE_YAW_KP, BASE_YAW_KV = 17000.0, 545.0
# Body pairs whose collision shapes overlap by design (found by sweeping the lift and 3000 random poses, 2026-10-01):
EXCLUDES = [
    ("base_link", "lift_carriage_link"),                         # the carriage runs inside the lift column
    ("armL5_link", "lg_base_link"), ("armR5_link", "rg_base_link"),  # wrist link 5 sits inside the gripper housing
    # Elbow: the convex hulls touch from 140 deg of joint 4, the CAD parts only at the 146.5 deg hard stop (FCL).
    ("armL3_link", "armL5_link"), ("armR3_link", "armR5_link"),
]
# Cameras: (body, name, pos, quat, fovy). eye_Link is a ROS optical frame (z forward, x right, y down) and a MuJoCo
# camera looks along -z with y up, hence the 180 deg turn about x. The wrist cameras (lg_link8 / rg_link8: a 71 x 25 x
# 42 mm block, two lenses on its +z face, which looks at the fingertips) sit at that face's centre and are turned 180
# deg about y, so the fingers are at the bottom of the image (ASSUMED orientation). Fields of view: the head's is the
# vendor's spec, 120 x 120 deg per eye (user manual §2.3; a square image gives it, as the robot's 960 x 960 eyes), the
# wrists' are NOMINAL. All are pinholes until the cameras' intrinsics are known (MJCF focalpixel / principalpixel /
# sensorsize then replace fovy, see CLAUDE.md).
CAMERAS = [
    ("eye_Link", "head", (0, 0, 0), (0, 1, 0, 0), 120),
    ("lg_link8", "left_wrist", (0, -0.0008, 0.0414), (0, 0, 1, 0), 58),
    ("rg_link8", "right_wrist", (0, -0.0008, 0.0414), (0, 0, 1, 0), 58),
]

# 各关节阻尼参数.docx ("damping parameters of each joint"): viscous damping (N m s/rad) and Coulomb friction (N m) per
# arm joint 1..7, the same for both arms. Table 2 is measured on the robot (joint 4's measured damping was negative and
# is given as 0); table 1 is the vendor's empirical reference, which also lists the gripper input joint.
DAMPING_TABLES = {
    "measured": [(0.165, 0.562), (0.099, 0.751), (0.082, 0.316), (0.0, 0.379), (0.018, 0.149), (0.026, 0.117),
                 (0.015, 0.121)],
    "empirical": [(1.2, 0.45), (1.5, 0.5), (0.6, 0.24), (0.55, 0.22), (0.18, 0.07), (0.12, 0.05), (0.10, 0.04)],
}
GRIPPER_DYNAMICS = (0.12, 0.08)  # table 1, lg_joint1 / rg_joint1

ARM_ARMATURE = 0.02          # NOMINAL (the vendor model's value; rotor inertias are not published)
ARM_KP, ARM_KV = 120.0, 8.0  # NOMINAL (the vendor model's position gains)
LIFT_KP, LIFT_KV, LIFT_DAMPING, LIFT_ARMATURE = 20000.0, 2000.0, 100.0, 0.1  # NOMINAL
GRIPPER_ARMATURE = 0.001  # NOMINAL (the motor's inertia is not published)
GRIPPER_LINK_ARMATURE = 0.0002  # NOMINAL, on the mimic joints: keeps the 10 g linkage bars numerically stable
# NOMINAL: the gripper's joint stops and linkage constraints use a 5 ms time constant (MuJoCo's default is 20 ms, the
# floor 2 timesteps). With the default, the documented torques (up to 2 N m) on the light linkage stretch them by up
# to 0.035 rad; with 5 ms by at most 0.002 rad, and the gripper stays still at rest.
GRIPPER_SOLREF = "0.005 1"
VIRTUAL_MASS = 1e-5  # links without geometry and below this mass are reference frames: they become sites


def vec(text: str) -> list[float]:
    return [float(value) for value in text.split()]


def fmt(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def rpy_matrix(rpy) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_quat(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix to a unit wxyz quaternion with w >= 0."""
    import mujoco
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation, dtype=float).ravel())
    return quat if quat[0] >= 0 else -quat


def transform(element: ET.Element | None) -> np.ndarray:
    """4x4 transform of a URDF <origin> (identity when absent)."""
    result = np.eye(4)
    node = None if element is None else element.find("origin")
    if node is not None:
        result[:3, :3] = rpy_matrix(vec(node.get("rpy", "0 0 0")))
        result[:3, 3] = vec(node.get("xyz", "0 0 0"))
    return result


def pose_attrs(matrix: np.ndarray) -> dict[str, str]:
    return {"pos": fmt(matrix[:3, 3]), "quat": fmt(matrix_quat(matrix[:3, :3]))}


def lexical(path: Path) -> Path:
    """Absolute path with .. removed but symlinks kept, so robot.xml stays the same when model/vendor/visual or
    collision are symlinks (e.g. to $HL on the cluster)."""
    return Path(os.path.normpath(os.path.abspath(path)))


def link_mass(link: ET.Element) -> float:
    mass = link.find("inertial/mass")
    return 0.0 if mass is None else float(mass.get("value"))


def material_of(obj_path: Path) -> tuple[str, Path]:
    """(material name, colour texture) from the MTL file that an OBJ names in its header."""
    with obj_path.open() as handle:
        for line in handle:
            if line.startswith("mtllib"):
                mtl = obj_path.parent / line.split(maxsplit=1)[1].strip()
                break
            if line.startswith("v "):
                raise ValueError(f"{obj_path} names no MTL file")
    name = texture = None
    for line in mtl.read_text().splitlines():
        if line.startswith("newmtl"):
            name = line.split(maxsplit=1)[1].strip()
        elif line.startswith("map_Kd"):
            texture = mtl.parent / line.split(maxsplit=1)[1].strip()
    if name is None or texture is None:
        raise ValueError(f"{mtl} has no material with a colour texture")
    return name, texture


@functools.lru_cache(maxsize=None)  # both variants use the same materials
def finish(texture: Path) -> dict[str, str]:
    """Material finish from the vendor's PBR maps next to the colour texture.

    metallic and roughness are the maps' means (used by MuJoCo's PBR-capable renderers). The classic OpenGL renderer
    only knows specular and shininess, so those are derived from them: smoother means shinier, metal means more
    specular (a heuristic; the MTL files themselves declare no specular term).
    """
    from PIL import Image, ImageStat
    stem = texture.name[: -len("_color.png")]
    means = []
    for kind in ("roughness", "metalness"):
        # Mean of the full-resolution greyscale map: integer luma and a plain sum, the same with any Pillow version.
        with Image.open(texture.with_name(f"{stem}_{kind}.png")) as image:
            means.append(ImageStat.Stat(image.convert("L")).mean[0] / 255.0)
    roughness, metallic = means
    return {"roughness": f"{roughness:.3f}", "metallic": f"{metallic:.3f}",
            "specular": f"{0.25 + 0.5 * metallic:.3f}", "shininess": f"{0.8 * (1.0 - roughness):.3f}"}


def resized_texture(source: Path, size: int, output_dir: Path) -> Path:
    """RGB copy of a colour texture scaled by size/2048 (the vendor's textures are 2048, the torso's 4096)."""
    if size <= 0:
        return source
    from PIL import Image
    target = output_dir / f"textures_{size}" / source.name
    with Image.open(source) as image:
        width, height = max(1, round(image.width * size / 2048)), max(1, round(image.height * size / 2048))
        if target.is_file():
            with Image.open(target) as existing:
                if existing.size == (width, height):
                    return target
        target.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").resize((width, height), Image.LANCZOS).save(target)
    return target


def build(urdf_path: Path = DEFAULT_URDF, output: Path = DEFAULT_OUTPUT, *, mobile_base: bool = False,
          texture_size: int = 1024, visual_faces: int = 0, gravcomp: bool = True, damping: str = "measured",
          lift_home: float = LIFT_HOME, verbose: bool = False) -> dict:
    """Write one model: the base fixed to the world (default), or drivable (mobile_base, see BASE_JOINTS)."""
    urdf_path, output = lexical(urdf_path), lexical(output)
    root = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    parent_joint = {joint.find("child").get("link"): joint for joint in joints.values()}
    child_joints = defaultdict(list)
    for joint in joints.values():
        child_joints[joint.find("parent").get("link")].append(joint)
    roots = [name for name in links if name not in parent_joint]
    if roots != ["base_link"]:
        raise ValueError(f"expected one tree rooted at base_link, found roots {roots}")
    missing = [name for name in [*ARM_JOINTS, LIFT] if name not in joints]
    if missing:
        raise ValueError(f"URDF lacks joints {missing}")
    home = {API_TO_MODEL[name]: value for name, value in API_HOME.items()}
    for name, value in [*home.items(), (LIFT, lift_home)]:
        limit = joints[name].find("limit")
        if not float(limit.get("lower")) <= value <= float(limit.get("upper")):
            raise ValueError(f"home value {value} outside the range of {name}")

    virtual = {name for name, link in links.items() if link.find("visual") is None
               and link.find("collision") is None and link_mass(link) < VIRTUAL_MASS}
    for name in virtual:
        if parent_joint[name].get("type") != "fixed":
            raise ValueError(f"reference frame {name} must hang on a fixed joint")
    pieces = json.loads((COLLISION_DIR / "index.json").read_text())["meshes"] \
        if (COLLISION_DIR / "index.json").is_file() else {}
    table = DAMPING_TABLES[damping]
    lite = TEXTURE_DIR / f"visual_{visual_faces}"
    if visual_faces and not lite.is_dir():
        raise FileNotFoundError(f"{lite} is missing; run model/build_visual_lite.py {visual_faces}")
    rel = lambda path: Path(os.path.relpath(path, output.parent)).as_posix()

    model = ET.Element("mujoco", {"model": "teleavatar_20260928"})
    if mobile_base:
        model.append(ET.Comment(
            f" Generated by model/convert.py from {urdf_path.name}; do not edit. Gains, armature, lift and base "
            "settings are NOMINAL simulation values; the grippers take motor torque in N m (developer docs §4.4.1; "
            "model/gripper.py maps the robot's command); joint damping/friction come from the vendor's damping "
            f"document ({damping} table). The base drives kinematically in the floor plane (ASSUMED, see "
            "convert.py). "))
    else:
        model.append(ET.Comment(
            f" Generated by model/convert.py from {urdf_path.name}; do not edit. Gains, armature and lift settings "
            "are NOMINAL simulation values; the grippers take motor torque in N m (developer docs §4.4.1; "
            "model/gripper.py maps the robot's command); joint damping/friction come from the vendor's damping "
            f"document ({damping} table). The base is fixed to the world and the wheels are welded, as in the "
            "vendor's simulator; robot_mobile.xml has a drivable base. "))
    ET.SubElement(model, "compiler", {"angle": "radian", "autolimits": "true"})
    # Elliptic friction cones with impratio > 1 reduce slip in grasps (MuJoCo's recommendation for manipulation).
    ET.SubElement(model, "option", {"timestep": "0.002", "integrator": "implicitfast", "cone": "elliptic",
                                     "impratio": "10"})
    default = ET.SubElement(model, "default")
    visual_class = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(visual_class, "geom", {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2",
                                         "density": "0"})
    collision_class = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(collision_class, "geom", {"type": "mesh", "group": "3"})
    asset = ET.SubElement(model, "asset")
    materials = {}
    worldbody = ET.SubElement(model, "worldbody")
    stats = defaultdict(int)

    def add_link_geoms(link: ET.Element, body: ET.Element) -> None:
        name = link.get("name")
        for index, item in enumerate(link.findall("visual")):
            obj = lexical(urdf_path.parent / item.find("geometry/mesh").get("filename"))
            material, texture = material_of(obj)
            if material not in materials:
                materials[material] = texture
                ET.SubElement(asset, "texture", {"name": material, "type": "2d",
                                                 "file": rel(resized_texture(texture, texture_size, TEXTURE_DIR))})
                ET.SubElement(asset, "material", {"name": material, "texture": material, **finish(texture)})
            mesh_name = f"{name}_visual" + (f"_{index}" if index else "")
            mesh_file = lite / obj.name if visual_faces and (lite / obj.name).is_file() else obj
            ET.SubElement(asset, "mesh", {"name": mesh_name, "file": rel(mesh_file), "inertia": "shell"})
            ET.SubElement(body, "geom", {"name": mesh_name, "class": "visual", "mesh": mesh_name,
                                         "material": material, **pose_attrs(transform(item))})
            stats["visual geoms"] += 1
        for item in link.findall("collision"):
            stl = lexical(urdf_path.parent / item.find("geometry/mesh").get("filename"))
            # The raw STLs cannot be used directly: MuJoCo reads at most 200k faces per STL (the chassis has 472k).
            count = pieces.get(stl.stem, {}).get("pieces", 0)
            files = [COLLISION_DIR / f"{stl.stem}_{k}.obj" for k in range(count)]
            if not files or not all(path.is_file() for path in files):
                raise FileNotFoundError(f"no collision pieces for {stl.name} in {COLLISION_DIR}; "
                                        "run model/build_collision.py")
            for k, path in enumerate(files):
                mesh_name = f"{name}_collision_{k}"
                ET.SubElement(asset, "mesh", {"name": mesh_name, "file": rel(path)})
                ET.SubElement(body, "geom", {"name": mesh_name, "class": "collision", "mesh": mesh_name,
                                             **pose_attrs(transform(item))})
                stats["collision geoms"] += 1

    def add_frames(link_name: str, body: ET.Element, offset: np.ndarray) -> None:
        """Reference frames (massless links on fixed joints) under link_name become sites of body."""
        for joint in child_joints[link_name]:
            child = joint.find("child").get("link")
            if child in virtual:
                pose = offset @ transform(joint)
                ET.SubElement(body, "site", {"name": child, "size": "0.01", "group": "4", **pose_attrs(pose)})
                stats["sites"] += 1
                add_frames(child, body, pose)

    def add_body(parent: ET.Element, link_name: str) -> None:
        link = links[link_name]
        incoming = parent_joint.get(link_name)
        attrs = {"name": link_name}
        if incoming is None:
            attrs["pos"] = fmt([0, 0, BASE_HEIGHT])
        else:
            attrs.update(pose_attrs(transform(incoming)))
        if gravcomp and link_name != "base_link" and not link_name.startswith("wheel"):
            attrs["gravcomp"] = "1"  # the arm, lift and gripper controllers are assumed to compensate gravity
        body = ET.SubElement(parent, "body", attrs)
        kind = None if incoming is None else incoming.get("type")
        if incoming is None and mobile_base:
            for name, joint_type, axis in BASE_JOINTS:
                ET.SubElement(body, "joint", {"name": name, "type": joint_type, "axis": axis})
                stats[f"{joint_type} joints"] += 1
        if kind in ("revolute", "prismatic"):
            name = incoming.get("name")
            limit = incoming.find("limit")
            joint = {"name": name, "type": "hinge" if kind == "revolute" else "slide",
                     "axis": fmt(vec(incoming.find("axis").get("xyz"))),
                     "range": f"{limit.get('lower')} {limit.get('upper')}"}
            effort = float(limit.get("effort"))
            if name in ARM_JOINTS:
                joint_damping, friction = table[int(name[4]) - 1]
                joint.update({"damping": f"{joint_damping:g}", "frictionloss": f"{friction:g}",
                              "armature": f"{ARM_ARMATURE:g}"})
            elif name == LIFT:
                joint.update({"damping": f"{LIFT_DAMPING:g}", "armature": f"{LIFT_ARMATURE:g}"})
            elif name in (f"{p}_joint1" for p in GRIPPERS.values()):
                joint.update({"damping": f"{GRIPPER_DYNAMICS[0]:g}", "frictionloss": f"{GRIPPER_DYNAMICS[1]:g}",
                              "armature": f"{GRIPPER_ARMATURE:g}", "solreflimit": GRIPPER_SOLREF})
            else:  # the gripper linkage
                joint.update({"armature": f"{GRIPPER_LINK_ARMATURE:g}", "solreflimit": GRIPPER_SOLREF})
            if incoming.find("mimic") is None:
                # Joint-level limit on the total actuator torque, including gravity compensation (URDF effort).
                joint.update({"actuatorfrcrange": f"{-effort:g} {effort:g}"})
                if gravcomp:
                    joint["actuatorgravcomp"] = "true"
            ET.SubElement(body, "joint", joint)
            stats[f"{joint['type']} joints"] += 1
        elif kind == "continuous":
            if incoming.get("name") not in WHEELS:
                raise ValueError(f"unexpected continuous joint {incoming.get('name')}")
            if mobile_base:  # an omni wheel: an unactuated, unlimited hinge that model/mobile_base.py rolls
                ET.SubElement(body, "joint", {"name": incoming.get("name"), "type": "hinge",
                                              "axis": fmt(vec(incoming.find("axis").get("xyz")))})
                stats["hinge joints"] += 1
            # with the base fixed to the world, the wheels are welded
        elif kind not in (None, "fixed"):
            raise ValueError(f"unsupported joint type {kind} at {incoming.get('name')}")
        inertial = link.find("inertial")
        if inertial is not None:
            frame = transform(inertial)
            moments = [float(inertial.find("inertia").get(k)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")]
            tensor = np.array([[moments[0], moments[1], moments[2]], [moments[1], moments[3], moments[4]],
                               [moments[2], moments[4], moments[5]]])
            tensor = frame[:3, :3] @ tensor @ frame[:3, :3].T  # MJCF fullinertia cannot carry an orientation
            ET.SubElement(body, "inertial", {"pos": fmt(frame[:3, 3]), "mass": inertial.find("mass").get("value"),
                "fullinertia": fmt([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]])})
        add_link_geoms(link, body)
        for camera_body, camera, pos, quat, fovy in CAMERAS:
            if camera_body == link_name:
                ET.SubElement(body, "camera", {"name": camera, "pos": fmt(pos), "quat": fmt(quat), "fovy": f"{fovy:g}"})
                stats["cameras"] += 1
        add_frames(link_name, body, np.eye(4))
        for joint in child_joints[link_name]:
            if joint.find("child").get("link") not in virtual:
                add_body(body, joint.find("child").get("link"))

    add_body(worldbody, "base_link")

    actuator = ET.SubElement(model, "actuator")
    for name in ARM_JOINTS:
        limit = joints[name].find("limit")
        ET.SubElement(actuator, "position", {"name": name.replace("_joint", ""), "joint": name, "kp": f"{ARM_KP:g}",
            "kv": f"{ARM_KV:g}", "ctrlrange": f"{limit.get('lower')} {limit.get('upper')}",
            "forcerange": f"-{limit.get('effort')} {limit.get('effort')}"})
    limit = joints[LIFT].find("limit")
    ET.SubElement(actuator, "position", {"name": "lift", "joint": LIFT, "kp": f"{LIFT_KP:g}", "kv": f"{LIFT_KV:g}",
        "ctrlrange": f"{limit.get('lower')} {limit.get('upper')}",
        "forcerange": f"-{limit.get('effort')} {limit.get('effort')}"})
    for side, prefix in GRIPPERS.items():  # motor torque in N m; the joint's actuatorfrcrange (URDF effort) caps it
        ET.SubElement(actuator, "motor", {"name": f"{side}_gripper", "joint": f"{prefix}_joint1",
                                          "ctrlrange": f"{CLOSE_TORQUE:g} {OPEN_TORQUE:g}"})
    # Base: integrated-velocity servos. The activation is the target pose, which the command moves; the force pulls the
    # joint to it (kp) and damps its velocity (kv). Yaw is unlimited, so it uses the general form of <intvelocity>.
    base_axes = (("base_x", BASE_SPEED, BASE_FORCE, BASE_KP, BASE_KV),
                 ("base_y", BASE_SPEED, BASE_FORCE, BASE_KP, BASE_KV),
                 ("base_yaw", BASE_TURN_RATE, BASE_TORQUE, BASE_YAW_KP, BASE_YAW_KV))
    for name, limit, force, kp, kv in base_axes if mobile_base else ():
        attrs = {"name": name, "joint": name, "dyntype": "integrator", "biastype": "affine", "gainprm": f"{kp:g}",
                 "biasprm": f"0 {-kp:g} {-kv:g}", "ctrlrange": f"{-limit:g} {limit:g}",
                 "forcerange": f"{-force:g} {force:g}", "actearly": "true"}
        if name != "base_yaw":
            attrs.update({"actlimited": "true", "actrange": f"{-BASE_TRAVEL:g} {BASE_TRAVEL:g}"})
        ET.SubElement(actuator, "general", attrs)

    contact = ET.SubElement(model, "contact")
    pairs = list(EXCLUDES)
    if mobile_base:  # the wheels roll above the floor
        pairs += [("world", joints[wheel].find("child").get("link")) for wheel in WHEELS]
    for prefix in GRIPPERS.values():
        parts = [f"{prefix}_base_link", f"{prefix}_link1"]
        fingers = [[f"{prefix}_link{k}" for k in finger] for finger in FINGERS]
        group = parts + fingers[0] + fingers[1]
        pairs += [(a, b) for i, a in enumerate(group) for b in group[i + 1:]
                  if not (a in fingers[0] and b in fingers[1])]
    for first, second in pairs:
        ET.SubElement(contact, "exclude", {"body1": first, "body2": second})
    stats["contact exclusions"] = len(pairs)

    equality = ET.SubElement(model, "equality")
    for prefix in GRIPPERS.values():
        ET.SubElement(equality, "joint", {"name": f"{prefix}_coupling", "joint1": f"{prefix}_joint2",
                                          "joint2": f"{prefix}_joint1", "polycoef": f"0 {GRIPPER_COUPLING:g} 0 0 0",
                                          "solref": GRIPPER_SOLREF})
    for joint in joints.values():
        mimic = joint.find("mimic")
        if mimic is not None:
            ET.SubElement(equality, "joint", {"name": f"{joint.get('name')}_mimic", "joint1": joint.get("name"),
                "joint2": mimic.get("joint"),
                "polycoef": f"{float(mimic.get('offset', 0)):g} {float(mimic.get('multiplier', 1)):g} 0 0 0",
                "solref": GRIPPER_SOLREF})

    if mobile_base:
        custom = ET.SubElement(model, "custom")
        ET.SubElement(custom, "numeric", {"name": "wheel_radius", "data": fmt([WHEEL_RADIUS])})  # for mobile_base.py

    keyframe_qpos = {**home, LIFT: lift_home, **{name: 0.0 for name, _, _ in BASE_JOINTS}, **dict.fromkeys(WHEELS, 0.0)}
    for prefix in GRIPPERS.values():
        keyframe_qpos[f"{prefix}_joint1"] = GRIPPER_HOME
        keyframe_qpos[f"{prefix}_joint2"] = GRIPPER_COUPLING * GRIPPER_HOME
    for joint in joints.values():
        mimic = joint.find("mimic")
        if mimic is not None:
            keyframe_qpos[joint.get("name")] = float(mimic.get("offset", 0)) + float(mimic.get("multiplier", 1)) * \
                keyframe_qpos[mimic.get("joint")]
    order = [element.get("name") for element in worldbody.iter("joint")]  # the base and wheels only when they move
    ctrl = [home[name] for name in ARM_JOINTS] + [lift_home] + [gripper_torque(GRIPPER_HOME_COMMAND)] * len(GRIPPERS)
    ctrl += [0.0] * len(BASE_JOINTS) if mobile_base else []
    keyframe = ET.SubElement(model, "keyframe")
    ET.SubElement(keyframe, "key", {"name": "home", "qpos": fmt([keyframe_qpos[name] for name in order]),
                                    "ctrl": fmt(ctrl)})

    ET.indent(model, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(ET.tostring(model, encoding="unicode") + "\n", encoding="utf-8")
    stats.update({"materials": len(materials), "actuators": len(ctrl), "equalities": len(equality)})
    if verbose:
        print(f"wrote {output}: " + ", ".join(f"{value} {key}" for key, value in stats.items()))
    return dict(stats)


def mobile_scene() -> str:
    """scene_mobile.xml: scene.xml with the drivable-base robot."""
    text = SCENE.read_text(encoding="utf-8")
    for old, new in (('<mujoco model="teleavatar_20260928_scene">',
                      '<mujoco model="teleavatar_20260928_mobile_scene">'),
                     ('<include file="robot.xml"/>', '<include file="robot_mobile.xml"/>')):
        if text.count(old) != 1:
            raise ValueError(f"{SCENE.name} no longer has exactly one {old}")
        text = text.replace(old, new)
    return text.replace("\n", "\n  <!-- Generated by model/convert.py from scene.xml; do not edit. -->\n", 1)


def build_all(urdf_path: Path = DEFAULT_URDF, **options) -> None:
    """The default outputs: robot.xml (base fixed), robot_mobile.xml (drivable base) and scene_mobile.xml."""
    build(urdf_path, DEFAULT_OUTPUT, **options)
    build(urdf_path, MOBILE_OUTPUT, mobile_base=True, **options)
    MOBILE_SCENE.write_text(mobile_scene(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path,
                        help="write only this file (default: robot.xml, robot_mobile.xml and scene_mobile.xml)")
    parser.add_argument("--mobile-base", action="store_true", help="with --output: the drivable-base variant")
    parser.add_argument("--texture-size", type=int, default=1024,
                        help="colour texture size relative to the vendor's 2048 (0 = use the vendor files as is)")
    parser.add_argument("--visual-faces", type=int, default=0,
                        help="use the decimated visual meshes of model/build_visual_lite.py N (0 = the vendor's)")
    parser.add_argument("--no-gravcomp", action="store_true", help="no gravity compensation in the joint controllers")
    parser.add_argument("--damping", choices=sorted(DAMPING_TABLES), default="measured")
    parser.add_argument("--lift-home", type=float, default=LIFT_HOME, help="lift position in the home keyframe, m")
    parser.add_argument("--check-only", action="store_true", help="build both in temporary files and compile them")
    args = parser.parse_args()
    if args.mobile_base and args.output is None:
        parser.error("--mobile-base selects the variant for --output; without --output both are written")
    options = dict(texture_size=args.texture_size, visual_faces=args.visual_faces, gravcomp=not args.no_gravcomp,
                   damping=args.damping, lift_home=args.lift_home, verbose=True)
    if args.check_only:
        import mujoco
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            for name, mobile in (("robot.xml", False), ("robot_mobile.xml", True)):
                output = Path(directory) / name
                build(args.urdf, output, mobile_base=mobile, **options)
                model = mujoco.MjModel.from_xml_path(str(output))
                print(f"MuJoCo compile check passed for {name}: nq={model.nq} nv={model.nv} nu={model.nu} "
                      f"neq={model.neq}")
    elif args.output is not None:
        build(args.urdf, args.output, mobile_base=args.mobile_base, **options)
    else:
        build_all(args.urdf, **options)


if __name__ == "__main__":
    main()
