"""The robot's cameras as the real ones see: a fisheye lens at each eye of the three stereo pairs.

MuJoCo renders pinhole cameras only. The real eyes have fisheye lenses (OpenCV fisheye model, developer docs §5.5),
and their images reach a policy unrectified, at half the sensors' resolution: 960 x 960 for each head eye, 640 x 400
(width x height) for each wrist eye. FisheyeCamera renders a model camera through such a lens: up to five 90 deg
pinhole views from the camera's position (the faces of a cube: front, left, right, up, down), then, for every output
pixel, the ray that the lens model gives, looked up on the face it falls on (bilinear, 8-bit fixed-point weights). The
faces are face_size(lens) pixels square in every tool, so an eye's image does not depend on which tool renders it:
960 for the robot's image sizes (one renderer then serves the three eyes a policy gets), the image's larger side for
other sizes. The lookup table is built once per lens.

  camera = FisheyeCamera(model, "head_left_eye")      # the lens from LENSES; makes its own face renderer
  image = camera.render(data)                          # (960, 960, 3) uint8; black beyond the lens model's range
  ids = camera.render_segmentation(data)               # (960, 960, 2) int32: object id and type, -1 for none
  u, v, seen = camera.project(data, world_points)      # where points land in the image

Lenses (LENSES; model/README.md, "Where the values come from", lists each value's source):
- head eyes: the calibration the vendor emailed on 2026-10-08 for one head eye at 1920 x 1920, halved for the
  960 x 960 stream; both eyes get it
- wrist eyes: ESTIMATED until the robot's calibration is known: an ideal equidistant lens (k1..k4 = 0) that spans the
  spec's 120 deg across the 640-pixel width, centred

Conventions (CLAUDE.md, "Rendering and camera images"): a MuJoCo camera looks along its -z with +x image right and
+y image up; the OpenCV camera frame is MuJoCo's turned 180 deg about x (+z forward, +y down), with pixel (u, v) =
(column, row), integer at pixel centres, row 0 at the top. OpenCV fisheye: theta = atan2(sqrt(X^2 + Y^2), Z),
theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8), u = fx theta_d X / sqrt(X^2 + Y^2) + cx,
v = fy theta_d Y / sqrt(X^2 + Y^2) + cy. theta_d is used only while it grows with theta: past that (the head image's
corners, 0.75% of it) there is no ray, and the pixels stay black.

Lighting: MuJoCo's headlight shines along the camera that the scene was last updated with. The faces are rendered by
turning that camera without updating the scene again, so every face keeps the headlight along the eye's own view, as
an ordinary render of it would (lit per face, the image would show seams).
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import cached_property

import mujoco
import numpy as np

_C = math.sqrt(0.5)
# Each face camera's orientation in the eye camera's own (MuJoCo) frame, (w, x, y, z).
FACE_QUATS = {
    "front": (1.0, 0.0, 0.0, 0.0),
    "left": (_C, 0.0, _C, 0.0),     # +90 deg about y: looks along the eye's -x (image left)
    "right": (_C, 0.0, -_C, 0.0),   # -90 deg about y: along +x (image right)
    "up": (_C, _C, 0.0, 0.0),       # +90 deg about x: along +y (image up)
    "down": (_C, -_C, 0.0, 0.0),    # -90 deg about x: along -y (image down)
}
FACES = tuple(FACE_QUATS)
CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])  # OpenCV camera frame <-> MuJoCo camera frame (its own inverse)
_RGB = np.dtype((np.void, 3))            # one RGB pixel as a single item, for fast gathers


def _quat_mat(quat) -> np.ndarray:
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


FACE_MATS = {face: _quat_mat(quat) for face, quat in FACE_QUATS.items()}  # columns: the face's axes, in the eye frame


@dataclass(frozen=True)
class Lens:
    """An OpenCV fisheye lens (k = 0: an ideal equidistant one) on a width x height image."""
    fx: float
    fy: float
    cx: float
    cy: float
    k: tuple = (0.0, 0.0, 0.0, 0.0)
    width: int = 960
    height: int = 960

    def scaled(self, width: int, height: int) -> "Lens":
        """The same lens on an image resized to width x height (pixel centres at integers, as in OpenCV)."""
        sx, sy = width / self.width, height / self.height
        return replace(self, fx=self.fx * sx, fy=self.fy * sy, cx=(self.cx + 0.5) * sx - 0.5,
                       cy=(self.cy + 0.5) * sy - 0.5, width=width, height=height)

    def theta_d(self, theta):
        k1, k2, k3, k4 = self.k
        t2 = np.square(theta)
        return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def _slope(self, theta):
        k1, k2, k3, k4 = self.k
        t2 = np.square(theta)
        return 1.0 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))

    @cached_property
    def theta_max(self) -> float:
        """The widest ray angle the model describes: where theta_d stops growing (pi if it never does)."""
        grid = np.linspace(0.0, math.pi, 200001)
        falling = np.flatnonzero(self._slope(grid) <= 0.0)
        if not falling.size:
            return math.pi
        low, high = grid[falling[0] - 1], grid[falling[0]]
        for _ in range(60):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if self._slope(middle) > 0 else (low, middle)
        return 0.5 * (low + high)

    def theta(self, theta_d) -> np.ndarray:
        """The inverse of theta_d (a dense table, then two Newton steps); nan beyond the lens model's range."""
        theta_max = self.theta_max
        table = np.linspace(0.0, theta_max, 100001)
        theta_d = np.asarray(theta_d, dtype=float)
        theta = np.interp(theta_d, self.theta_d(table), table)
        for _ in range(2):
            theta = np.clip(theta - (self.theta_d(theta) - theta_d) / self._slope(theta), 0.0, theta_max)
        return np.where(theta_d <= self.theta_d(theta_max), theta, np.nan)

    def project(self, points_cv):
        """Points in the OpenCV camera frame (..., 3) -> u, v (pixels) and whether the lens model covers the ray."""
        points = np.asarray(points_cv, dtype=float)
        rho = np.hypot(points[..., 0], points[..., 1])
        theta = np.arctan2(rho, points[..., 2])
        scale = self.theta_d(theta) / np.where(rho > 0, rho, 1.0)
        return (self.fx * scale * points[..., 0] + self.cx, self.fy * scale * points[..., 1] + self.cy,
                theta <= self.theta_max)

    def rays(self):
        """Unit rays (H, W, 3) through every pixel centre, in the OpenCV frame, and where the lens model has one."""
        v, u = np.mgrid[0:self.height, 0:self.width].astype(float)
        x, y = (u - self.cx) / self.fx, (v - self.cy) / self.fy
        theta_d = np.hypot(x, y)
        theta = self.theta(theta_d)
        valid = np.isfinite(theta)
        theta = np.where(valid, theta, 0.0)
        scale = np.where(theta_d > 0, np.sin(theta) / np.where(theta_d > 0, theta_d, 1.0), 1.0)
        return np.stack([x * scale, y * scale, np.cos(theta)], axis=-1), valid

    def field_of_view(self) -> tuple:
        """Degrees across the image through the principal point, edge to edge: (horizontal, vertical)."""
        edge = lambda d, f: math.degrees(float(self.theta(abs(d) / f)))
        return (edge(self.cx + 0.5, self.fx) + edge(self.width - 0.5 - self.cx, self.fx),
                edge(self.cy + 0.5, self.fy) + edge(self.height - 0.5 - self.cy, self.fy))


# The head eyes: one head eye's calibration as the vendor emailed it (2026-10-08), for 1920 x 1920 images; the stream
# halves it (developer docs §3.1.2 and §5.2: the eyes are downsampled to 960 x 960 before stitching). ASSUMED: a plain
# 2x downscale, and the same calibration for both eyes (the email gives one eye, and does not say which).
HEAD_1920 = Lens(fx=957.7522723586153, fy=957.058449178608, cx=953.0942242558152, cy=961.8638796212753,
                 k=(-0.021403661461248613, 0.0025639478168715288, -0.0031704737916179947, -0.0018083392738524986),
                 width=1920, height=1920)
HEAD_EYE = HEAD_1920.scaled(960, 960)
# The wrist eyes, ESTIMATED: the spec (user manual §2.3) gives 120 x 76 deg on 1280 x 800 eyes; an equidistant lens
# with f = 640 / (pi / 3) px spans 120 deg across them (75 deg up and down), centred. Halved for the 640 x 400 stream.
WRIST_EYE = Lens(fx=640 / (math.pi / 3), fy=640 / (math.pi / 3), cx=639.5, cy=399.5, width=1280,
                 height=800).scaled(640, 400)
LENSES = {
    "head_left_eye": HEAD_EYE, "head_right_eye": HEAD_EYE,
    "left_wrist_left_eye": WRIST_EYE, "left_wrist_right_eye": WRIST_EYE,
    "right_wrist_left_eye": WRIST_EYE, "right_wrist_right_eye": WRIST_EYE,
}
# Cube-face size (px) for the robot's image sizes, 960 x 960 (head) and 640 x 400 (wrist): CHOSEN. One renderer then
# serves the three eyes a policy gets, and for the wrists 960-px faces come nearer a 1920-px reference than 640-px ones
# (up to 1.0% of pixels off by more than 10/255, against up to 1.8%).
FACE_SIZE = 960
STREAM_SIZES = ((960, 960), (640, 400))


def face_size(lens: Lens) -> int:
    """The cube-face size an image is rendered from, the same in every tool: FACE_SIZE at the robot's image sizes,
    the image's larger side at other (preview) sizes."""
    return FACE_SIZE if (lens.width, lens.height) in STREAM_SIZES else max(lens.width, lens.height)


# The eyes the vendor's pi0.5 deployment feeds the policy (github.com/dexteleop/openpi, examples/teleavatar_v2/
# ros2_interface.py): the head's left eye and each wrist's inner eye, which faces the middle of the desk.
POLICY_EYES = {"head_camera": "head_left_eye", "left_color": "left_wrist_right_eye",
               "right_color": "right_wrist_left_eye"}


def face_fovy(face_size: int, margin: int) -> float:
    """fovy (deg) of a face whose 45 deg rays fall `margin` pixels inside its border, so that bilinear sampling never
    reads past it: 2 atan(N / (N - 2 margin)), 90.12 deg for N = 960 and margin 1."""
    return 2.0 * math.degrees(math.atan(face_size / (face_size - 2.0 * margin)))


class CubeRemap:
    """For each output pixel: the face its ray falls on and where, then bilinear (colour) or nearest (labels) lookups.

    The used faces sit one after another in a flat buffer, rows top first (as mujoco.Renderer returns them), followed
    by one extra entry that the pixels without a ray read (black, or label -1)."""

    def __init__(self, rays_mj: np.ndarray, valid: np.ndarray, face_size: int, margin: int = 1,
                 threads: int | None = None) -> None:
        n, (h, w) = face_size, valid.shape
        rays, valid = rays_mj.reshape(-1, 3), valid.reshape(-1)
        best = np.argmax(np.stack([rays @ -FACE_MATS[face][:, 2] for face in FACES], axis=1), axis=1)
        focal, centre = (n - 2.0 * margin) / 2.0, (n - 1) / 2.0
        u, v = np.full(len(rays), np.nan), np.full(len(rays), np.nan)
        for i, face in enumerate(FACES):
            chosen = valid & (best == i)
            local = rays[chosen] @ FACE_MATS[face]  # the ray in the face camera's frame
            u[chosen] = centre + focal * local[:, 0] / -local[:, 2]
            v[chosen] = centre - focal * local[:, 1] / -local[:, 2]
        inside = (u >= 0) & (u <= n - 1) & (v >= 0) & (v <= n - 1)
        if np.any(valid & ~inside):
            raise RuntimeError(f"{np.count_nonzero(valid & ~inside)} rays fall outside every face")
        self.pixels_per_face = {face: int(np.count_nonzero(valid & (best == i))) for i, face in enumerate(FACES)}
        self.faces = tuple(face for face in FACES if self.pixels_per_face[face])
        slot = np.array([self.faces.index(face) if face in self.faces else 0 for face in FACES])
        self.shape, self.face_size, self.invalid_pixels = (h, w), n, int(np.count_nonzero(~valid))
        self.size = len(self.faces) * n * n
        base = np.where(valid, slot[best], 0).astype(np.int64) * n * n
        u, v = np.where(valid, u, 0.0), np.where(valid, v, 0.0)
        u0, v0 = np.minimum(np.floor(u).astype(np.int64), n - 2), np.minimum(np.floor(v).astype(np.int64), n - 2)
        au, av = u - u0, v - v0
        corner = base + v0 * n + u0
        self.index = np.stack([corner, corner + 1, corner + n, corner + n + 1])
        self.index[:, ~valid] = self.size
        weight = np.stack([(1 - au) * (1 - av), au * (1 - av), (1 - au) * av, au * av])
        fixed = np.rint(weight * 256).astype(np.int32)  # 8-bit fixed point; each pixel's four weights sum to 256
        fixed[np.argmax(fixed, axis=0), np.arange(fixed.shape[1])] += 256 - fixed.sum(axis=0)
        fixed[:, ~valid] = np.array([256, 0, 0, 0])[:, None]
        self.weight = np.repeat(fixed.astype(np.uint16)[:, :, None], 3, axis=2)  # repeated per channel: 4x faster
        self.nearest = np.where(valid, base + np.rint(v).astype(np.int64) * n + np.rint(u).astype(np.int64),
                                self.size)
        self.threads = threads or min(8, os.cpu_count() or 1)
        edges = np.linspace(0, h * w, self.threads + 1).astype(int)
        self._chunks = list(zip(edges[:-1], edges[1:]))
        self._pool = ThreadPoolExecutor(self.threads) if self.threads > 1 else None

    def buffer(self, channels: int, dtype, fill) -> tuple:
        """A flat face buffer (size + 1, channels) and its (faces, N, N, channels) view to render into."""
        flat = np.zeros((self.size + 1, channels), dtype)
        flat[self.size] = fill
        return flat, flat[:self.size].reshape(-1, self.face_size, self.face_size, channels)

    def remap(self, flat: np.ndarray) -> np.ndarray:
        """(size + 1, 3) uint8 faces -> (H, W, 3) uint8, bilinear."""
        out = np.empty((self.shape[0] * self.shape[1], 3), np.uint8)
        items, index, weight = flat.view(_RGB).ravel(), self.index, self.weight

        def work(chunk):
            start, end = chunk
            total = items[index[0, start:end]].view(np.uint8).reshape(-1, 3).astype(np.uint16) * weight[0, start:end]
            for k in range(1, 4):
                total += items[index[k, start:end]].view(np.uint8).reshape(-1, 3) * weight[k, start:end]
            out[start:end] = (total + 128) >> 8

        if self._pool is None:
            work((0, len(out)))
        else:
            list(self._pool.map(work, self._chunks))
        return out.reshape(*self.shape, 3)

    def remap_nearest(self, flat: np.ndarray) -> np.ndarray:
        """(size + 1, C) faces -> (H, W, C), the nearest face pixel."""
        return flat[self.nearest].reshape(*self.shape, flat.shape[1])

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


class FisheyeCamera:
    """What a model camera sees through a fisheye lens.

    lens: default LENSES[camera]. renderer: a face_size(lens) square mujoco.Renderer to share between cameras (one is
    made otherwise). The model's offscreen buffer must be at least that large (<visual><global offwidth
    offheight>). margin: face pixels kept beyond the 90 deg view, for bilinear sampling. The camera must be a plain
    perspective one (fovy, no intrinsics): the faces replace its frustum.
    """

    def __init__(self, model: mujoco.MjModel, camera: str, lens: Lens | None = None,
                 renderer: mujoco.Renderer | None = None, margin: int = 1, shadows: bool = True) -> None:
        self.model, self.camera = model, camera
        self.camera_id = model.camera(camera).id
        self.lens = LENSES[camera] if lens is None else lens
        size = face_size(self.lens)
        if renderer is None:
            renderer, self._own_renderer = mujoco.Renderer(model, size, size), True
        else:
            self._own_renderer = False
        if (renderer.width, renderer.height) != (size, size):
            raise ValueError(f"{camera} renders from {size} x {size} faces; the renderer is "
                             f"{renderer.width} x {renderer.height}")
        self.renderer = renderer
        self.shadows = shadows
        self.fovy = face_fovy(renderer.height, margin)
        rays_cv, valid = self.lens.rays()
        self.lut = CubeRemap(rays_cv @ CV_FROM_MJ, valid, renderer.height, margin)
        self._colour_flat, self._colour = self.lut.buffer(3, np.uint8, 0)
        self._label_flat, self._label = self.lut.buffer(2, np.int32, -1)
        self._labels = None  # a second renderer, without multisampling, made on the first segmentation

    def _faces(self, renderer: mujoco.Renderer, data: mujoco.MjData, scene_option):
        """Set the renderer's scene up for each used face in turn (yields the face's slot)."""
        renderer.update_scene(data, camera=self.camera_id, scene_option=scene_option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.shadows
        eye = data.cam_xmat[self.camera_id].reshape(3, 3)
        for slot, face in enumerate(self.lut.faces):
            axes = eye @ FACE_MATS[face]
            for camera in renderer.scene.camera:  # both of the scene's cameras (the same for a mono view)
                camera.forward[:] = -axes[:, 2]
                camera.up[:] = axes[:, 1]
                camera.orthographic = 0
                camera.frustum_width = 0.0  # 0: the width follows the (square) image; a camera's intrinsics set it
                camera.frustum_center = 0.0
                camera.frustum_top = camera.frustum_near * math.tan(math.radians(self.fovy) / 2)
                camera.frustum_bottom = -camera.frustum_top
            yield slot

    def render(self, data: mujoco.MjData, scene_option=None) -> np.ndarray:
        """(H, W, 3) uint8 image."""
        for slot in self._faces(self.renderer, data, scene_option):
            self.renderer.render(out=self._colour[slot])
        return self.lut.remap(self._colour_flat)

    def render_segmentation(self, data: mujoco.MjData, scene_option=None) -> np.ndarray:
        """(H, W, 2) int32: the object id and type of each pixel (as mujoco.Renderer's segmentation), -1 for none."""
        if self._labels is None:
            samples = self.model.vis.quality.offsamples  # with multisampling, ids blend at object edges
            self.model.vis.quality.offsamples = 0
            try:
                self._labels = mujoco.Renderer(self.model, self.renderer.height, self.renderer.width)
            finally:
                self.model.vis.quality.offsamples = samples
            self._labels.enable_segmentation_rendering()
        for slot in self._faces(self._labels, data, scene_option):
            self._label[slot] = self._labels.render()  # (render's out= takes colour images only)
        return self.lut.remap_nearest(self._label_flat)

    def project(self, data: mujoco.MjData, points) -> tuple:
        """World points (..., 3) -> u, v (pixels) and whether each lands on the image."""
        rotation = data.cam_xmat[self.camera_id].reshape(3, 3) @ CV_FROM_MJ
        u, v, covered = self.lens.project((np.asarray(points, dtype=float) - data.cam_xpos[self.camera_id]) @ rotation)
        inside = (u >= -0.5) & (u <= self.lens.width - 0.5) & (v >= -0.5) & (v <= self.lens.height - 0.5)
        return u, v, covered & inside

    def close(self) -> None:
        self.lut.close()
        if self._labels is not None:
            self._labels.close()
            self._labels = None
        if self._own_renderer:
            self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
