"""Persistent FEM material markers and a pinhole observation camera (NumPy).

Bindings are computed once in the rest mesh. They are never reassigned to a
nearest vertex during deformation. All lengths are metres; image coordinates
are pixels, with +u right, +v down and camera +z forward.
"""
from dataclasses import dataclass

import numpy as np


def _points(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite (N, 3)")
    return value


def _readonly(value):
    value = np.array(value, copy=True)
    value.setflags(write=False)
    return value


def _elements(value, width, vertex_count):
    value = np.asarray(value)
    if (value.ndim != 2 or value.shape[1] != width or len(value) == 0
            or not np.issubdtype(value.dtype, np.integer)
            or value.min() < 0 or value.max() >= vertex_count):
        raise ValueError(f"elements must be nonempty integer (E, {width}) vertex indices")
    return value.astype(np.int64, copy=False)


@dataclass(frozen=True)
class MaterialBinding:
    """Fixed vertex indices and barycentric weights for material sample points."""

    rest_m: np.ndarray
    element_ids: np.ndarray
    vertex_ids: np.ndarray
    weights: np.ndarray
    vertex_count: int

    @classmethod
    def bind(cls, rest_positions, elements, points_m, *, tolerance_m=1e-9):
        """Bind to triangles (surface) or tetrahedra (volume), rejecting outsiders.

        Shared-face/edge points use the first containing element deterministically.
        No extrapolation or nearest-element fallback is performed. Triangle samples
        must lie on the rest triangle, not just within its projected footprint.
        """
        rest = _points(rest_positions, "rest_positions")
        points = _points(points_m, "points_m")
        elements = np.asarray(elements)
        if elements.ndim != 2 or elements.shape[1] not in (3, 4):
            raise ValueError("elements must be triangles or tetrahedra")
        elements = _elements(elements, elements.shape[1], len(rest))
        if not np.isfinite(tolerance_m) or tolerance_m < 0:
            raise ValueError("tolerance_m must be finite and nonnegative")
        vertices = rest[elements]
        edges = vertices[:, 1:] - vertices[:, :1]
        scale = np.linalg.norm(edges, axis=2).max(axis=1)
        if elements.shape[1] == 4:
            basis = np.swapaxes(edges, 1, 2)
            measure = np.abs(np.linalg.det(basis))
            degenerate = measure <= 1e-12 * scale**3
        else:
            basis = np.swapaxes(edges, 1, 2)
            measure = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
            degenerate = measure <= 1e-12 * scale**2
        if np.any(degenerate):
            raise ValueError("rest mesh contains degenerate elements")
        # Pseudoinverse only for the rectangular triangle basis, never a fallback
        # for singular tetrahedra. Bounding boxes keep per-point work small.
        inverse = np.linalg.inv(basis) if elements.shape[1] == 4 else np.linalg.pinv(basis)
        lower = vertices.min(axis=1) - tolerance_m
        upper = vertices.max(axis=1) + tolerance_m
        element_ids, weights = [], []
        for index, point in enumerate(points):
            candidates = np.flatnonzero(np.all((point >= lower) & (point <= upper), axis=1))
            delta = point - vertices[candidates, 0]
            local = np.einsum("nij,nj->ni", inverse[candidates], delta)
            bary = np.column_stack((1 - local.sum(axis=1), local))
            eps = tolerance_m * np.linalg.norm(inverse[candidates], axis=(1, 2)) + 1e-12
            residual = np.linalg.norm(np.einsum("nij,nj->ni", basis[candidates], local) - delta, axis=1)
            inside = ((bary >= -eps[:, None]).all(axis=1)
                      & (bary <= 1 + eps[:, None]).all(axis=1)
                      & (residual <= tolerance_m + 1e-14))
            # Prefer an actual containing element over an earlier neighbouring
            # element admitted only by tolerance. Otherwise a point near a face
            # could be unnecessarily snapped to the wrong side of that face.
            strict = inside & (bary >= -1e-12).all(axis=1) & (bary <= 1 + 1e-12).all(axis=1)
            found = np.flatnonzero(strict if np.any(strict) else inside)
            if len(found) == 0:
                raise ValueError(f"material point {index} is outside the supplied rest mesh")
            selected = int(found[0])
            weight = np.clip(bary[selected], 0, 1)
            weight /= weight.sum()
            element_ids.append(candidates[selected])
            weights.append(weight)
        ids = np.asarray(element_ids, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1, elements.shape[1])
        return cls(_readonly(points), _readonly(ids), _readonly(elements[ids]),
                   _readonly(weights), len(rest))

    def evaluate(self, positions):
        q = _points(positions, "positions")
        if len(q) != self.vertex_count:
            raise ValueError("positions must retain the bound mesh's vertex ordering and count")
        return np.einsum("mi,mij->mj", self.weights, q[self.vertex_ids])


def planar_marker_grid(rest_positions, *, depth_m=0.0, spacing_m=.002,
                       margin_m=.001):
    """Marker centres in a horizontal layer below the rest mesh's highest z.

    This convenience factory assumes a rectangular gel. For curved/irregular
    gels supply explicit material points to MaterialBinding.bind instead.
    Grid origin is world x=y=0; the margin keeps finite marker disks off edges.
    """
    rest = _points(rest_positions, "rest_positions")
    if len(rest) == 0:
        raise ValueError("rest_positions must not be empty")
    if (not np.all(np.isfinite([depth_m, spacing_m, margin_m]))
            or depth_m < 0 or spacing_m <= 0 or margin_m < 0
            or depth_m > np.ptp(rest[:, 2]) + 1e-9):
        raise ValueError("invalid marker depth, spacing or margin")
    low, high = rest.min(axis=0), rest.max(axis=0)
    axes = [np.arange(np.ceil((low[i] + margin_m) / spacing_m),
                      np.floor((high[i] - margin_m) / spacing_m) + 1) * spacing_m
            for i in (0, 1)]
    x, y = np.meshgrid(*axes)
    if x.size == 0:
        raise ValueError("marker grid is empty")
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, high[2] - depth_m)))


def top_surface_triangles(rest_positions, triangles, *, tolerance_m=1e-8):
    """Select the flat top coating of a rest mesh, retaining vertex indices."""
    rest = _points(rest_positions, "rest_positions")
    triangles = _elements(triangles, 3, len(rest))
    selected = triangles[np.all(np.abs(rest[triangles, 2] - rest[:, 2].max()) <= tolerance_m, axis=1)]
    if len(selected) == 0:
        raise ValueError("no planar top surface found; supply coating triangles explicitly")
    return selected


@dataclass(frozen=True)
class PinholeCamera:
    """OpenCV-style camera: X_cam = R_world_to_camera @ X_world + t.

    R must be a proper rotation. For a camera below a +z-up gel looking upwards,
    R=I is valid: world +y then appears DOWN in the image. Flipping only y would
    be a reflection, not a physical camera rotation. No lens/refraction model.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: np.ndarray
    translation_m: np.ndarray
    near_m: float = 1e-6

    def __post_init__(self):
        r = np.asarray(self.rotation, dtype=float)
        t = np.asarray(self.translation_m, dtype=float)
        if (isinstance(self.width, bool) or isinstance(self.height, bool)
                or int(self.width) != self.width or int(self.height) != self.height
                or min(self.width, self.height) < 2
                or not np.all(np.isfinite([self.fx, self.fy, self.cx, self.cy, self.near_m]))
                or min(self.fx, self.fy, self.near_m) <= 0):
            raise ValueError("invalid camera intrinsics, resolution or near plane")
        if (r.shape != (3, 3) or not np.all(np.isfinite(r))
                or not np.allclose(r @ r.T, np.eye(3), atol=1e-9, rtol=0)
                or not np.isclose(np.linalg.det(r), 1, atol=1e-9, rtol=0)
                or t.shape != (3,) or not np.all(np.isfinite(t))):
            raise ValueError("camera requires a proper rotation and finite translation")
        object.__setattr__(self, "rotation", _readonly(r))
        object.__setattr__(self, "translation_m", _readonly(t))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))

    def camera_points(self, positions):
        return _points(positions, "positions") @ self.rotation.T + self.translation_m

    def project(self, positions):
        """Return UV, positive optical z and in-frame mask (NOT an occlusion test).

        Points behind the near plane have NaN UV. In-front offscreen points keep
        finite UV, allowing flows to be exported with an explicit validity mask.
        """
        q = self.camera_points(positions)
        uv = np.full((len(q), 2), np.nan)
        front = q[:, 2] > self.near_m
        uv[front] = q[front, :2] / q[front, 2, None] * [self.fx, self.fy] + [self.cx, self.cy]
        valid = (front & (uv[:, 0] >= 0) & (uv[:, 0] <= self.width - 1)
                 & (uv[:, 1] >= 0) & (uv[:, 1] <= self.height - 1))
        return uv, q[:, 2], valid


@dataclass(frozen=True)
class MarkerFrame:
    position_m: np.ndarray
    displacement_m: np.ndarray
    camera_position_m: np.ndarray
    image_uv_px: np.ndarray
    displacement_px: np.ndarray
    in_frame: np.ndarray
    flow_valid: np.ndarray


class MarkerTracker:
    """Stable marker IDs are row indices; references never change implicitly."""

    def __init__(self, binding, camera, reference_positions):
        self.binding = binding
        self.camera = camera
        self.set_reference(reference_positions)

    def set_reference(self, positions, *, camera=None):
        camera = self.camera if camera is None else camera
        self.reference_m = _readonly(self.binding.evaluate(positions))
        uv, _, valid = camera.project(self.reference_m)
        self.reference_uv_px = _readonly(uv)
        self.reference_in_frame = _readonly(valid)

    def observe(self, positions, *, camera=None):
        camera = self.camera if camera is None else camera
        q = self.binding.evaluate(positions)
        uv, _, valid = camera.project(q)
        return MarkerFrame(q, q - self.reference_m, camera.camera_points(q), uv,
                           uv - self.reference_uv_px, valid, valid & self.reference_in_frame)


class MarkerLayer:
    """Material centres AND disk rims, deforming through the same FEM mesh.

    Rims are horizontal polygons in rest coordinates. Their samples may occupy
    different elements, so disks can stretch/shear with the material rather than
    staying screen-space dots. The centres are the authoritative marker tracks.
    """

    def __init__(self, rest_positions, elements, centres_m, *, radius_m=.00011,
                 rim_samples=12):
        centres = _points(centres_m, "centres_m")
        if (not np.isfinite(radius_m) or radius_m <= 0 or rim_samples < 8
                or int(rim_samples) != rim_samples):
            raise ValueError("invalid marker radius or rim sample count")
        self.centres = MaterialBinding.bind(rest_positions, elements, centres)
        angles = np.arange(int(rim_samples)) * (2 * np.pi / rim_samples)
        offset = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles)))) * radius_m
        self.rim_shape = (len(centres), len(angles), 3)
        self.rims = MaterialBinding.bind(rest_positions, elements,
                                        (centres[:, None] + offset).reshape(-1, 3))

    def rim_positions(self, positions):
        return self.rims.evaluate(positions).reshape(self.rim_shape)
