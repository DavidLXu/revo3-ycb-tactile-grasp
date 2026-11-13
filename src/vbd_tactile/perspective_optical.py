"""Geometry-driven pinhole RGB with FEM-bound finite material markers.

This diagnostic renderer views a translucent gel's coating from its interior.
It is not calibrated optics: no refraction, lens distortion, multiple scattering
or housing/marker-to-marker occlusion. Coating occlusion is depth tested.
Physics is not advanced here. The mesh,
marker centres, and deforming marker rims share one camera projection.
"""
from dataclasses import dataclass

import numpy as np

from .markers import MarkerFrame, MarkerTracker, PinholeCamera, _elements, _points
from .surface_geometry import projected_surface_depth


@dataclass(frozen=True)
class PerspectiveOpticalFrame:
    rgb: np.ndarray
    markers: MarkerFrame
    coating_coverage: np.ndarray | None
    visible: np.ndarray
    visible_flow_valid: np.ndarray


def _raster_triangle(uv, width, height):
    """Pixel-centre samples with screen-space barycentric coordinates."""
    lo = np.maximum(np.ceil(uv.min(axis=0)).astype(int), 0)
    hi = np.minimum(np.floor(uv.max(axis=0)).astype(int), [width - 1, height - 1])
    if np.any(hi < lo):
        return None
    e1, e2 = uv[1] - uv[0], uv[2] - uv[0]
    determinant = e1[0] * e2[1] - e1[1] * e2[0]
    if abs(determinant) < 1e-12:
        return None
    x, y = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    dx, dy = x - uv[0, 0], y - uv[0, 1]
    b = (dx * e2[1] - dy * e2[0]) / determinant
    c = (e1[0] * dy - e1[1] * dx) / determinant
    weights = np.stack((1 - b - c, b, c), axis=-1)
    mask = (weights >= -1e-9).all(axis=-1)
    return y[mask], x[mask], weights[mask]


class PerspectiveTactileRenderer:
    """Render a specified coating mesh and a bound marker layer using one camera.

    Shading uses the inward side of a coating whose rest outward normal is +z.
    Three illustrative coloured directional lights are on the camera side,
    expressed in camera coordinates. `coating_coverage` is geometric
    coverage, while marker `in_frame`/`flow_valid` only describe the camera FOV.
    Supersampling antialiases finite marker disks; it never magnifies their flow.
    With ``outward_winding=True`` the coating must be consistently outward
    oriented: shade its inward normals and render only its internal side. This
    supports wraparound skins without forcing every normal toward camera -z.
    Camera and supersampling are fixed at construction: changing calibration
    requires a new renderer with matching marker references and raster buffers.
    """

    def __init__(self, rest_positions, coating_triangles, marker_layer, camera,
                 *, supersample=2, outward_winding=False):
        self.rest = _points(rest_positions, "rest_positions").copy()
        self.triangles = _elements(coating_triangles, 3, len(self.rest)).copy()
        if supersample not in (1, 2, 3, 4):
            raise ValueError("supersample must be an integer from 1 to 4")
        self._scale = int(supersample)
        self.layer = marker_layer
        self.outward_winding = bool(outward_winding)
        self._camera = camera
        self.tracker = MarkerTracker(marker_layer.centres, camera, self.rest)
        # Half-pixel offset preserves the original integer pixel centres after
        # area downsampling: u_hi = s * (u + 0.5) - 0.5.
        s = self.scale
        self._raster_camera = PinholeCamera(
            camera.width * s, camera.height * s, camera.fx * s, camera.fy * s,
            s * (camera.cx + .5) - .5, s * (camera.cy + .5) - .5,
            camera.rotation, camera.translation_m, camera.near_m)
        angles = np.deg2rad([0, 120, 240])
        # Direction from the coating towards internal LEDs: camera -z. The
        # lights follow the camera/sensor, rather than remaining world-fixed.
        lights = np.column_stack((np.cos(angles), np.sin(angles), np.full(3, -.38)))
        self.lights = lights / np.linalg.norm(lights, axis=1, keepdims=True)
        self.flat_light = .20 - .68 * self.lights[:, 2]
        self.surface_ink = marker_layer.centres.vertex_ids.shape[1] == 3
        self.projected_surface_ink = bool(getattr(marker_layer, "projected_surface_ink", False))
        if self.surface_ink:
            self._prepare_surface_ink()
        self.reference_visible = self.visibility(self.rest)

    @property
    def camera(self):
        """Bound observation camera; construct a new renderer to recalibrate."""
        return self._camera

    @property
    def raster_camera(self):
        """Supersampled camera derived from the bound observation camera."""
        return self._raster_camera

    @property
    def scale(self):
        """Fixed supersampling factor, shared by projection and raster storage."""
        return self._scale

    def _prepare_surface_ink(self):
        # Evaluate ink in rest-material coordinates of the winning coating
        # fragment. This splits a disk exactly at FEM triangle edges, so a
        # centre/rim fan cannot bridge across bends or leak onto a front face.
        polygons = self.layer.rims.rest_m.reshape(self.layer.rim_shape)
        self.ink_polygons = polygons
        tri = self.rest[self.triangles]
        lower, upper = tri.min(axis=1), tri.max(axis=1)
        parent = np.arange(len(self.rest))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for ids in self.triangles:
            for i in ids[1:]:
                parent[root(int(i))] = root(int(ids[0]))
        components = np.array([root(int(ids[0])) for ids in self.triangles])
        faces = {tuple(sorted(ids)) for ids in self.triangles}
        self.ink_candidates = [[] for _ in self.triangles]
        # A curved coating can wrap around the finger. The same UV footprint
        # can then cover several sheets: ink belongs only to the connected
        # patch containing its material centre, not every sheet below it.
        adjacency = [[] for _ in self.triangles]
        edges = {}
        for face, ids in enumerate(self.triangles):
            for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
                edge = tuple(sorted((int(a), int(b))))
                for other in edges.get(edge, []):
                    adjacency[face].append(other)
                    adjacency[other].append(face)
                edges.setdefault(edge, []).append(face)
        face_lookup = {tuple(sorted(ids)): i for i, ids in enumerate(self.triangles)}
        for index, (polygon, ids) in enumerate(zip(polygons, self.layer.centres.vertex_ids)):
            if tuple(sorted(ids)) not in faces:
                raise ValueError("surface markers must bind to the rendered coating")
            dimensions = 2 if self.projected_surface_ink else 3
            candidates = ((upper[:, :dimensions] >= polygon.min(axis=0)[:dimensions] - 1e-10)
                          & (lower[:, :dimensions] <= polygon.max(axis=0)[:dimensions] + 1e-10)).all(axis=1)
            candidates &= components == root(int(ids[0]))
            if self.projected_surface_ink:
                seed = face_lookup[tuple(sorted(ids))]
                connected = np.zeros(len(self.triangles), dtype=bool)
                stack = [seed]
                while stack:
                    face = stack.pop()
                    if connected[face] or not candidates[face]:
                        continue
                    connected[face] = True
                    stack.extend(adjacency[face])
                candidates &= connected
            for face in np.flatnonzero(candidates):
                self.ink_candidates[face].append(index)

    def _surface_ink_mask(self, face, weights):
        points = weights @ self.rest[self.triangles[face]]
        ink = np.zeros(len(points), dtype=bool)
        for index in self.ink_candidates[face]:
            polygon = self.ink_polygons[index]
            edge = np.roll(polygon[:, :2], -1, axis=0) - polygon[:, :2]
            delta = points[:, None, :2] - polygon[None, :, :2]
            cross = edge[None, :, 0] * delta[:, :, 1] - edge[None, :, 1] * delta[:, :, 0]
            # Rims are counterclockwise horizontal polygons in rest space.
            in_plane = True if self.projected_surface_ink else (abs(points[:, 2] - polygon[0, 2]) <= 1e-10)
            ink |= (cross >= -1e-18).all(axis=1) & in_plane
        return ink

    def visibility(self, positions, markers=None):
        """Centre visibility against the coating, independently of FOV masks.

        A partly covered disk can have an occluded centre. Finite ink is tested
        per fragment when rendering; this mask is specifically for flow tracks.
        Embedded markers remain an illustrative transparent-volume model.
        """
        q = _points(positions, "positions")
        markers = self.tracker.observe(q) if markers is None else markers
        uv, z, _ = self.camera.project(q)
        if np.any(z[np.unique(self.triangles)] <= self.camera.near_m):
            raise ValueError("coating crosses camera near plane; clipping is not supported")
        triangles = self.triangles[self._visible_faces(q)]
        if not len(triangles):
            return np.zeros(len(markers.in_frame), dtype=bool)
        nearest = projected_surface_depth(uv, z, triangles, markers.image_uv_px)
        # Numerical tolerance, 0.1 nm; not an adjustable optical thickness.
        return (markers.in_frame & np.isfinite(nearest)
                & (markers.camera_position_m[:, 2] <= nearest + 1e-10))

    def set_reference(self, positions):
        # Visibility can reject an otherwise valid marker binding (for example
        # a coating vertex behind the camera). Validate it before changing any
        # reference: CUDA uploads happen only after this method succeeds, and
        # callers can also read the host tracker's reference UV coordinates.
        visible = self.visibility(positions)
        self.tracker.set_reference(positions)
        self.reference_visible = visible

    backend = "CPU / NumPy"

    def _visible_faces(self, positions):
        """For an oriented skin the internal camera sees its inward side only."""
        if not self.outward_winding:
            return np.ones(len(self.triangles), dtype=bool)
        tri = self.camera.camera_points(positions)[self.triangles]
        outward = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        return np.einsum("ij,ij->i", outward, tri.mean(axis=1)) > 0

    def _frame_geometry(self, positions):
        q = _points(positions, "positions")
        markers = self.tracker.observe(q)
        camera = self.raster_camera
        uv, z, _ = camera.project(q)
        if np.any(z[np.unique(self.triangles)] <= camera.near_m):
            raise ValueError("coating crosses camera near plane; clipping is not supported")
        tri = q[self.triangles]
        face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        # The optical side is the BACK of the same mechanical contact surface.
        # Orient inward (-z at rest), independent of mesh triangle winding.
        rest_tri = self.rest[self.triangles]
        rest_face = np.cross(rest_tri[:, 1] - rest_tri[:, 0], rest_tri[:, 2] - rest_tri[:, 0])
        if self.outward_winding:
            # A consistently oriented wraparound skin has genuine side/back
            # normals. Flipping each face toward -z creates false colour seams.
            face *= -1
        else:
            face *= np.where(rest_face[:, 2] < 0, 1, -1)[:, None]
        normals = np.zeros_like(q)
        for i in range(3):
            np.add.at(normals, self.triangles[:, i], face)
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths[np.unique(self.triangles)] < 1e-16):
            raise ValueError("degenerate coating normals")
        normals /= np.maximum(lengths[:, None], 1e-30)
        camera_normals = normals @ self.camera.rotation.T
        light = (.20 + .68 * np.maximum(camera_normals @ self.lights.T, 0)) / self.flat_light * .34
        return q, markers, uv, z, light

    def render(self, positions):
        q, markers, uv, z, light = self._frame_geometry(positions)
        camera = self.raster_camera
        image = np.full((camera.height, camera.width, 3), .025)
        zbuffer = np.full(image.shape[:2], np.inf)
        visible_faces = self._visible_faces(q)
        for face_index, ids in enumerate(self.triangles):
            if not visible_faces[face_index]:
                continue
            raster = _raster_triangle(uv[ids], camera.width, camera.height)
            if raster is None:
                continue
            y, x, weights = raster
            weights = weights / z[ids]
            inv_z = weights.sum(axis=1)
            depth = 1 / inv_z
            keep = depth < zbuffer[y, x]
            y, x = y[keep], x[keep]
            zbuffer[y, x] = depth[keep]
            # Perspective-correct interpolation, not affine screen interpolation.
            image[y, x] = (weights[keep] @ light[ids]) / inv_z[keep, None]
            if self.surface_ink and self.ink_candidates[face_index]:
                ink = self._surface_ink_mask(face_index, weights[keep] / inv_z[keep, None])
                image[y[ink], x[ink]] *= .04
        coverage = np.isfinite(zbuffer)
        if not self.surface_ink:
            self._render_embedded_ink(q, markers, image, zbuffer, coverage)
        s = self.scale
        # Average linear radiance before gamma conversion.
        image = image.reshape(self.camera.height, s, self.camera.width, s, 3).mean(axis=(1, 3))
        coverage = coverage.reshape(self.camera.height, s, self.camera.width, s).mean(axis=(1, 3))
        rgb = np.rint(np.clip(image, 0, 1)**(1 / 2.2) * 255).astype(np.uint8)
        visible = self.visibility(q, markers)
        return PerspectiveOpticalFrame(rgb, markers, coverage, visible,
                                       markers.flow_valid & visible & self.reference_visible)

    def _render_embedded_ink(self, q, markers, image, zbuffer, coverage):
        camera = self.raster_camera
        rims = self.layer.rim_positions(q)
        rim_uv, rim_z, _ = camera.project(rims.reshape(-1, 3))
        rim_uv = rim_uv.reshape(*rims.shape[:2], 2)
        rim_z = rim_z.reshape(rims.shape[:2])
        centre_uv, centre_z, _ = camera.project(markers.position_m)
        ink = np.zeros(coverage.shape, dtype=bool)
        for centre, cz, rim, rz in zip(centre_uv, centre_z, rim_uv, rim_z):
            if cz <= camera.near_m or np.any(rz <= camera.near_m):
                continue
            for j in range(len(rim)):
                raster = _raster_triangle(np.array([centre, rim[j], rim[(j + 1) % len(rim)]]),
                                          camera.width, camera.height)
                if raster is not None:
                    y, x, w = raster
                    depth = 1 / (w / [cz, rz[j], rz[(j + 1) % len(rim)]]).sum(axis=1)
                    keep = depth <= zbuffer[y, x] + 1e-10
                    ink[y[keep], x[keep]] = True
        image[ink & coverage] *= .04
