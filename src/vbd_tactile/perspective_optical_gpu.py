"""CUDA rasterization of the same finite surface ink and pinhole optical model.

Each supersampled pixel queries a conservative screen-space triangle BVH.
Double precision barycentrics, rest-material ink and continuous marker visibility
preserve the CPU renderer's semantics. Both pixel and marker visibility queries
run on CUDA, together with marker interpolation, projection and lighting.
"""
import numpy as np
import warp as wp

from .perspective_optical import PerspectiveTactileRenderer, PerspectiveOpticalFrame
from .markers import MarkerFrame, _points
from . import optical_geometry_gpu as geometry

wp.set_module_options({"enable_backward": False})


@wp.kernel
def screen_bounds(uv: wp.array(dtype=wp.vec2d), triangles: wp.array(dtype=wp.vec3i),
                  visible_faces: wp.array(dtype=bool),
                  lower: wp.array(dtype=wp.vec3), upper: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if not visible_faces[i]:
        # Both raster and continuous visibility queries use z=0. Put a
        # back-facing coating outside that query plane, without deleting it.
        lower[i] = wp.vec3(0., 0., 10.)
        upper[i] = wp.vec3(0., 0., 11.)
        return
    ids = triangles[i]
    lo = wp.min(uv[ids[0]], wp.min(uv[ids[1]], uv[ids[2]]))
    hi = wp.max(uv[ids[0]], wp.max(uv[ids[1]], uv[ids[2]]))
    scale = wp.max(wp.max(wp.abs(lo[0]), wp.abs(lo[1])), wp.max(wp.abs(hi[0]), wp.abs(hi[1])))
    pad = wp.float64(1e-4)+wp.float64(8.*1.1920928955078125e-7)*scale
    lower[i] = wp.vec3(float(lo[0]-pad), float(lo[1]-pad), -1.)
    upper[i] = wp.vec3(float(hi[0]+pad), float(hi[1]+pad), 1.)


@wp.kernel
def raster(width: int, uv: wp.array(dtype=wp.vec2d), z: wp.array(dtype=wp.float64),
           light: wp.array(dtype=wp.vec3d), triangles: wp.array(dtype=wp.vec3i),
           rest: wp.array(dtype=wp.vec3d), bvh: wp.uint64,
           offsets: wp.array(dtype=int), candidates: wp.array(dtype=int),
           polygons: wp.array(dtype=wp.vec3d), rim_count: int, projected_ink: int,
           radiance: wp.array(dtype=wp.vec3d), covered: wp.array(dtype=int)):
    pixel = wp.tid()
    x, y = pixel % width, pixel // width
    p = wp.vec3(float(x), float(y), 0.)
    query = wp.bvh_query_aabb(bvh, p, p)
    face = int(0)
    winner = int(-1)
    nearest = wp.float64(wp.inf)
    best = wp.vec3d(0.)
    while wp.bvh_query_next(query, face):
        ids = triangles[face]
        a, b, c = uv[ids[0]], uv[ids[1]], uv[ids[2]]
        # Match the CPU rasterizer's integer bounding rectangle exactly.
        lo, hi = wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))
        if wp.float64(x) >= wp.ceil(lo[0]) and wp.float64(x) <= wp.floor(hi[0]):
            if wp.float64(y) >= wp.ceil(lo[1]) and wp.float64(y) <= wp.floor(hi[1]):
                e1, e2 = b-a, c-a
                determinant = e1[0]*e2[1]-e1[1]*e2[0]
                if wp.abs(determinant) >= wp.float64(1e-12):
                    dx, dy = wp.float64(x)-a[0], wp.float64(y)-a[1]
                    wb = (dx*e2[1]-dy*e2[0])/determinant
                    wc = (e1[0]*dy-e1[1]*dx)/determinant
                    w = wp.vec3d(wp.float64(1.)-wb-wc, wb, wc)
                    if wp.min(w[0], wp.min(w[1], w[2])) >= wp.float64(-1e-9):
                        w = wp.vec3d(w[0]/z[ids[0]], w[1]/z[ids[1]], w[2]/z[ids[2]])
                        total = w[0]+w[1]+w[2]
                        depth = wp.float64(1.)/total
                        if depth < nearest or (depth == nearest and (winner < 0 or face < winner)):
                            nearest, winner, best = depth, face, w/total
    color = wp.vec3d(0.025)
    if winner >= 0:
        ids = triangles[winner]
        color = best[0]*light[ids[0]]+best[1]*light[ids[1]]+best[2]*light[ids[2]]
        material = best[0]*rest[ids[0]]+best[1]*rest[ids[1]]+best[2]*rest[ids[2]]
        ink = bool(False)
        for item in range(offsets[winner], offsets[winner+1]):
            start = candidates[item]*rim_count
            inside = projected_ink != 0 or wp.abs(material[2]-polygons[start][2]) <= wp.float64(1e-10)
            for j in range(rim_count):
                pa = polygons[start+j]
                pb = polygons[start+(j+1) % rim_count]
                cross = (pb[0]-pa[0])*(material[1]-pa[1])-(pb[1]-pa[1])*(material[0]-pa[0])
                inside = inside and cross >= wp.float64(-1e-18)
            ink = ink or inside
        if ink:
            color *= wp.float64(.04)
    radiance[pixel] = color
    covered[pixel] = int(winner >= 0)


@wp.kernel
def marker_visibility(scale: int, samples: wp.array(dtype=wp.vec3d),
                      uv: wp.array(dtype=wp.vec2d), z: wp.array(dtype=wp.float64),
                      triangles: wp.array(dtype=wp.vec3i), bvh: wp.uint64,
                      visible: wp.array(dtype=bool)):
    i = wp.tid()
    sample = samples[i]
    s = wp.float64(scale)
    # Query the raster BVH, but use continuous base-camera coordinates and
    # the CPU reference's tolerances for the actual depth test. Never round.
    p = wp.vec3(float((sample[0]+wp.float64(.5))*s-wp.float64(.5)),
                float((sample[1]+wp.float64(.5))*s-wp.float64(.5)), 0.)
    query = wp.bvh_query_aabb(bvh, p, p)
    face = int(0)
    nearest = wp.float64(wp.inf)
    while wp.bvh_query_next(query, face):
        ids = triangles[face]
        a = (uv[ids[0]]+wp.vec2d(.5))/s-wp.vec2d(.5)
        b = (uv[ids[1]]+wp.vec2d(.5))/s-wp.vec2d(.5)
        c = (uv[ids[2]]+wp.vec2d(.5))/s-wp.vec2d(.5)
        lo, hi = wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))
        eps = wp.float64(1e-9)
        if sample[0] >= lo[0]-eps and sample[0] <= hi[0]+eps and sample[1] >= lo[1]-eps and sample[1] <= hi[1]+eps:
            e1, e2 = b-a, c-a
            det = e1[0]*e2[1]-e1[1]*e2[0]
            if wp.abs(det) > wp.float64(1e-12):
                dx, dy = sample[0]-a[0], sample[1]-a[1]
                wb = (dx*e2[1]-dy*e2[0])/det
                wc = (e1[0]*dy-e1[1]*dx)/det
                wa = wp.float64(1.)-wb-wc
                if wp.min(wa, wp.min(wb, wc)) >= -eps:
                    depth = wp.float64(1.)/(wa/z[ids[0]]+wb/z[ids[1]]+wc/z[ids[2]])
                    nearest = wp.min(nearest, depth)
    visible[i] = wp.isfinite(nearest) and sample[2] <= nearest+wp.float64(1e-10)


@wp.kernel
def downsample(width: int, scale: int, radiance: wp.array(dtype=wp.vec3d),
               covered: wp.array(dtype=int), rgb: wp.array3d(dtype=wp.uint8),
               coverage: wp.array2d(dtype=wp.float64)):
    y, x = wp.tid()
    color = wp.vec3d(0.)
    hits = int(0)
    for dy in range(scale):
        for dx in range(scale):
            p = (y*scale+dy)*width*scale+x*scale+dx
            color += radiance[p]
            hits += covered[p]
    for channel in range(3):
        value = wp.clamp(color[channel]/wp.float64(scale*scale), wp.float64(0.), wp.float64(1.))
        rgb[y, x, channel] = wp.uint8(wp.rint(wp.pow(value, wp.float64(1./2.2))*wp.float64(255.)))
    coverage[y, x] = wp.float64(hits)/wp.float64(scale*scale)


class CUDAPerspectiveTactileRenderer(PerspectiveTactileRenderer):
    backend = "CUDA geometry, marker tracking and raster"

    def __init__(self, *args, device="cuda:0", **kwargs):
        super().__init__(*args, **kwargs)
        self.device = wp.get_device(device)
        if not self.device.is_cuda or not self.surface_ink:
            raise ValueError("CUDA optical renderer requires CUDA and a surface marker film")
        device = self.device
        self._uv = wp.empty(len(self.rest), dtype=wp.vec2d, device=device)
        self._z = wp.empty(len(self.rest), dtype=wp.float64, device=device)
        self._light = wp.empty(len(self.rest), dtype=wp.vec3d, device=device)
        self._triangles = wp.array(self.triangles, dtype=wp.vec3i, device=device)
        self._visible_faces_device = wp.empty(len(self.triangles), dtype=bool, device=device)
        self._rest = wp.array(self.rest, dtype=wp.vec3d, device=device)
        sizes = np.array([len(c) for c in self.ink_candidates], dtype=np.int32)
        self._offsets = wp.array(np.r_[0, np.cumsum(sizes)].astype(np.int32), dtype=int, device=device)
        self._candidates = wp.array(np.array([i for c in self.ink_candidates for i in c], dtype=np.int32), dtype=int, device=device)
        self._polygons = wp.array(self.ink_polygons.reshape(-1, 3), dtype=wp.vec3d, device=device)
        self._lower = wp.zeros(len(self.triangles), dtype=wp.vec3, device=device)
        self._upper = wp.empty_like(self._lower)
        self._upper.zero_()
        self._bvh = wp.Bvh(self._lower, self._upper)
        pixels = self.raster_camera.width*self.raster_camera.height
        self._radiance = wp.empty(pixels, dtype=wp.vec3d, device=device)
        self._covered = wp.empty(pixels, dtype=int, device=device)
        self._rgb = wp.empty((self.camera.height, self.camera.width, 3), dtype=wp.uint8, device=device)
        self._coverage = wp.empty((self.camera.height, self.camera.width), dtype=wp.float64, device=device)
        self._marker_samples = wp.empty(len(self.layer.centres.vertex_ids), dtype=wp.vec3d, device=device)
        self._visible = wp.empty(len(self._marker_samples), dtype=bool, device=device)
        self._local = wp.empty_like(self._rest)
        self._camera_points = wp.empty_like(self._rest)
        self._face_normals = wp.empty(len(self.triangles), dtype=wp.vec3d, device=device)
        self._status = wp.zeros(1, dtype=int, device=device)
        self._marker_data = wp.zeros((len(self._marker_samples)+1, 17), dtype=wp.float64, device=device)
        self._marker_ids = wp.array(self.layer.centres.vertex_ids.astype(np.int32), dtype=wp.vec3i, device=device)
        self._marker_weights = wp.array(self.layer.centres.weights, dtype=wp.vec3d, device=device)
        surface = np.zeros(len(self.rest), dtype=np.int32)
        surface[np.unique(self.triangles)] = 1
        self._surface = wp.array(surface, device=device)
        adjacency = [[] for _ in self.rest]
        for corner in range(3):
            for face, vertex in enumerate(self.triangles[:, corner]):
                adjacency[vertex].append(face)
        self._face_offsets = wp.array(np.r_[0, np.cumsum([len(a) for a in adjacency])].astype(np.int32), device=device)
        self._face_adjacency = wp.array(np.array([i for a in adjacency for i in a], dtype=np.int32), device=device)
        rest_tri = self.rest[self.triangles]
        rest_normal = np.cross(rest_tri[:, 1]-rest_tri[:, 0], rest_tri[:, 2]-rest_tri[:, 0])
        sign = np.where(rest_normal[:, 2] < 0, 1., -1.)
        self._sign = wp.array(sign, dtype=wp.float64, device=device)
        self._reference = wp.array(self.tracker.reference_m, dtype=wp.vec3d, device=device)
        self._reference_uv = wp.array(self.tracker.reference_uv_px, dtype=wp.vec2d, device=device)
        self._reference_in_frame = wp.array(self.tracker.reference_in_frame, dtype=bool, device=device)
        self._reference_visible = wp.array(self.reference_visible, dtype=bool, device=device)
        self._host_rgb = wp.empty_like(self._rgb, device="cpu", pinned=True)
        self._host_coverage = wp.empty_like(self._coverage, device="cpu", pinned=True)
        self._host_markers = wp.empty_like(self._marker_data, device="cpu", pinned=True)
        self._graph = None
        self._graph_key = None
        self._pending = False
        self._copied = False
        self._copy_done = wp.Event(self.device)
        self._input_owner = None

    def set_reference(self, positions):
        if getattr(self, "_pending", False):
            raise RuntimeError("collect the pending optical frame before resetting the reference")
        super().set_reference(positions)
        if hasattr(self, "_reference"):
            self._reference.assign(self.tracker.reference_m)
            self._reference_uv.assign(self.tracker.reference_uv_px)
            self._reference_in_frame.assign(self.tracker.reference_in_frame)
            self._reference_visible.assign(self.reference_visible)

    def render(self, positions):
        """Compatibility API; NumPy inputs upload once, device inputs stay resident."""
        self.enqueue(positions)
        self.copy_outputs()
        wp.synchronize_device(self.device)
        return self.collect()

    def enqueue(self, positions, *, offset=0, pose=None):
        """Submit without downloads; call for every camera before copying outputs.

        ``pose`` is a rigid transform from material to world coordinates. Offset
        selects this camera's contiguous vertices from a shared world array.
        A pending frame owns the staging buffers until ``collect`` completes.
        """
        if self._pending:
            raise RuntimeError("collect the pending optical frame before submitting another")
        self._submit_stream = wp.get_stream(self.device)
        if isinstance(positions, wp.array):
            if (positions.device != self.device or positions.ndim != 1
                    or positions.dtype not in (wp.vec3, wp.vec3d)
                    or not isinstance(offset, (int, np.integer)) or offset < 0
                    or offset+len(self.rest) > len(positions)):
                raise ValueError("optical positions must be matching device vec3/vec3d vertices with valid bounds")
            matrix = np.eye(4) if pose is None else np.asarray(pose, dtype=float)
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
                    or not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T,
                                       np.eye(3), atol=1e-9, rtol=0)
                    or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-9, rtol=0)):
                # local_positions uses R.T as the inverse. A scale, shear or
                # reflection would silently change marker motion and lighting.
                raise ValueError("optical pose must be a finite proper rigid 4x4 transform")
            self._input_owner = positions  # Retain asynchronous input until copies complete.
            wp.launch(geometry.local_positions, len(self.rest),
                      [positions, int(offset), wp.mat44d(matrix), self._local], device=self.device)
        else:
            if offset != 0 or pose is not None:
                raise ValueError("offset/pose require device world positions")
            q = _points(positions, "positions")
            if q.shape != self.rest.shape:
                raise ValueError("positions must retain the bound mesh's vertex ordering and count")
            self._local.assign(q)
        key = (self.camera, self.raster_camera, tuple(self.lights.ravel()), tuple(self.flat_light), self.outward_winding)
        # Dataclass camera equality includes NumPy arrays: use identity for the
        # immutable camera objects, values for mutable lighting configuration.
        key = (id(key[0]), id(key[1]), *key[2:])
        if self._graph is None or key != self._graph_key:
            self._record_frame()  # Load kernels before capture; no host readback.
            if self._graph is None:
                # Construct spatial topology from actual projected triangles,
                # not the constructor's all-zero placeholder bounds.
                self._bvh.rebuild()
            with wp.ScopedCapture(device=self.device) as capture:
                self._record_frame()
            self._graph, self._graph_key = capture.graph, key
        wp.capture_launch(self._graph)
        self._pending, self._copied = True, False

    def _record_frame(self):
        c, r = self.camera, self.raster_camera
        rotation, translation = wp.mat33d(c.rotation), wp.vec3d(c.translation_m)
        self._status.zero_()
        wp.launch(geometry.project_vertices, len(self.rest),
                  [self._local, rotation, translation, wp.vec4d(r.fx, r.fy, r.cx, r.cy), c.near_m,
                   self._surface, self._camera_points, self._uv, self._z, self._status], device=self.device)
        wp.launch(geometry.face_geometry, len(self.triangles),
                  [self._local, self._camera_points, self._triangles, self._sign, self.outward_winding,
                   self._face_normals, self._visible_faces_device], device=self.device)
        wp.launch(geometry.vertex_lighting, len(self.rest),
                  [self._face_normals, self._face_offsets, self._face_adjacency, rotation,
                   wp.mat33d(self.lights), wp.vec3d(self.flat_light), self._light, self._status], device=self.device)
        wp.launch(geometry.marker_geometry, len(self._marker_samples),
                  [self._local, self._marker_ids, self._marker_weights, rotation, translation,
                   wp.vec4d(c.fx, c.fy, c.cx, c.cy), c.near_m, c.width, c.height,
                   self._reference, self._reference_uv, self._reference_in_frame,
                   self._marker_samples, self._marker_data], device=self.device)
        wp.launch(screen_bounds, len(self.triangles),
                  [self._uv, self._triangles, self._visible_faces_device, self._lower, self._upper], device=self.device)
        self._bvh.refit()
        wp.launch(raster, len(self._radiance), [self.raster_camera.width,
            self._uv, self._z, self._light, self._triangles, self._rest, self._bvh.id,
            self._offsets, self._candidates, self._polygons, self.layer.rim_shape[1], int(self.projected_surface_ink),
            self._radiance, self._covered], device=self.device)
        wp.launch(downsample, (self.camera.height, self.camera.width),
                  [self.camera.width, self.scale, self._radiance, self._covered, self._rgb, self._coverage], device=self.device)
        wp.launch(marker_visibility, len(self._marker_samples), [self.scale, self._marker_samples,
            self._uv, self._z, self._triangles, self._bvh.id, self._visible], device=self.device)
        wp.launch(geometry.finish_markers, max(1, len(self._visible)),
                  [self._visible, self._reference_visible, self._status, self._marker_data], device=self.device)

    def copy_outputs(self, *, coverage=True):
        """Queue copies to persistent pinned buffers, without synchronizing."""
        if not self._pending or self._copied:
            raise RuntimeError("outputs require one pending, uncopied optical frame")
        with wp.ScopedStream(self._submit_stream):
            wp.copy(self._host_rgb, self._rgb)
            wp.copy(self._host_markers, self._marker_data)
            if coverage:
                wp.copy(self._host_coverage, self._coverage)
            wp.record_event(self._copy_done)
        self._include_coverage, self._copied = bool(coverage), True

    def collect(self):
        """Collect after the caller synchronizes the device (once for all cameras)."""
        if not self._pending or not self._copied:
            raise RuntimeError("enqueue and copy outputs before collecting an optical frame")
        # An event guards standalone callers too; after the batch synchronize
        # this is an already-complete wait, not another device-wide barrier.
        # Host arrays must never be exposed before the asynchronous copies end.
        wp.synchronize_event(self._copy_done)
        self._pending, self._copied = False, False
        self._input_owner = None
        packet = self._host_markers.numpy().copy()
        code = int(packet[0, 0])
        if code:
            raise ValueError({1: "coating crosses camera near plane; clipping is not supported",
                              2: "degenerate coating normals", 3: "nonfinite optical geometry"}[code])
        a = packet[1:]
        markers = MarkerFrame(a[:, :3], a[:, 3:6], a[:, 6:9], a[:, 9:11], a[:, 11:13],
                              a[:, 13].astype(bool), a[:, 14].astype(bool))
        coverage = self._host_coverage.numpy().copy() if self._include_coverage else None
        return PerspectiveOpticalFrame(self._host_rgb.numpy().copy(), markers, coverage,
                                       a[:, 15].astype(bool), a[:, 16].astype(bool))

    def discard_pending(self):
        """Release staging after an aborted multi-camera frame."""
        if self._pending:
            wp.synchronize_stream(self._submit_stream)
            self._pending, self._copied = False, False
            self._input_owner = None
