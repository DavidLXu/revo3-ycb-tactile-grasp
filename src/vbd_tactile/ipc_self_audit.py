"""GPU validation of initial tet volumes and affine surface motion.

This checks nonincident point/triangle and edge/edge features, including features
of triangles sharing a vertex. It is an audit, not a contact-force solver.
TetVolumeAudit checks initial volume overlap, including containment. Motion
failure means contact or an uncertified path, not a proven overlap. Neither
check supplies collision response forces.
"""

import numpy as np
from typing import Any
from .ipc_precision import position_overloads
import warp as wp
from .ipc_precision import position_kernel,device_scalar_kernel,scalar_value

from .ipc import positive_scalar
from .ipc_primitives_warp import _delta, _finite, _weights

wp.set_module_options({"enable_backward": False})


def collision_groups(groups, particle_count):
    if groups is None:
        return np.full(particle_count, -1, dtype=np.int32)
    groups = np.asarray(groups)
    if (groups.shape != (particle_count,) or not np.issubdtype(groups.dtype, np.integer)
            or np.any(groups < -1) or np.any(groups >= np.iinfo(np.int32).max)):
        raise ValueError("rigid collision groups must be per-node integers >= -1")
    return groups.astype(np.int32)


@wp.func
def same_rigid(groups: wp.array(dtype=int), a: int, b: int, c: int, d: int):
    return groups[a] >= 0 and groups[a] == groups[b] and groups[a] == groups[c] and groups[a] == groups[d]


@wp.func
def _projected_gap(
    n: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, kind: int
):
    gap = wp.dot(n, a) - wp.max(wp.dot(n, b), wp.max(wp.dot(n, c), wp.dot(n, d)))
    if kind == 1:
        gap = wp.min(wp.dot(n, a), wp.dot(n, b)) - wp.max(wp.dot(n, c), wp.dot(n, d))
    elif kind == 2:
        # Convergent correction: point c against segment (a,b); d belongs
        # only to the mollifier, never to this distance primitive.
        gap = wp.min(wp.dot(n, a), wp.dot(n, b)) - wp.dot(n, c)
    return gap


@wp.func
def _feature_motion_prefix(
    a: wp.vec3d,
    b: wp.vec3d,
    c: wp.vec3d,
    d: wp.vec3d,
    aa: wp.vec3d,
    bb: wp.vec3d,
    cc: wp.vec3d,
    dd: wp.vec3d,
    kind: int,
    min_gap: wp.float64,
):
    va, vb, vc, vd = aa - a, bb - b, cc - c, dd - d
    scale = wp.max(
        wp.max(wp.length(a), wp.length(b)), wp.max(wp.length(c), wp.length(d))
    )
    scale += wp.max(
        wp.max(wp.length(va), wp.length(vb)), wp.max(wp.length(vc), wp.length(vd))
    )
    mean = (va + vb + vc + vd) / wp.float64(4.0)
    va, vb, vc, vd = va - mean, vb - mean, vc - mean, vd - mean
    speed = wp.max(wp.length(va), wp.length(vb)) + wp.max(wp.length(vc), wp.length(vd))
    if kind == 0:
        speed = wp.length(va) + wp.max(
            wp.length(vb), wp.max(wp.length(vc), wp.length(vd))
        )
    # Work in local coordinates. Support planes give distance lower bounds even
    # when the closest-feature routine selects an imperfect separating normal.
    b, c, d = b - a, c - a, d - a
    a = wp.vec3d(0.0)
    threshold = min_gap + wp.float64(128.0 * 2.220446049250313e-16) * scale
    t = wp.float64(0.0)
    complete = bool(False)  # noqa: UP018 -- Warp dynamic-loop variable
    for attempt in range(128):
        at, bt, ct, dt = a + t * va, b + t * vb, c + t * vc, d + t * vd
        delta = _delta(_weights(at, bt, ct, dt, kind), at, bt, ct, dt)
        length = wp.length(delta)
        if not wp.isfinite(length) or length <= wp.float64(0.0):
            break
        n = delta / length
        gap = _projected_gap(n, at, bt, ct, dt, kind)
        if not wp.isfinite(gap) or not wp.isfinite(speed) or gap <= threshold:
            break
        if gap - speed * (wp.float64(1.0) - t) > threshold:
            complete = True
            break
        advance = wp.float64(0.9) * (gap - threshold) / speed
        # A fixed separating plane yields a tighter bound for tangential
        # sliding than total relative speed. Projected separation is the min
        # of affine vertex-pair gaps, hence concave: its endpoint chord is a
        # lower bound throughout the remaining interval. This also avoids
        # exhausting 128 iterations on safe, close parallel motion.
        end_gap = _projected_gap(n, a + va, b + vb, c + vc, d + vd, kind)
        if not wp.isfinite(end_gap):
            break
        if end_gap > threshold:
            complete = True
            break
        plane_advance = (
            wp.float64(0.9)
            * (gap - threshold)
            * (wp.float64(1.0) - t)
            / (gap - end_gap)
        )
        advance = wp.max(advance, plane_advance)
        if advance <= wp.float64(2.220446049250313e-16):
            break
        t += advance
    if complete:
        return wp.float64(1.0)
    # Every advancement was bounded below by a supporting plane or the
    # relative-speed Lipschitz bound. Thus [0,t] is certified even if the
    # remaining interval exhausted the iteration budget.
    return t


@wp.func
def certify_feature_motion(
    a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d,
    aa: wp.vec3d, bb: wp.vec3d, cc: wp.vec3d, dd: wp.vec3d,
    kind: int, min_gap: wp.float64,
):
    certified = bool(False)
    for contraction in range(32):
        prefix = _feature_motion_prefix(a, b, c, d, aa, bb, cc, dd, kind, min_gap)
        if prefix >= wp.float64(1.0):
            certified = True
            break
        # The same affine path has the same clearance in either direction.
        # Only their certified intervals may be removed from the query.
        suffix = _feature_motion_prefix(aa, bb, cc, dd, a, b, c, d, kind, min_gap)
        if prefix+suffix >= wp.float64(1.0):
            certified = True
            break
        if prefix+suffix <= wp.float64(1e-12):
            break
        # Keep testing the uncovered middle. This handles a rotating closest
        # plane without discarding a gap in the certificate or forcing the
        # physical integrator to subdivide a provably safe affine trajectory.
        va, vb, vc, vd = aa-a, bb-b, cc-c, dd-d
        aa, bb, cc, dd = aa-suffix*va, bb-suffix*vb, cc-suffix*vc, dd-suffix*vd
        a, b, c, d = a+prefix*va, b+prefix*vb, c+prefix*vc, d+prefix*vd
    return certified


@wp.func
def _certify(
    old: wp.array(dtype=Any),
    target: wp.array(dtype=Any),
    ids: wp.vec4i,
    kind: int,
    min_gap: wp.float64,
):
    return certify_feature_motion(
        wp.vec3d(old[ids[0]]),
        wp.vec3d(old[ids[1]]),
        wp.vec3d(old[ids[2]]),
        wp.vec3d(old[ids[3]]),
        wp.vec3d(target[ids[0]]),
        wp.vec3d(target[ids[1]]),
        wp.vec3d(target[ids[2]]),
        wp.vec3d(target[ids[3]]),
        kind,
        min_gap,
    )


@position_overloads("old", "target", scalars={"min_gap":wp.float32})
@wp.kernel
def _bounds(
    old: wp.array(dtype=Any),
    target: wp.array(dtype=Any),
    features: wp.array2d(dtype=int),
    width: int,
    min_gap: Any,
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    invalid: wp.array(dtype=int),
):
    i = wp.tid()
    lo, hi = wp.vec3(wp.inf), wp.vec3(-wp.inf)
    valid = bool(True)  # noqa: UP018 -- Warp dynamic-loop variable
    for k in range(width):
        a, b = wp.vec3(old[features[i, k]]), wp.vec3(target[features[i, k]])
        if not _finite(wp.vec3d(a)) or not _finite(wp.vec3d(b)):
            valid = False
        lo, hi = wp.min(lo, wp.min(a, b)), wp.max(hi, wp.max(a, b))
    if valid:
        # Outward float32 padding prevents BVH rounding from discarding a pair.
        scale = wp.max(
            wp.max(wp.abs(lo[0]), wp.abs(hi[0])),
            wp.max(
                wp.max(wp.abs(lo[1]), wp.abs(hi[1])),
                wp.max(wp.abs(lo[2]), wp.abs(hi[2])),
            ),
        )
        gap=float(scalar_value(min_gap))
        if not wp.isfinite(gap) or gap < 0.:
            gap=0.
            valid=False
        pad = gap + 8.0 * 1.1920928955078125e-7 * scale
        lower[i] = lo - wp.vec3(pad)
        upper[i] = hi + wp.vec3(pad)
        if not _finite(wp.vec3d(lower[i])) or not _finite(wp.vec3d(upper[i])):
            valid = False
    if not valid:
        wp.atomic_add(invalid, 0, 1)
        lower[i] = wp.vec3(0.0)
        upper[i] = wp.vec3(0.0)


@position_overloads("old", "target", scalars={"min_gap":wp.float64})
@wp.kernel
def _points_against_faces(
    old: wp.array(dtype=Any),
    target: wp.array(dtype=Any),
    invalid: wp.array(dtype=int),
    groups: wp.array(dtype=int),
    vertices: wp.array2d(dtype=int),
    faces: wp.array2d(dtype=int),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    bvh: wp.uint64,
    min_gap: Any,
    stats: wp.array(dtype=int),
    witness: wp.array2d(dtype=int),
    enabled: wp.array(dtype=int),
):
    if invalid[0] or not enabled[0]:
        return
    i = wp.tid()
    vertex = vertices[i, 0]
    query = wp.bvh_query_aabb(bvh, lower[i], upper[i])
    j = int(0)  # noqa: UP018, RUF046 -- Warp BVH iterator output
    while wp.bvh_query_next(query, j):
        a, b, c = faces[j, 0], faces[j, 1], faces[j, 2]
        if vertex != a and vertex != b and vertex != c and not same_rigid(groups, vertex, a, b, c):
            wp.atomic_add(stats, 0, 1)
            if not _certify(old, target, wp.vec4i(vertex, a, b, c), 0, scalar_value(min_gap)):
                slot = wp.atomic_add(stats, 1, 1)
                if slot < witness.shape[0]:
                    witness[slot, 0] = 0
                    witness[slot, 1] = vertex
                    witness[slot, 2] = j


@position_overloads("old", "target", scalars={"min_gap":wp.float64})
@wp.kernel
def _edges_against_edges(
    old: wp.array(dtype=Any),
    target: wp.array(dtype=Any),
    invalid: wp.array(dtype=int),
    groups: wp.array(dtype=int),
    edges: wp.array2d(dtype=int),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    bvh: wp.uint64,
    min_gap: Any,
    stats: wp.array(dtype=int),
    witness: wp.array2d(dtype=int),
    enabled: wp.array(dtype=int),
):
    if invalid[0] or not enabled[0]:
        return
    i = wp.tid()
    a, b = edges[i, 0], edges[i, 1]
    query = wp.bvh_query_aabb(bvh, lower[i], upper[i])
    j = int(0)  # noqa: UP018, RUF046 -- Warp BVH iterator output
    while wp.bvh_query_next(query, j):
        c, d = edges[j, 0], edges[j, 1]
        if j > i and a != c and a != d and b != c and b != d and not same_rigid(groups, a, b, c, d):
            wp.atomic_add(stats, 0, 1)
            if not _certify(old, target, wp.vec4i(a, b, c, d), 1, scalar_value(min_gap)):
                slot = wp.atomic_add(stats, 1, 1)
                if slot < witness.shape[0]:
                    witness[slot, 0] = 1
                    witness[slot, 1] = i
                    witness[slot, 2] = j


@wp.kernel
def _validate_device_gap(gap: wp.array(dtype=wp.float64), minimum: wp.float64,
                         invalid: wp.array(dtype=int)):
    if not wp.isfinite(gap[0]) or gap[0]<minimum or gap[0]>wp.float64(8.5070586659632215e37):
        invalid[0]=1


@wp.kernel
def _motion_status(invalid: wp.array(dtype=int), stats: wp.array(dtype=int)):
    stats[2] = invalid[0]


class SurfaceMotionAudit:
    """Certify nonincident feature separation along one affine motion segment.

    `certified` assumes the initial volume was valid and nonoverlapping. The
    audit does not resolve contact, detect every initial solid overlap, or
    model nonlinear motion between the supplied endpoints. Query every actual
    solver substep, not just interpolated video frames, to check its trajectory.
    """

    def __init__(self, triangles, particle_count, *, device="cuda:0", min_gap_m=1e-7, rigid_groups=None):
        faces = np.asarray(triangles)
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0)
            or np.any(faces >= particle_count)
            or np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0)
        ):
            raise ValueError("triangles must be nonempty valid particle-index triples")
        self.device = wp.get_device(device)
        self.particle_count = int(particle_count)
        self.fast_reject = False
        self.min_gap = positive_scalar(min_gap_m, "self-contact audit gap")
        if self.min_gap > np.finfo(np.float32).max / 4 or np.float32(self.min_gap) <= 0:
            raise ValueError("self-contact gap must be representable in float32")
        self.groups = wp.array(collision_groups(rigid_groups, particle_count), dtype=int, device=self.device)
        self.host_faces = faces.astype(np.int32)
        self.host_edges = np.unique(
            np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0
        )
        self.vertices = wp.array(
            np.unique(faces).reshape(-1, 1), dtype=int, device=self.device
        )
        self.faces = wp.array(self.host_faces, dtype=int, device=self.device)
        self.edges = wp.array(self.host_edges, dtype=int, device=self.device)
        self.bounds = [
            (
                wp.zeros(len(f), dtype=wp.vec3, device=self.device),
                wp.zeros(len(f), dtype=wp.vec3, device=self.device),
            )
            for f in (self.vertices, self.faces, self.edges)
        ]
        self.face_bvh = wp.Bvh(*self.bounds[1])
        self.edge_bvh = wp.Bvh(*self.bounds[2])
        self._bvh_initialized = False
        self.invalid = wp.zeros(1, dtype=int, device=self.device)
        self.stats = wp.zeros(3, dtype=int, device=self.device)
        self.witness = wp.zeros((16, 3), dtype=int, device=self.device)
        # Smaller blocks may distribute divergent, register-heavy feature
        # certification more evenly. This changes scheduling, never pairs.
        self.query_block_dim = 256
        self.parallel_pairs = False
        self.pair_capacity = 0
        self.query_enabled = wp.ones(1,dtype=int,device=self.device)
        self.pair_count = wp.zeros(1,dtype=int,device=self.device)
        self.pair_work = wp.zeros(2,dtype=int,device=self.device)
        self.pairs = None
        self.pair_meta = None
        self.full_certificate = wp.ones(1,dtype=int,device=self.device)

    def configure_parallel(self,enabled=True,capacity=None):
        """Allocate before capture; capacity overflow uses the full GPU audit."""
        if capacity is None:capacity=max(64,8*(len(self.vertices)+len(self.edges)))
        if isinstance(capacity,bool) or not isinstance(capacity,int) or capacity<1:
            raise ValueError('CCD pair capacity must be a positive integer')
        if enabled and (self.pairs is None or capacity!=self.pair_capacity):
            self.pairs=wp.empty(capacity,dtype=wp.vec4i,device=self.device)
            self.pair_meta=wp.empty(capacity,dtype=wp.vec3i,device=self.device)
            self.pair_capacity=capacity
        self.parallel_pairs=bool(enabled)

    def _prepare_query(self, old, new, gap_override=None):
        """Validate endpoints and build outward-padded swept bounds on device."""
        if not isinstance(self.fast_reject,bool):raise ValueError('fast_reject must be a bool')
        if self.query_block_dim not in (32,64,128,256):
            raise ValueError('CCD query block dimension must be 32, 64, 128 or 256')
        if self.parallel_pairs and self.pairs is None:
            raise ValueError('configure parallel CCD buffers before recording')
        device_gap=isinstance(gap_override,wp.array)
        gap = gap_override if device_gap else self.min_gap if gap_override is None else positive_scalar(gap_override, "CCD certificate gap")
        if device_gap and (gap.shape!=(1,) or gap.dtype!=wp.float64 or gap.device!=self.device):
            raise ValueError('CCD device gap must be one float64 on the audit device')
        if not device_gap and (gap < self.min_gap or gap > np.finfo(np.float32).max / 4):
            raise ValueError("CCD certificate gap must not reduce the configured clearance")
        for array in (old, new):
            if (
                array.shape != (self.particle_count,)
                or array.dtype not in (wp.vec3, wp.vec3d)
                or array.device != self.device
            ):
                raise ValueError("positions must be matching Warp vec3 arrays")
        self.invalid.zero_()
        self.stats.zero_()
        if device_gap:
            wp.launch(_validate_device_gap,1,[gap,self.min_gap,self.invalid],device=self.device)
        def kernel(value):
            value=position_kernel(value,old)
            return device_scalar_kernel(value) if device_gap else value
        for features, (lo, hi) in zip(
            (self.vertices, self.faces, self.edges), self.bounds
        ):
            wp.launch(
                kernel(_bounds),
                len(features),
                inputs=[
                    old,
                    new,
                    features,
                    features.shape[1],
                    gap,
                    lo,
                    hi,
                    self.invalid,
                ],
                device=self.device,
            )
        # Build spatial topology from real bounds once. Refitting the original
        # all-zero tree is correct but causes excessive traversal on dense gels.
        for bvh in (self.face_bvh, self.edge_bvh):
            if self._bvh_initialized:
                bvh.refit()
            else:
                bvh.rebuild()
        return gap, device_gap

    def check(self, old, new, *, readback=True, gap_override=None, stop_on_witness=False, stop_after_rejection=False):
        # A trial only needs the Boolean result. Once one pair fails the same
        # certificate, other pairs may cancel; cancellation never certifies an
        # untested pair. Public diagnostics and final/seed audits enumerate all.
        if (stop_on_witness or stop_after_rejection) and readback:
            raise ValueError("early witness mode is only for Boolean device trial checks")
        gap, device_gap = self._prepare_query(old, new, gap_override)
        boolean_only = (stop_after_rejection and self.parallel_pairs and self.device.is_cuda
                        and old.dtype == wp.vec3d and new.dtype == wp.vec3d and not device_gap)
        def kernel(value):
            value = position_kernel(value, old)
            return device_scalar_kernel(value) if device_gap else value
        if self.parallel_pairs:
            from . import ipc_gpu_ccd_pairs as parallel
            self.pair_count.zero_()
            wp.launch(parallel.collect_points,len(self.vertices),[self.invalid,self.groups,self.vertices,self.faces,
                *self.bounds[0],self.face_bvh.id,self.pair_count,self.pairs,self.pair_meta],device=self.device)
            wp.launch(parallel.collect_edges,len(self.edges),[self.invalid,self.groups,self.edges,
                *self.bounds[2],self.edge_bvh.id,self.pair_count,self.pairs,self.pair_meta],device=self.device)
            wp.launch(parallel.classify,1,[self.pair_count,self.pair_capacity,self.query_enabled,self.stats,self.pair_work],device=self.device)
        else:
            self.query_enabled.fill_(1)
        def full():
            if self.parallel_pairs:
                if boolean_only:
                    wp.launch(parallel.certify_pairs_boolean,self.pair_capacity,
                        [old,new,self.invalid,self.query_enabled,self.pair_count,self.pairs,self.pair_meta,
                         gap,self.stats,self.witness,self.fast_reject],device=self.device,block_dim=self.query_block_dim)
                else:
                    pair_kernel=parallel.certify_pairs_fast if getattr(self,'fast_reject',False) else parallel.certify_pairs
                    wp.launch(kernel(pair_kernel),self.pair_capacity,[old,new,self.invalid,self.query_enabled,
                        self.pair_count,self.pairs,self.pair_meta,gap,self.stats,self.witness],device=self.device,block_dim=self.query_block_dim)
            wp.launch(
                kernel(_points_against_faces),
                len(self.vertices),
                inputs=[
                    old,
                    new,
                    self.invalid,
                    self.groups,
                    self.vertices,
                    self.faces,
                    *self.bounds[0],
                    self.face_bvh.id,
                    gap,
                    self.stats,
                    self.witness,
                    self.query_enabled,
                ],
                device=self.device,
                block_dim=self.query_block_dim,
            )
            wp.launch(
                kernel(_edges_against_edges),
                len(self.edges),
                inputs=[
                    old,
                    new,
                    self.invalid,
                    self.groups,
                    self.edges,
                    *self.bounds[2],
                    self.edge_bvh.id,
                    gap,
                    self.stats,
                    self.witness,
                    self.query_enabled,
                ],
                device=self.device,
                block_dim=self.query_block_dim,
            )
        # Unsupported dtypes/device gaps use the unchanged full path. Overflow
        # bypasses the prepass and triggers the existing exhaustive traversal.
        early=stop_on_witness and self.parallel_pairs and self.device.is_cuda and old.dtype==wp.vec3d and new.dtype==wp.vec3d and not device_gap
        if early:
            wp.launch(parallel.rejection_witnesses,self.pair_capacity,
                [old,new,self.invalid,self.query_enabled,self.pair_count,self.pairs,self.pair_meta,
                 gap,self.stats,self.witness],device=self.device,block_dim=self.query_block_dim)
            wp.launch(parallel.needs_full_certificate,1,[self.stats,self.full_certificate],device=self.device)
            wp.capture_if(self.full_certificate,full)
        else:full()
        # Invalid bounds have finite placeholders and both query kernels return
        # immediately. Keep the tree uninitialized after an invalid first call.
        wp.launch(_motion_status, 1, [self.invalid, self.stats], device=self.device)
        if not readback:
            return
        candidates, uncertified, invalid = (int(v) for v in self.stats.numpy())
        if invalid:
            raise ValueError("nonfinite surface positions or bounding boxes")
        self._bvh_initialized = True
        witnesses = []
        if uncertified:
            for kind, a, b in self.witness.numpy()[: min(16, uncertified)]:
                ids = (
                    [int(a), *self.host_faces[b].tolist()]
                    if kind == 0
                    else [*self.host_edges[a].tolist(), *self.host_edges[b].tolist()]
                )
                witnesses.append(
                    {
                        "kind": "point_triangle" if kind == 0 else "edge_edge",
                        "particle_indices": ids,
                    }
                )
        return {
            "certified": uncertified == 0,
            "candidate_pairs": candidates,
            "uncertified_pairs": uncertified,
            "witnesses": witnesses,
        }


@position_overloads("a", "b")
@wp.kernel
def positions_changed(
    a: wp.array(dtype=Any), b: wp.array(dtype=Any), changed: wp.array(dtype=int)
):
    i = wp.tid()
    if a[i][0] != b[i][0] or a[i][1] != b[i][1] or a[i][2] != b[i][2]:
        wp.atomic_max(changed, 0, 1)


@wp.func
def _axis_separates_tets(
    q: wp.array(dtype=Any),
    a: wp.vec4i,
    b: wp.vec4i,
    origin: wp.vec3d,
    axis: wp.vec3d,
    threshold: wp.float64,
):
    length = wp.length(axis)
    separate = False
    if length > wp.float64(0.0) and wp.isfinite(length):
        normal = axis / length
        amin, amax = wp.float64(wp.inf), wp.float64(-wp.inf)
        bmin, bmax = wp.float64(wp.inf), wp.float64(-wp.inf)
        for k in range(4):
            pa = wp.dot(normal, wp.vec3d(q[a[k]]) - origin)
            pb = wp.dot(normal, wp.vec3d(q[b[k]]) - origin)
            amin, amax = wp.min(amin, pa), wp.max(amax, pa)
            bmin, bmax = wp.min(bmin, pb), wp.max(bmax, pb)
        separate = bmin - amax >= threshold or amin - bmax >= threshold
    return separate


@wp.func
def _tets_disjoint(q: wp.array(dtype=Any), a: wp.vec4i, b: wp.vec4i):
    shared = int(0)  # noqa: UP018, RUF046 -- Warp dynamic-loop variable
    common = wp.vec3i(0)
    unique_a, unique_b = int(-1), int(-1)  # noqa: UP018, RUF046 -- Warp dynamic-loop variables
    for i in range(4):
        found = bool(False)  # noqa: UP018 -- Warp dynamic-loop variable
        for j in range(4):
            if a[i] == b[j]:
                found = True
        if found:
            if shared < 3:
                common[shared] = a[i]
            shared += 1
        else:
            unique_a = a[i]
    if shared == 3:
        for j in range(4):
            if b[j] != common[0] and b[j] != common[1] and b[j] != common[2]:
                unique_b = b[j]
        origin = wp.vec3d(q[common[0]])
        normal = wp.cross(
            wp.vec3d(q[common[1]]) - origin, wp.vec3d(q[common[2]]) - origin
        )
        da = wp.dot(normal, wp.vec3d(q[unique_a]) - origin)
        db = wp.dot(normal, wp.vec3d(q[unique_b]) - origin)
        # Shared-face cells are valid neighbors only on opposite sides.
        return (da > wp.float64(0.0) and db < wp.float64(0.0)) or (
            da < wp.float64(0.0) and db > wp.float64(0.0)
        )
    origin = wp.vec3d(q[a[0]])
    scale = wp.length(origin)
    for i in range(4):
        scale = wp.max(
            scale, wp.max(wp.length(wp.vec3d(q[a[i]])), wp.length(wp.vec3d(q[b[i]])))
        )
    threshold = wp.float64(128.0 * 2.220446049250313e-16) * scale
    # Incident cells must be allowed to share their prescribed edge/vertex.
    # Only those topological contacts receive a double-roundoff allowance.
    # Nonincident cells require a positively certified separating projection.
    if shared > 0:
        threshold = -threshold
    for face in range(4):
        i, j, k = (face + 1) % 4, (face + 2) % 4, (face + 3) % 4
        na = wp.cross(
            wp.vec3d(q[a[j]]) - wp.vec3d(q[a[i]]), wp.vec3d(q[a[k]]) - wp.vec3d(q[a[i]])
        )
        nb = wp.cross(
            wp.vec3d(q[b[j]]) - wp.vec3d(q[b[i]]), wp.vec3d(q[b[k]]) - wp.vec3d(q[b[i]])
        )
        if _axis_separates_tets(q, a, b, origin, na, threshold) or _axis_separates_tets(
            q, a, b, origin, nb, threshold
        ):
            return True
    for i in range(4):
        for j in range(i + 1, 4):
            ea = wp.vec3d(q[a[j]]) - wp.vec3d(q[a[i]])
            for k in range(4):
                for l in range(k + 1, 4):
                    eb = wp.vec3d(q[b[l]]) - wp.vec3d(q[b[k]])
                    if _axis_separates_tets(
                        q, a, b, origin, wp.cross(ea, eb), threshold
                    ):
                        return True
    return False


@position_overloads("q")
@wp.kernel
def _tet_pairs(
    q: wp.array(dtype=Any),
    tets: wp.array2d(dtype=int),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    bvh: wp.uint64,
    stats: wp.array(dtype=int),
    witnesses: wp.array2d(dtype=int),
):
    i = wp.tid()
    a = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    query = wp.bvh_query_aabb(bvh, lower[i], upper[i])
    j = int(0)  # noqa: UP018, RUF046 -- Warp BVH iterator output
    while wp.bvh_query_next(query, j):
        if j > i:
            wp.atomic_add(stats, 0, 1)
            b = wp.vec4i(tets[j, 0], tets[j, 1], tets[j, 2], tets[j, 3])
            if not _tets_disjoint(q, a, b):
                slot = wp.atomic_add(stats, 1, 1)
                if slot < witnesses.shape[0]:
                    witnesses[slot, 0] = i
                    witnesses[slot, 1] = j


class TetVolumeAudit:
    """SAT checks of all BVH candidate tet pairs, including containment.

    Requires valid nondegenerate tet geometry. Contact along shared topological
    features permits a scale-dependent float64 roundoff allowance; other pairs
    must be positively separated. An unresolved pair fails validation.
    """

    def __init__(self, tets, particle_count, *, device="cuda:0"):
        from .ipc_surface import boundary_triangles

        indices = np.asarray(tets)
        boundary_triangles(indices)
        if not len(indices) or np.any(indices >= particle_count):
            raise ValueError("tet indices must reference existing particles")
        self.device = wp.get_device(device)
        self.particle_count = int(particle_count)
        self.tets = wp.array(indices, dtype=int, device=self.device)
        self.lower = wp.zeros(len(indices), dtype=wp.vec3, device=self.device)
        self.upper = wp.zeros_like(self.lower)
        self.bvh = wp.Bvh(self.lower, self.upper)
        self._bvh_initialized = False
        self.invalid = wp.zeros(1, dtype=int, device=self.device)
        self.stats = wp.zeros(2, dtype=int, device=self.device)
        self.witnesses = wp.zeros((16, 2), dtype=int, device=self.device)

    def check(self, positions):
        if (
            positions.shape != (self.particle_count,)
            or positions.dtype not in (wp.vec3, wp.vec3d)
            or positions.device != self.device
        ):
            raise ValueError("positions must be a matching Warp vec3 array")
        self.invalid.zero_()
        self.stats.zero_()
        wp.launch(
            position_kernel(_bounds, positions),
            len(self.tets),
            inputs=[
                positions,
                positions,
                self.tets,
                4,
                0.0,
                self.lower,
                self.upper,
                self.invalid,
            ],
            device=self.device,
        )
        if self.invalid.numpy()[0]:
            raise ValueError("nonfinite tet positions or bounding boxes")
        if self._bvh_initialized:
            self.bvh.refit()
        else:
            self.bvh.rebuild()
            self._bvh_initialized = True
        wp.launch(
            position_kernel(_tet_pairs, positions),
            len(self.tets),
            inputs=[
                positions,
                self.tets,
                self.lower,
                self.upper,
                self.bvh.id,
                self.stats,
                self.witnesses,
            ],
            device=self.device,
        )
        candidates, unresolved = (int(v) for v in self.stats.numpy())
        return {
            "valid": unresolved == 0,
            "candidate_pairs": candidates,
            "overlapping_or_unresolved_pairs": unresolved,
            "witnesses": self.witnesses.numpy()[: min(16, unresolved)].tolist()
            if unresolved
            else [],
        }
