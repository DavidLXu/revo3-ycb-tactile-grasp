"""GPU BVH and area-weighted IPC PT/EE candidates, with explicit overflow recovery.

For tetrahedral boundary meshes, IPC Toolkit's IPC collision set can be summed
without merging duplicate PP/PE stencils: their positive weights simply add.
This keeps feature classification, narrow phase and all changing pairs on GPU.
"""

import numpy as np
from typing import Any
from .ipc_precision import position_overloads
import warp as wp
from .ipc_precision import position_kernel
from .ipc import positive_scalar
from .ipc_gpu_math import mat12, squared_distance
from .ipc_self_audit import _bounds, _projected_gap, same_rigid, collision_groups
from .ipc_primitives_warp import _delta
from .ipc_vbd import IPCStepRejected

wp.set_module_options({"enable_backward": False})


@wp.func
def candidate_close(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d,
                    kind: int, threshold: wp.float64, conservative: bool):
    q, g, h, w = squared_distance(a, b, c, d, kind, False)
    if not conservative:
        return q < threshold * threshold
    # A closest-feature routine can overestimate distance in a degenerate
    # configuration. Only discard a cached pair using a separating-plane LOWER
    # bound; an inaccurate normal merely admits extra candidates.
    length = wp.sqrt(q)
    if not wp.isfinite(length) or length <= wp.float64(0.):
        return True
    normal = _delta(w, a, b, c, d) / length
    gap = _projected_gap(normal, a, b, c, d, kind)
    scale = wp.max(wp.max(wp.length(a), wp.length(b)), wp.max(wp.length(c), wp.length(d)))
    guard = wp.float64(128. * 2.220446049250313e-16) * (scale + threshold)
    return not wp.isfinite(gap) or gap <= threshold + guard


@wp.func
def append_pair(
    ids: wp.vec4i,
    kind: int,
    area: wp.float64,
    eps: wp.float64,
    output: wp.array(dtype=wp.vec4i),
    kinds: wp.array(dtype=int),
    areas: wp.array(dtype=wp.float64),
    moll_eps: wp.array(dtype=wp.float64),
    count: wp.array(dtype=int),
    capacity: int,
):
    slot = wp.atomic_add(count, 0, 1)
    if slot < capacity:
        output[slot] = ids
        kinds[slot] = kind
        areas[slot] = area
        moll_eps[slot] = eps


@position_overloads("x")
@wp.kernel
def find_pt(
    x: wp.array(dtype=Any),
    invalid: wp.array(dtype=int),
    groups: wp.array(dtype=int),
    vertices: wp.array2d(dtype=int),
    faces: wp.array2d(dtype=int),
    vertex_area: wp.array(dtype=wp.float64),
    bvh: wp.uint64,
    threshold: wp.float64,
    conservative: bool,
    output: wp.array(dtype=wp.vec4i),
    kinds: wp.array(dtype=int),
    areas: wp.array(dtype=wp.float64),
    moll_eps: wp.array(dtype=wp.float64),
    count: wp.array(dtype=int),
    capacity: int,
):
    if invalid[0]:
        return
    i = vertices[wp.tid(), 0]
    a = wp.vec3d(x[i])
    padding = float(threshold) * 0.5 + 8.0 * 1.1920928955078125e-7 * wp.length(
        wp.vec3(x[i])
    )
    query = wp.bvh_query_aabb(
        bvh, wp.vec3(x[i]) - wp.vec3(padding), wp.vec3(x[i]) + wp.vec3(padding)
    )
    f = int(0)
    while wp.bvh_query_next(query, f):
        j, k, l = faces[f, 0], faces[f, 1], faces[f, 2]
        if i != j and i != k and i != l and not same_rigid(groups, i, j, k, l):
            if candidate_close(a, wp.vec3d(x[j]), wp.vec3d(x[k]), wp.vec3d(x[l]),
                               0, threshold, conservative):
                append_pair(
                    wp.vec4i(i, j, k, l),
                    0,
                    wp.float64(0.25) * vertex_area[i],
                    wp.float64(0.0),
                    output,
                    kinds,
                    areas,
                    moll_eps,
                    count,
                    capacity,
                )


@position_overloads("x")
@wp.kernel
def find_ee(
    x: wp.array(dtype=Any),
    invalid: wp.array(dtype=int),
    groups: wp.array(dtype=int),
    edges: wp.array2d(dtype=int),
    edge_area: wp.array(dtype=wp.float64),
    edge_length2: wp.array(dtype=wp.float64),
    lo: wp.array(dtype=wp.vec3),
    hi: wp.array(dtype=wp.vec3),
    bvh: wp.uint64,
    threshold: wp.float64,
    conservative: bool,
    mollifier_coefficient: wp.float64,
    ee_multiplicity: wp.float64,
    output: wp.array(dtype=wp.vec4i),
    kinds: wp.array(dtype=int),
    areas: wp.array(dtype=wp.float64),
    moll_eps: wp.array(dtype=wp.float64),
    count: wp.array(dtype=int),
    capacity: int,
):
    if invalid[0]:
        return
    i = wp.tid()
    a, b = edges[i, 0], edges[i, 1]
    query = wp.bvh_query_aabb(bvh, lo[i], hi[i])
    j = int(0)
    while wp.bvh_query_next(query, j):
        if j > i:
            c, d = edges[j, 0], edges[j, 1]
            if a != c and a != d and b != c and b != d and not same_rigid(groups, a, b, c, d):
                if candidate_close(wp.vec3d(x[a]), wp.vec3d(x[b]), wp.vec3d(x[c]),
                                   wp.vec3d(x[d]), 1, threshold, conservative):
                    area = wp.float64(0.25) * ee_multiplicity * (edge_area[i] + edge_area[j])
                    eps = mollifier_coefficient * edge_length2[i] * edge_length2[j]
                    append_pair(
                        wp.vec4i(a, b, c, d),
                        1,
                        area,
                        eps,
                        output,
                        kinds,
                        areas,
                        moll_eps,
                        count,
                        capacity,
                    )


@wp.kernel
def query_status(count: wp.array(dtype=int), invalid: wp.array(dtype=int), out: wp.array(dtype=int)):
    out[0] = count[0]
    out[1] = invalid[0]


class ContactBatchGPU:
    def __init__(self, capacity, device):
        self.device = wp.get_device(device)
        capacity = int(capacity)
        self.capacity = capacity
        self.ids = wp.zeros(capacity, dtype=wp.vec4i, device=device)
        self.kinds = wp.zeros(capacity, dtype=int, device=device)
        self.areas = wp.zeros(capacity, dtype=wp.float64, device=device)
        self.eps = wp.zeros(capacity, dtype=wp.float64, device=device)
        self.count = wp.zeros(1, dtype=int, device=device)
        self.h = wp.zeros(capacity, dtype=mat12, device=device)
        self.active_indices = wp.zeros(capacity, dtype=int, device=device)
        self.active_count = wp.zeros(1, dtype=int, device=device)

    def frame(self):
        self.weights = wp.zeros(self.capacity, dtype=wp.vec4d, device=self.device)
        self.normals = wp.zeros(self.capacity, dtype=wp.vec3d, device=self.device)
        self.loads = wp.zeros(self.capacity, dtype=wp.float64, device=self.device)
        return self

    def copy_candidates(self, source, n):
        if n > self.capacity:
            raise ValueError("friction capacity too small")
        # Warp interprets count=0 as "copy the entire source", not no-op.
        # A rejected trial may grow source while the frozen frame stays small.
        if n:
            for name in ("ids", "kinds", "areas", "eps"):
                wp.copy(getattr(self, name), getattr(source, name), count=n)
        wp.copy(self.count, source.count)


class MeshContactGPU:
    def __init__(
        self,
        rest,
        faces,
        *,
        device,
        dhat,
        stiffness,
        min_gap,
        friction_mu=0.0,
        initial_capacity=4096,
        max_capacity=262144,
        candidate_skin=0.0,
        rigid_groups=None,
    ):
        self.device = wp.get_device(device)
        self.dhat = positive_scalar(dhat, "self activation distance")
        self.dmin = positive_scalar(min_gap, "self minimum gap")
        self.stiffness = positive_scalar(stiffness, "self barrier stiffness")
        self.mu = positive_scalar(friction_mu, "self friction", allow_zero=True)
        self.candidate_skin = positive_scalar(candidate_skin, "candidate skin", allow_zero=True)
        self.validate_parameters()
        for name, value in (
            ("initial capacity", initial_capacity),
            ("maximum capacity", max_capacity),
        ):
            if (
                isinstance(value, bool)
                or int(value) != value
                or not 1 <= value <= np.iinfo(np.int32).max
            ):
                raise ValueError(name + " must be a positive int32")
        if initial_capacity > max_capacity:
            raise ValueError("initial capacity exceeds maximum")
        self.max_capacity = int(max_capacity)
        rest = np.asarray(rest, dtype=np.float64)
        self._rest_reference = rest.copy()
        self.groups = wp.array(collision_groups(rigid_groups, len(rest)), dtype=int, device=device)
        faces = np.asarray(faces, dtype=np.int32)
        edges, inverse = np.unique(
            np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1),
            axis=0,
            return_inverse=True,
        )
        vertex_ids = np.unique(faces)
        maximum = len(vertex_ids) * len(faces) + len(edges) * (len(edges) - 1) // 2
        if maximum >= np.iinfo(np.int32).max:
            raise ValueError("mesh exceeds the int32 candidate-counter domain")
        tri = rest[faces]
        area = (
            np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
            )
            / 2
        )
        if not np.all(np.isfinite(area)) or np.any(area <= 0):
            raise ValueError("rest boundary must have finite nondegenerate faces")
        va = np.zeros(len(rest))
        ea = np.zeros(len(edges))
        np.add.at(va, faces.ravel(), np.repeat(area / 3, 3))
        np.add.at(ea, inverse, np.repeat(area / 3, 3))
        self.face_area = area
        self.vertices = wp.array(vertex_ids.reshape(-1, 1), dtype=int, device=device)
        self.faces = wp.array(faces, dtype=int, device=device)
        self.edges = wp.array(edges, dtype=int, device=device)
        self.vertex_area = wp.array(va, dtype=wp.float64, device=device)
        self.edge_area = wp.array(ea, dtype=wp.float64, device=device)
        self.edge_length2 = wp.array(
            ((rest[edges[:, 1]] - rest[edges[:, 0]]) ** 2).sum(1),
            dtype=wp.float64,
            device=device,
        )
        self.bounds = [
            (
                wp.zeros(len(f), dtype=wp.vec3, device=device),
                wp.zeros(len(f), dtype=wp.vec3, device=device),
            )
            for f in (faces, edges)
        ]
        self.bvhs = [wp.Bvh(*bounds) for bounds in self.bounds]
        self._bvh_initialized = False
        self.invalid = wp.zeros(1, dtype=int, device=device)
        self._status = wp.zeros(2, dtype=int, device=device)
        self.batch = ContactBatchGPU(initial_capacity, device)
        self.growths = 0
        self.last_count = 0
        self.revision = 0
        self.candidate_cache = None
        self.parallel_search = False
        self.parallel_query = None
        # Optional broad-phase reserve, independent of the narrow-phase pool.
        # Zero preserves the established sizing policy for existing callers.
        self.parallel_capacity_floor = 0
        self._node_count = len(rest)
        self._aggregation_power = 0.0
        self.aggregation = None
        self._convergent_contact = False
        self.convergent_surface = None
        self.taccel_discrete = False
        self.taccel_rigid_area = 'local'
        self.mollifier_coefficient = 1e-3
        self.ee_multiplicity = 1.

    @property
    def convergent_contact(self):
        return self._convergent_contact

    @convergent_contact.setter
    def convergent_contact(self,value):
        if not isinstance(value,(bool,np.bool_)):
            raise ValueError('convergent_contact must be boolean')
        if value and self.aggregation_power:
            raise ValueError('convergent contact cannot be combined with experimental aggregation')
        if value and self.convergent_surface is None:
            from .ipc_convergent import ConvergentSurface
            self.convergent_surface=ConvergentSurface(self)
        if bool(value)!=self._convergent_contact:
            self._convergent_contact=bool(value);self.revision+=1
            if self.candidate_cache is not None:self.candidate_cache.valid.zero_()

    @property
    def aggregation_power(self):
        return self._aggregation_power

    @aggregation_power.setter
    def aggregation_power(self, value):
        if value and self.convergent_contact:
            raise ValueError('aggregation cannot be combined with convergent contact')
        if isinstance(value, bool) or not np.isfinite(value) or (value not in (-3, -2, -1, 0) and not 2 <= value <= 128):
            raise ValueError('contact aggregation power must be 0 (legacy sum), -1 (maximum), -2 (global area softmax), -3 (local area norm), or in [2, 128]')
        if float(value) != self._aggregation_power:
            self._aggregation_power = float(value)
            self.revision += 1
            self.aggregation = None

    def prepare_aggregation(self):
        if self.aggregation_power and (self.aggregation is None or self.aggregation.capacity != self.batch.capacity):
            from .ipc_contact_aggregation import ContactAggregation
            if self.aggregation_power == -3:
                from .ipc_contact_local_area import LocalAreaNorm
                self.aggregation = LocalAreaNorm(self)
            elif self.aggregation_power == -2:
                from .ipc_contact_softmax import AreaSoftmax
                self.aggregation = AreaSoftmax(self)
            else:self.aggregation = ContactAggregation(self, self.aggregation_power)

    def validate_parameters(self):
        """Mutable contact settings must retain a valid barrier and friction."""
        if getattr(self,'taccel_discrete',False) and (self.convergent_contact or self.aggregation_power):
            raise ValueError('Taccel discrete contact is incompatible with aggregation/convergent contact')
        positive_scalar(self.dhat, "self activation distance")
        positive_scalar(self.dmin, "self minimum gap")
        positive_scalar(self.stiffness, "self barrier stiffness")
        positive_scalar(self.mu, "self friction", allow_zero=True)
        positive_scalar(self.candidate_skin, "candidate skin", allow_zero=True)
        if self.dmin >= self.dhat:
            raise ValueError("self minimum gap must be below activation distance")

    def build(self, x, *, readback=True):
        self.validate_parameters()
        if self.parallel_search:self.prepare_candidate_cache()
        # Invalidate any pending Hessian evaluation even if this build fails.
        self.revision += 1
        if self.candidate_skin > 0:
            self.prepare_candidate_cache()
            result = self.candidate_cache.build(x, readback=readback)
        else:
            # An uncached query may intervene between two cached queries.
            if self.candidate_cache is not None:
                self.candidate_cache.valid.zero_()
            result = self._build_uncached(x, readback=readback)
        self.prepare_aggregation()
        if self.aggregation is not None:
            self.aggregation.build(x)
            if readback and self.invalid.numpy()[0]:
                raise IPCStepRejected('Nonfinite aggregated contact potential')
        return result

    def prepare_candidate_cache(self):
        """Allocate only outside graph capture; keep growth transactional."""
        self.prepare_aggregation()
        if self.parallel_search:
            if not self.device.is_cuda:
                raise ValueError('parallel contact search requires CUDA')
            from .ipc_gpu_parallel_search import ParallelContactSearch
            floor = self.parallel_capacity_floor
            if (isinstance(floor, (bool, np.bool_)) or not isinstance(floor, (int, np.integer))
                    or floor < 0 or floor > self.max_capacity):
                raise ValueError('parallel capacity floor must be an integer between zero and max_capacity')
            capacity=min(self.max_capacity,max(4096,4*self.batch.capacity,int(floor)))
            if self.parallel_query is None or self.parallel_query.capacity!=capacity:
                self.parallel_query=ParallelContactSearch(self,capacity)
        if self.candidate_skin > 0:
            from .ipc_gpu_candidate_cache import CandidateCache
            if self.candidate_cache is None or self.candidate_cache.capacity != self.batch.capacity:
                self.candidate_cache = CandidateCache(self)

    def _search(self, x, batch, threshold, *, conservative=False):
        """One GPU BVH refit and query, with no allocation or host readback."""
        self.invalid.zero_()
        for features, bounds, bvh in zip(
            (self.faces, self.edges), self.bounds, self.bvhs
        ):
            wp.launch(
                position_kernel(_bounds, x),
                len(features),
                inputs=[
                    x,
                    x,
                    features,
                    features.shape[1],
                    float(threshold / 2),
                    *bounds,
                    self.invalid,
                ],
                device=self.device,
            )
        # Refit preserves topology. The placeholder zero bounds above cannot
        # define a useful spatial tree; build topology on the first valid input.
        for bvh in self.bvhs:
            if self._bvh_initialized:
                bvh.refit()
            else:
                bvh.rebuild()
        b = batch
        b.count.zero_()
        if self.parallel_search:
            self.parallel_query.search(self,x,b,threshold,conservative,
                lambda:self._search_serial(x,b,threshold,conservative))
        else:
            self._search_serial(x,b,threshold,conservative)
        if self.convergent_contact:
            self.convergent_surface.search(x,batch,threshold,conservative)

    def _search_serial(self,x,b,threshold,conservative):
        """Original fused traversal, including complete overflow fallback."""
        args = [b.ids, b.kinds, b.areas, b.eps, b.count, b.capacity]
        wp.launch(
            position_kernel(find_pt, x),
            len(self.vertices),
            inputs=[
                x,
                self.invalid,
                self.groups,
                self.vertices,
                self.faces,
                self.vertex_area,
                self.bvhs[0].id,
                threshold,
                conservative,
                *args,
            ],
            device=self.device,
        )
        wp.launch(
            position_kernel(find_ee, x),
            len(self.edges),
            inputs=[
                x,
                self.invalid,
                self.groups,
                self.edges,
                self.edge_area,
                self.edge_length2,
                *self.bounds[1],
                self.bvhs[1].id,
                threshold,
                conservative,
                self.mollifier_coefficient,
                self.ee_multiplicity,
                *args,
            ],
            device=self.device,
        )

    def _build_uncached(self, x, *, readback=True):
        while True:
            b = self.batch
            self._search(x, b, self.dhat + self.dmin)
            # _bounds substitutes finite zero boxes on invalid input; queries
            # are device-guarded. Read validity and capacity in one transaction.
            wp.launch(query_status, 1, [b.count, self.invalid, self._status], device=self.device)
            if not readback:
                # Caller must device-check invalid/count before consuming pairs.
                # Capacity growth is an explicit whole-attempt retry on the host.
                return
            required, invalid = (int(v) for v in self._status.numpy())
            if invalid:
                raise IPCStepRejected("Nonfinite GPU collision bounds")
            self._bvh_initialized = True
            if required <= b.capacity:
                self.last_count = required
                return required
            if required > self.max_capacity:
                raise IPCStepRejected(
                    f"GPU contact capacity exceeded: {required} > {self.max_capacity}; no pairs discarded"
                )
            new_capacity = min(
                self.max_capacity, max(b.capacity * 2, 1 << (required - 1).bit_length())
            )
            self.batch = ContactBatchGPU(new_capacity, self.device)
            self.growths += 1
