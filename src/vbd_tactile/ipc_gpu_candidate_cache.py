"""Conservative GPU neighbor list, separate from active and lagged contacts.

The pool contains all pairs within radius + skin of reference positions. A
pair's distance can decrease by at most twice the largest vertex displacement.
We reuse only while 2 * (displacement + rounding allowance) < skin. Every query
filters the pool with the original narrow phase at CURRENT positions.
"""
from typing import Any

import warp as wp

from .ipc_gpu_contacts import append_pair, candidate_close, query_status, ContactBatchGPU
from .ipc_precision import position_kernel, position_overloads
from .ipc_vbd import IPCStepRejected

wp.set_module_options({"enable_backward": False})


@position_overloads("x")
@wp.kernel
def displacement(x: wp.array(dtype=Any), reference: wp.array(dtype=wp.vec3d),
                 radius: wp.float64, skin: wp.float64,
                 largest: wp.array(dtype=wp.float64), invalid: wp.array(dtype=int)):
    i = wp.tid()
    p, r = wp.vec3d(x[i]), reference[i]
    for j in range(3):
        if not wp.isfinite(p[j]) or not wp.isfinite(float(p[j])):
            wp.atomic_max(invalid, 0, 1)
            return
    distance = wp.length(p-r)
    guard = wp.float64(128. * 2.220446049250313e-16) * (wp.length(p)+wp.length(r)+radius+skin)
    if not wp.isfinite(distance+guard):
        wp.atomic_max(invalid, 0, 1)
        return
    wp.atomic_max(largest, 0, distance+guard)


@wp.kernel
def decide(largest: wp.array(dtype=wp.float64), invalid: wp.array(dtype=int),
           valid: wp.array(dtype=int), parameters: wp.array(dtype=wp.float64),
           radius: wp.float64, skin: wp.float64,
           rebuild: wp.array(dtype=int), statistics: wp.array(dtype=int)):
    rebuild[0] = 0
    if invalid[0] != 0:
        valid[0] = 0
    elif (valid[0] == 0 or parameters[0] != radius or parameters[1] != skin
          or wp.float64(2.)*largest[0] >= skin):
        rebuild[0] = 1
        statistics[0] += 1
    else:
        statistics[1] += 1


@position_overloads("x")
@wp.kernel
def save_positions(x: wp.array(dtype=Any), reference: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    reference[i] = wp.vec3d(x[i])


@wp.kernel
def commit(count: wp.array(dtype=int), capacity: int, invalid: wp.array(dtype=int),
           valid: wp.array(dtype=int), parameters: wp.array(dtype=wp.float64),
           radius: wp.float64, skin: wp.float64):
    valid[0] = int(invalid[0] == 0 and count[0] <= capacity)
    parameters[0] = radius
    parameters[1] = skin


@position_overloads("x")
@wp.kernel
def filter_current(x: wp.array(dtype=Any), invalid: wp.array(dtype=int),
                   ids: wp.array(dtype=wp.vec4i), kinds: wp.array(dtype=int),
                   areas: wp.array(dtype=wp.float64), eps: wp.array(dtype=wp.float64),
                   count: wp.array(dtype=int), capacity: int, radius: wp.float64,
                   output: wp.array(dtype=wp.vec4i), output_kinds: wp.array(dtype=int),
                   output_areas: wp.array(dtype=wp.float64), output_eps: wp.array(dtype=wp.float64),
                   output_count: wp.array(dtype=int)):
    i = wp.tid()
    if invalid[0] == 0 and count[0] <= capacity and i < count[0]:
        ind = ids[i]
        if candidate_close(wp.vec3d(x[ind[0]]), wp.vec3d(x[ind[1]]),
                           wp.vec3d(x[ind[2]]), wp.vec3d(x[ind[3]]), kinds[i], radius, False):
            append_pair(ind, kinds[i], areas[i], eps[i], output, output_kinds,
                        output_areas, output_eps, output_count, capacity)


@wp.kernel
def overflow(count: wp.array(dtype=int), capacity: int, maximum: int,
             invalid: wp.array(dtype=int), output_count: wp.array(dtype=int),
             fallback: wp.array(dtype=int), statistics: wp.array(dtype=int)):
    fallback[0] = 0
    if invalid[0] == 0 and count[0] > capacity:
        if capacity < maximum:
            # Existing capacity handling retries the whole attempt. No partial
            # pool is ever filtered or consumed by an objective evaluation.
            output_count[0] = wp.min(count[0], maximum)
        else:
            # A large skin must not reject a scene whose ACTIVE set fits. Use
            # the original full GPU search; this is not a CPU physics fallback.
            fallback[0] = 1
            statistics[2] += 1


class CandidateCache:
    def __init__(self, mesh):
        self.mesh, self.device, self.capacity = mesh, mesh.device, mesh.batch.capacity
        self.ids = wp.empty(self.capacity, dtype=wp.vec4i, device=self.device)
        self.kinds = wp.empty(self.capacity, dtype=int, device=self.device)
        self.areas = wp.empty(self.capacity, dtype=wp.float64, device=self.device)
        self.eps = wp.empty(self.capacity, dtype=wp.float64, device=self.device)
        self.count = wp.zeros(1, dtype=int, device=self.device)
        self.reference = wp.zeros(mesh._node_count, dtype=wp.vec3d, device=self.device)
        self.largest = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.valid = wp.zeros(1, dtype=int, device=self.device)
        self.parameters = wp.zeros(2, dtype=wp.float64, device=self.device)
        self.rebuild = wp.zeros(1, dtype=int, device=self.device)
        self.fallback = wp.zeros(1, dtype=int, device=self.device)
        previous = mesh.candidate_cache
        self.statistics = (previous.statistics if previous is not None else
                           wp.zeros(3, dtype=int, device=self.device))

    def launch(self, kernel, dim, args):
        if dim:
            wp.launch(kernel, dim, args, device=self.device)

    def build(self, x, *, readback):
        m, b = self.mesh, self.mesh.batch
        if x.ndim != 1 or len(x) != m._node_count:
            raise ValueError("candidate cache positions must match the mesh vertex count")
        radius, skin = m.dhat + m.dmin, m.candidate_skin
        m.invalid.zero_()
        self.largest.zero_()
        self.launch(position_kernel(displacement, x), len(x),
                    [x, self.reference, radius, skin, self.largest, m.invalid])
        self.launch(decide, 1, [self.largest, m.invalid, self.valid, self.parameters,
                               radius, skin, self.rebuild, self.statistics])

        def rebuild():
            m._search(x, self, radius + skin, conservative=True)
            self.launch(position_kernel(save_positions, x), len(x), [x, self.reference])
            self.launch(commit, 1, [self.count, self.capacity, m.invalid, self.valid,
                                   self.parameters, radius, skin])

        def conditional(condition, callback):
            if readback:
                if int(condition.numpy()[0]):
                    callback()
            else:
                wp.capture_if(condition, callback)

        conditional(self.rebuild, rebuild)
        b.count.zero_()
        self.launch(position_kernel(filter_current, x), self.capacity,
                    [x, m.invalid, self.ids, self.kinds, self.areas, self.eps, self.count,
                     self.capacity, radius, b.ids, b.kinds, b.areas, b.eps, b.count])
        self.launch(overflow, 1, [self.count, self.capacity, m.max_capacity, m.invalid,
                                 b.count, self.fallback, self.statistics])
        conditional(self.fallback, lambda: m._search(x, b, radius))
        self.launch(query_status, 1, [b.count, m.invalid, m._status])
        if not readback:
            return
        required, invalid = (int(v) for v in m._status.numpy())
        if invalid:
            raise IPCStepRejected("Nonfinite GPU collision bounds")
        m._bvh_initialized = True
        if required <= b.capacity:
            m.last_count = required
            return required
        if required > m.max_capacity:
            # If only the optional pool is too large, grow to the cap and let
            # the next query take the exact-search path above.
            if self.capacity == m.max_capacity:
                raise IPCStepRejected(f"GPU contact capacity exceeded: {required} > {m.max_capacity}; no pairs discarded")
        capacity = min(m.max_capacity, max(2*b.capacity, 1 << (required-1).bit_length()))
        m.batch = ContactBatchGPU(capacity, self.device)
        m.growths += 1
        m.prepare_candidate_cache()
        return m.candidate_cache.build(x, readback=True)
