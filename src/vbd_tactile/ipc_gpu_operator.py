"""Fixed FEM/sphere sparsity assembled once per Newton iteration on CUDA.

The immutable node-pair map is prepared on the host during construction. All
changing Hessian values and products stay on device. Dynamic self-contact is
handled separately, so new nonlocal contacts never change this fixed topology.
"""
import numpy as np
import warp as wp

from .ipc_gpu_math import mat12

wp.set_module_options({"enable_backward": False})


@wp.kernel
def compact_nonzero(h: wp.array(dtype=mat12), count: wp.array(dtype=int),
                    indices: wp.array(dtype=int), active_count: wp.array(dtype=int)):
    i = wp.tid()
    if i < count[0]:
        block = h[i]
        nonzero = False
        for a in range(12):
            for b in range(12):
                # NaNs also remain active and fail the subsequent validation.
                nonzero = nonzero or block[a, b] != wp.float64(0.)
        if nonzero:
            slot = wp.atomic_add(active_count, 0, 1)
            indices[slot] = i


@wp.kernel
def compact_nonzero_free(h: wp.array(dtype=mat12), ids: wp.array(dtype=wp.vec4i),
                    mass: wp.array(dtype=float), count: wp.array(dtype=int),
                    indices: wp.array(dtype=int), active_count: wp.array(dtype=int)):
    i = wp.tid()
    if i < count[0]:
        ind = ids[i]
        any_free = False
        for a in range(4):
            if ind[a] >= 0:
                any_free = any_free or mass[ind[a]] > 0.
        if any_free:
            block = h[i]
            nonzero = False
            for a in range(12):
                for b in range(12):
                    nonzero = nonzero or block[a, b] != wp.float64(0.)
            if nonzero:
                slot = wp.atomic_add(active_count, 0, 1)
                indices[slot] = i


@wp.kernel
def initialize_values(mass: wp.array(dtype=float), dt: wp.array(dtype=wp.float64),
                      diagonal_slots: wp.array(dtype=int), values: wp.array(dtype=wp.mat33d)):
    i = wp.tid()
    scale = wp.float64(1.)
    if mass[i] > 0.:
        scale = wp.float64(mass[i]) / (dt[0]*dt[0])
    values[diagonal_slots[i]] = scale * wp.identity(n=3, dtype=wp.float64)


@wp.kernel
def assemble_blocks(h: wp.array(dtype=mat12), ids: wp.array(dtype=wp.vec4i),
                    slots: wp.array3d(dtype=int), mass: wp.array(dtype=float),
                    active: wp.array(dtype=int), count: wp.array(dtype=int),
                    compact: bool, values: wp.array(dtype=wp.mat33d)):
    i = wp.tid()
    if i < count[0]:
        if compact:
            i = active[i]
        ind, block = ids[i], h[i]
        for a in range(4):
            for b in range(4):
                row, col = ind[a], ind[b]
                if row >= 0 and col >= 0:
                    if mass[row] > 0. and mass[col] > 0.:
                        value = wp.mat33d(0.)
                        for k in range(3):
                            for l in range(3):
                                value[k, l] = block[3*a+k, 3*b+l]
                        wp.atomic_add(values, slots[i, a, b], value)


@wp.kernel
def csr_product(offsets: wp.array(dtype=int), columns: wp.array(dtype=int),
                values: wp.array(dtype=wp.mat33d), x: wp.array(dtype=wp.vec3d),
                out: wp.array(dtype=wp.vec3d)):
    row = wp.tid()
    value = wp.vec3d(0.)
    for p in range(offsets[row], offsets[row+1]):
        value += values[p] * x[columns[p]]
    out[row] = value


@wp.kernel
def csr_product_rows(offsets: wp.array(dtype=int), columns: wp.array(dtype=int),
                     values: wp.array3d(dtype=wp.float64), x: wp.array(dtype=wp.vec3d),
                     out: wp.array2d(dtype=wp.float64)):
    """One thread per scalar row, retaining the original summation grouping."""
    row, k = wp.tid()
    value = wp.float64(0.)
    for p in range(offsets[row], offsets[row+1]):
        vector = x[columns[p]]
        part = values[p, k, 0]*vector[0] + values[p, k, 1]*vector[1] + values[p, k, 2]*vector[2]
        value += part
    out[row, k] = value


@wp.kernel
def pack_product_rows(offsets: wp.array(dtype=int), columns: wp.array(dtype=int),
                      values: wp.array(dtype=wp.mat33d), mass: wp.array(dtype=float),
                      counts: wp.array(dtype=int), packed_columns: wp.array(dtype=int),
                      packed_values: wp.array(dtype=wp.mat33d)):
    """Pack free/free blocks and every diagonal once after assembly, on GPU.

    Each row retains its original storage range, avoiding a prefix scan or
    capacity changes. Update counts even for empty/fixed rows. The topology is
    never pruned on the host, so new constraints cannot leave it stale.
    """
    row = wp.tid()
    start = offsets[row]
    count = int(0)
    for p in range(start, offsets[row+1]):
        col = columns[p]
        if row == col or (mass[row] > 0. and mass[col] > 0.):
            packed_columns[start+count] = col
            packed_values[start+count] = values[p]
            count += 1
    counts[row] = count


@wp.kernel
def csr_packed_product_rows(offsets: wp.array(dtype=int), counts: wp.array(dtype=int),
                     columns: wp.array(dtype=int), values: wp.array3d(dtype=wp.float64),
                     x: wp.array(dtype=wp.vec3d), out: wp.array2d(dtype=wp.float64)):
    row, k = wp.tid()
    value = wp.float64(0.)
    for p in range(offsets[row], offsets[row]+counts[row]):
        col = columns[p]
        vector = x[col]
        part = values[p, k, 0]*vector[0] + values[p, k, 1]*vector[1] + values[p, k, 2]*vector[2]
        value += part
    out[row, k] = value


@wp.kernel
def extract_diagonal(slots: wp.array(dtype=int), values: wp.array(dtype=wp.mat33d),
                     diagonal: wp.array(dtype=wp.mat33d)):
    i = wp.tid()
    diagonal[i] = values[slots[i]]


@wp.kernel
def active_diagonal(ids: wp.array(dtype=wp.vec4i), indices: wp.array(dtype=int),
                    count: wp.array(dtype=int), h: wp.array(dtype=mat12),
                    mass: wp.array(dtype=float), out: wp.array(dtype=wp.mat33d)):
    slot = wp.tid()
    if slot < count[0]:
        i = indices[slot]
        ind, block = ids[i], h[i]
        for a in range(4):
            row = ind[a]
            if row >= 0:
                if mass[row] > 0.:
                    value = wp.mat33d(0.)
                    for k in range(3):
                        for l in range(3):
                            value[k, l] = block[3*a+k, 3*a+l]
                    wp.atomic_add(out, row, value)


@wp.kernel
def active_product(x: wp.array(dtype=wp.vec3d), ids: wp.array(dtype=wp.vec4i),
                   indices: wp.array(dtype=int), count: wp.array(dtype=int),
                   h: wp.array(dtype=mat12), mass: wp.array(dtype=float),
                   out: wp.array(dtype=wp.vec3d)):
    slot = wp.tid()
    if slot < count[0]:
        i = indices[slot]
        ind, block = ids[i], h[i]
        for a in range(4):
            row = ind[a]
            if row >= 0:
                if mass[row] > 0.:
                    value = wp.vec3d(0.)
                    for b in range(4):
                        col = ind[b]
                        if col >= 0:
                            if mass[col] > 0.:
                                for k in range(3):
                                    for l in range(3):
                                        value[k] += block[3*a+k, 3*b+l]*x[col][l]
                    wp.atomic_add(out, row, value)


@wp.kernel
def active_product_rows(x: wp.array(dtype=wp.vec3d), ids: wp.array(dtype=wp.vec4i),
                        indices: wp.array(dtype=int), count: wp.array(dtype=int),
                        h: wp.array3d(dtype=wp.float64), mass: wp.array(dtype=float),
                        out: wp.array2d(dtype=wp.float64)):
    """Twelve independent rows per contact, without per-thread mat12 copies."""
    slot, local = wp.tid()
    if slot < count[0]:
        i = indices[slot]
        ind = ids[i]
        row = ind[local//3]
        if row >= 0:
            if mass[row] > 0.:
                value = wp.float64(0.)
                for b in range(4):
                    col = ind[b]
                    if col >= 0:
                        if mass[col] > 0.:
                            for l in range(3):
                                value += h[i, local, 3*b+l]*x[col][l]
                wp.atomic_add(out, row, local%3, value)


@wp.kernel
def pack_cg_status(iterations: wp.array(dtype=int), residual_sq: wp.array(dtype=wp.float64),
                   stats: wp.array(dtype=wp.float64)):
    stats[3] = wp.float64(iterations[0])
    stats[4] = residual_sq[0]


@wp.kernel
def merge_invalid(source: wp.array(dtype=wp.float64), target: wp.array(dtype=wp.float64)):
    target[2] += source[2]


class FixedBlockMatrix:
    def __init__(self, n, tets, faces, device):
        self.n, self.device = n, device
        tets = np.asarray(tets, dtype=np.int64)
        sphere = np.column_stack((np.full(len(faces), -1), faces)).astype(np.int64)
        codes = np.unique(np.concatenate((np.arange(n, dtype=np.int64)*(n+1),
            (tets[:, :, None]*n+tets[:, None, :]).ravel(),
            (faces[:, :, None].astype(np.int64)*n+faces[:, None, :]).ravel())))
        if len(codes) > np.iinfo(np.int32).max:
            raise ValueError("IPC block topology exceeds int32 domain")
        offsets = np.r_[0, np.cumsum(np.bincount(codes//n, minlength=n))].astype(np.int32)
        self.offsets = wp.array(offsets, dtype=int, device=device)
        self.columns = wp.array((codes % n).astype(np.int32), dtype=int, device=device)
        self.values = wp.zeros(len(codes), dtype=wp.mat33d, device=device)
        self.diagonal_slots = wp.array(np.searchsorted(codes, np.arange(n)*(n+1)).astype(np.int32), dtype=int, device=device)
        maps = []
        for ids in (tets, sphere):
            valid = (ids[:, :, None] >= 0) & (ids[:, None, :] >= 0)
            keys = ids[:, :, None]*n+ids[:, None, :]
            slots = np.full(keys.shape, -1, dtype=np.int32)
            slots[valid] = np.searchsorted(codes, keys[valid])
            maps.append(wp.array(slots, dtype=int, device=device))
        self.tet_slots, self.sphere_slots = maps
        self.product_counts = wp.zeros(n, dtype=int, device=device)
        self.product_columns = wp.empty_like(self.columns)
        self.product_values = wp.empty_like(self.values)

    def assemble(self, mass, dt, tets, sphere, friction, *, packed_product=True):
        self.values.zero_()
        wp.launch(initialize_values, self.n, [mass, dt, self.diagonal_slots, self.values], device=self.device)
        for batch, slots, compact in ((tets, self.tet_slots, False),
                (sphere, self.sphere_slots, True), (friction, self.sphere_slots, True)):
            wp.launch(assemble_blocks, batch.capacity,
                [batch.h, batch.ids, slots, mass, batch.active_indices,
                 batch.active_count if compact else batch.count, compact, self.values], device=self.device)
        if packed_product:
            self.prepare_product(mass)

    def prepare_product(self, mass):
        wp.launch(pack_product_rows, self.n, [self.offsets, self.columns, self.values,
            mass, self.product_counts, self.product_columns, self.product_values], device=self.device)

    def product(self, x, out, *, packed=False):
        # Views share GPU storage. Recreate them when recording so callers that
        # replace matrix buffers (e.g. structural pruning) cannot leave stale views.
        scalar_out = out.view(wp.float64).reshape((self.n, 3))
        if not packed:
            values = self.values.view(wp.float64).reshape((len(self.values), 3, 3))
            wp.launch(csr_product_rows, (self.n, 3),
                [self.offsets, self.columns, values, x, scalar_out], device=self.device)
        else:
            values = self.product_values.view(wp.float64).reshape((len(self.product_values), 3, 3))
            wp.launch(csr_packed_product_rows, (self.n, 3),
                [self.offsets, self.product_counts, self.product_columns, values, x, scalar_out], device=self.device)

    def diagonal(self, out):
        wp.launch(extract_diagonal, self.n, [self.diagonal_slots, self.values, out], device=self.device)
