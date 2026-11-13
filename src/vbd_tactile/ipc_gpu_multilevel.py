"""GPU additive Schwarz over a fixed, balanced spatial aggregation hierarchy.

M^-1 = sum_l w_l P_l blockdiag(P_l^T A P_l)^-1 P_l^T. Eight 3D nodes per
domain; every fine free node appears at level zero, so the sum is SPD when A
is SPD. All changing matrix values (including dynamic self contact), local
Cholesky factors, restrictions and solves stay on device. Only immutable
topology/gather plans are constructed with NumPy, respecting disconnected
components. Domains need not be internally connected. This is an independent
compact MAS implementation, not the full StiffGIPC/METIS algorithm.
"""
import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator
from .ipc_gpu_math import mat12

wp.set_module_options({"enable_backward": False, "enable_mathdx_solver": False})
B = 8
D = 24


def hierarchy(positions, mass, tets):
    free = np.flatnonzero(np.asarray(mass) > 0)
    current = np.full(len(mass), -1, np.int32)
    current[free] = np.arange(len(free))
    adjacency = [set() for _ in free]
    for tet in np.asarray(tets):
        ids = current[tet]; ids = ids[ids >= 0]
        for a in ids:
            adjacency[a].update(int(b) for b in ids if b != a)
    components = np.full(len(free), -1, np.int32)
    component = 0
    for seed in range(len(free)):
        if components[seed] >= 0:
            continue
        pending = [seed]
        components[seed] = component
        while pending:
            node = pending.pop()
            for neighbor in adjacency[node]:
                if components[neighbor] < 0:
                    components[neighbor] = component
                    pending.append(neighbor)
        component += 1
    coords = np.asarray(positions)[free].astype(float)
    maps, offsets, counts = [], [], []
    while len(coords):
        # Balanced Morton aggregates prevent greedy graph grouping from leaving
        # long tails of singleton domains. Never merge disconnected components.
        extent = np.ptp(coords, axis=0)
        grid = np.floor((coords-coords.min(axis=0))/np.where(extent > 0, extent, 1.)*1023).astype(np.int64)
        morton = np.zeros(len(coords), np.int64)
        for bit in range(10):
            for axis in range(3):
                morton |= ((grid[:, axis] >> bit)&1) << (3*bit+axis)
        spatial = np.lexsort((np.arange(len(coords)), morton, components))
        boundaries = np.r_[0, np.flatnonzero(np.diff(components[spatial]))+1, len(coords)]
        groups = [spatial[i:min(i+B, end)] for start, end in zip(boundaries[:-1], boundaries[1:])
                  for i in range(start, end, B)]
        order = np.empty(len(coords), np.int32)
        for domain, group in enumerate(groups):
            order[group] = domain*B+np.arange(len(group))
        mapped = np.full(len(mass), -1, np.int32)
        mapped[free] = order[current[free]]
        maps.append(mapped)
        offsets.append(sum(counts)); counts.append(len(groups))
        if len(groups) <= 1 or len(groups) == len(coords):
            break
        coords = np.array([coords[g].mean(axis=0) for g in groups])
        components = np.array([components[g[0]] for g in groups])
        current[free] = mapped[free]//B
    return np.asarray(maps, dtype=np.int32).reshape(-1, len(mass)), np.asarray(offsets, np.int32), counts


def gather_plan(keys, indices, chunk=32):
    if not len(keys):
        return np.empty(0, np.int32), np.array([0], np.int32), np.empty(0, np.int32)
    order = np.argsort(keys, kind='stable')
    keys, indices = np.asarray(keys)[order], np.asarray(indices, np.int32)[order]
    boundaries = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1, len(keys)]
    starts = np.concatenate([np.arange(a, b, chunk) for a, b in zip(boundaries[:-1], boundaries[1:])])
    return keys[starts].astype(np.int32), np.r_[starts, len(keys)].astype(np.int32), indices


@wp.kernel
def assemble_fixed(values: wp.array(dtype=wp.mat33d), keys: wp.array(dtype=int),
                   offsets: wp.array(dtype=int), indices: wp.array(dtype=int),
                   blocks: wp.array3d(dtype=wp.float64)):
    i = wp.tid()
    key = keys[i]
    domain, row, col = key//64, (key//8)%8, key%8
    value = wp.mat33d(0.)
    for p in range(offsets[i], offsets[i+1]):
        value += values[indices[p]]
    for a in range(3):
        for b in range(3):
            wp.atomic_add(blocks, domain, 3*row+a, 3*col+b, value[a, b])


@wp.kernel
def assemble_contacts(ids: wp.array(dtype=wp.vec4i), active: wp.array(dtype=int),
                      count: wp.array(dtype=int), h: wp.array(dtype=mat12),
                      maps: wp.array2d(dtype=int), offsets: wp.array(dtype=int),
                      blocks: wp.array3d(dtype=wp.float64)):
    pair = wp.tid()
    slot, a, b = pair//16, (pair//4)%4, pair%4
    if slot < count[0]:
        i = active[slot]
        row, col = ids[i][a], ids[i][b]
        if row >= 0 and col >= 0:
            for level in range(maps.shape[0]):
                r, c = maps[level, row], maps[level, col]
                if r >= 0 and c >= 0 and r//8 == c//8:
                    domain = offsets[level]+r//8
                    for k in range(3):
                        for l in range(3):
                            wp.atomic_add(blocks, domain, 3*(r%8)+k, 3*(c%8)+l, h[i][3*a+k, 3*b+l])


@wp.kernel
def prepare_blocks(blocks: wp.array3d(dtype=wp.float64), used: wp.array2d(dtype=int),
                   prepared: wp.array3d(dtype=wp.float64)):
    d, i, j = wp.tid()
    value = (blocks[d, i, j]+blocks[d, j, i])*wp.float64(.5)
    if i == j and used[d, i//3] == 0:
        value = wp.float64(1.)
    prepared[d, i, j] = value


@wp.kernel
def factor_inverse(prepared: wp.array3d(dtype=wp.float64), identity: wp.array2d(dtype=wp.float64),
                   factors: wp.array3d(dtype=wp.float64), inverse: wp.array3d(dtype=wp.float64)):
    # One cooperating GPU block per domain. Reused for all subsequent PCG
    # products; no per-iteration triangular solve or host-side factorization.
    d = wp.tid()
    a = wp.tile_load(prepared[d], shape=(24, 24))
    l = wp.tile_cholesky(a)
    eye = wp.tile_load(identity, shape=(24, 24))
    inv = wp.tile_cholesky_solve(l, eye)
    wp.tile_store(factors[d], l)
    wp.tile_store(inverse[d], inv)


@wp.kernel
def check_factor(factors: wp.array3d(dtype=wp.float64), inverse: wp.array3d(dtype=wp.float64),
                 stats: wp.array(dtype=wp.float64)):
    d, row = wp.tid()
    valid = factors[d, row, row] > wp.float64(0.)
    for col in range(24):
        valid = valid and wp.isfinite(factors[d, row, col]) and wp.isfinite(inverse[d, row, col])
    if not valid:
        wp.atomic_add(stats, 2, wp.float64(1.))


@wp.kernel
def restrict(x: wp.array(dtype=wp.vec3d), keys: wp.array(dtype=int),
             offsets: wp.array(dtype=int), indices: wp.array(dtype=int),
             rhs: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    key = keys[i]
    value = wp.vec3d(0.)
    for p in range(offsets[i], offsets[i+1]):
        value += x[indices[p]]
    for a in range(3):
        wp.atomic_add(rhs, key//8, 3*(key%8)+a, value[a])


@wp.kernel
def local_product(inverse: wp.array3d(dtype=wp.float64), rhs: wp.array2d(dtype=wp.float64),
                  solution: wp.array2d(dtype=wp.float64)):
    d, row = wp.tid()
    value = wp.float64(0.)
    for col in range(24):
        # Enforce exact symmetry of the computed inverse used by PCG.
        value += wp.float64(.5)*(inverse[d, row, col]+inverse[d, col, row])*rhs[d, col]
    solution[d, row] = value


@wp.kernel
def prolong(x: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d),
            maps: wp.array2d(dtype=int), offsets: wp.array(dtype=int),
            weights: wp.array(dtype=wp.float64), solution: wp.array2d(dtype=wp.float64), alpha: wp.float64, beta: wp.float64,
            out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    value = wp.vec3d(0.)
    fixed = bool(True)
    for level in range(maps.shape[0]):
        slot = maps[level, i]
        if slot >= 0:
            fixed = False
            for a in range(3):
                value[a] += weights[level]*solution[offsets[level]+slot//8, 3*(slot%8)+a]
    if fixed:
        value = x[i]
    value *= alpha
    if beta != wp.float64(0.):
        value += beta*y[i]
    out[i] = value


class MultilevelSchwarz:
    def __init__(self, positions, mass, tets, matrix, device, *, hierarchy_data=None):
        self.device, self.n = device, len(mass)
        maps, offsets, self.domain_counts = hierarchy(positions, mass, tets) if hierarchy_data is None else hierarchy_data
        self.host_maps, self.host_offsets = maps, offsets
        self.levels = len(maps)
        count = sum(self.domain_counts)
        self.maps = wp.array(maps, dtype=int, device=device)
        self.offsets = wp.array(offsets, dtype=int, device=device)
        # Under-relax coarse corrections to avoid counting smooth modes equally
        # at every level. Positive weights preserve the SPD additive operator.
        weights = np.full(self.levels, .1)
        if self.levels:
            weights[0] = 1.
        self.weights = wp.array(weights, dtype=wp.float64, device=device)
        self.blocks = wp.zeros((count, D, D), dtype=wp.float64, device=device)
        self.prepared = wp.empty_like(self.blocks)
        self.identity = wp.array(np.eye(D), dtype=wp.float64, device=device)
        self.factors = wp.empty((count, D, D), dtype=wp.float64, device=device)
        self.inverse = wp.empty_like(self.factors)
        self.rhs = wp.zeros((count, D), dtype=wp.float64, device=device)
        self.solution = wp.empty((count, D), dtype=wp.float64, device=device)
        used = np.zeros((count, B), np.int32)
        rkeys, rindices, akeys, aindices = [], [], [], []
        # Immutable CSR topology is read only during construction.
        rows = np.repeat(np.arange(self.n), np.diff(matrix.offsets.numpy()))
        cols = matrix.columns.numpy()
        for mapping, offset in zip(maps, offsets):
            nodes = np.flatnonzero(mapping >= 0)
            slots = offset*B+mapping[nodes]
            used[slots//B, slots%B] = 1
            rkeys.extend(slots); rindices.extend(nodes)
            r, c = mapping[rows], mapping[cols]
            selected = np.flatnonzero((r >= 0)&(c >= 0)&(r//B == c//B))
            akeys.extend((offset+r[selected]//B)*B*B+(r[selected]%B)*B+c[selected]%B)
            aindices.extend(selected)
        self.used = wp.array(used, dtype=int, device=device)
        self.assembly_plan = tuple(wp.array(a, dtype=int, device=device) for a in gather_plan(np.array(akeys), aindices))
        self.restriction_plan = tuple(wp.array(a, dtype=int, device=device) for a in gather_plan(np.array(rkeys), rindices))
        self.operator = LinearOperator((3*self.n, 3*self.n), wp.float64, device, self.apply)

    def launch(self, kernel, n, args):
        if n and (not isinstance(n, tuple) or all(n)):
            wp.launch(kernel, n, args, device=self.device)

    def build(self, matrix, contacts, stats):
        self.blocks.zero_()
        self.launch(assemble_fixed, len(self.assembly_plan[0]), [matrix.values, *self.assembly_plan, self.blocks])
        for batch in contacts:
            self.launch(assemble_contacts, batch.capacity*16,
                [batch.ids, batch.active_indices, batch.active_count, batch.h, self.maps, self.offsets, self.blocks])
        count = len(self.factors)
        self.launch(prepare_blocks, (count, D, D), [self.blocks, self.used, self.prepared])
        if count:
            wp.launch_tiled(factor_inverse, dim=count, inputs=[self.prepared, self.identity, self.factors, self.inverse],
                            block_dim=32, device=self.device)
        self.launch(check_factor, (count, D), [self.factors, self.inverse, stats])

    def apply(self, x, y, z, alpha, beta):
        self.rhs.zero_()
        self.launch(restrict, len(self.restriction_plan[0]), [x, *self.restriction_plan, self.rhs])
        self.launch(local_product, (len(self.factors), D), [self.inverse, self.rhs, self.solution])
        self.launch(prolong, self.n, [x, y, self.maps, self.offsets, self.weights, self.solution, alpha, beta, z])
