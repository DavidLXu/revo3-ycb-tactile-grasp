"""Assemble six-node surface tangents once per Newton linearization on GPU.

All FEM triangle cliques and affine couplings belong to JointMatrix's pattern.
A missing nonzero entry is an error, never permission to omit a contact term.
"""
import warp as wp
from .ipc_surface_gpu import vec6i
from .abd_operator import coordinate, coefficient, slot_of

wp.set_module_options({'enable_backward': False})


@wp.kernel
def assemble(ids: wp.array(dtype=vec6i), head: wp.array(dtype=int),
             h: wp.array3d(dtype=wp.float64), count: wp.array(dtype=int),
             invalid: wp.array(dtype=int), owner: wp.array(dtype=int),
             weights: wp.array(dtype=wp.vec4d), mass: wp.array(dtype=float), n: int,
             offsets: wp.array(dtype=int), columns: wp.array(dtype=int),
             values: wp.array(dtype=wp.mat33d), error: wp.array(dtype=int), lanes: int):
    lane, a, b = wp.tid()
    if invalid[0] != 0 or count[0] < 0 or count[0] > ids.shape[0]:
        wp.atomic_max(error, 0, 1)
        return
    for cell in range(lane, count[0], lanes):
        if head[cell] < 0:
            continue
        row, col = ids[cell][a], ids[cell][b]
        if (owner[row] >= 0 or mass[row] > 0.) and (owner[col] >= 0 or mass[col] > 0.):
            block = wp.mat33d(0.)
            nonzero = bool(False)
            for i in range(3):
                for j in range(3):
                    block[i, j] = h[cell, 3*a+i, 3*b+j]
                    nonzero = nonzero or block[i, j] != wp.float64(0.)
            if nonzero:
                nr, nc = int(1), int(1)
                if owner[row] >= 0: nr = 4
                if owner[col] >= 0: nc = 4
                for i in range(nr):
                    rr = coordinate(row, i, owner, n)
                    rw = coefficient(row, i, owner, weights)
                    for j in range(nc):
                        cc = coordinate(col, j, owner, n)
                        cw = coefficient(col, j, owner, weights)
                        # Exact zero only; no small-value truncation.
                        if rw*cw != wp.float64(0.):
                            slot = slot_of(rr, cc, offsets, columns)
                            if slot >= 0:
                                wp.atomic_add(values, slot, (rw*cw)*block)
                            else:
                                wp.atomic_max(error, 0, 2)


def add_surface(matrix, surface, coupled):
    for frame in ((surface, surface.frame) if surface.mu else (surface,)):
        lanes = min(frame.cell_capacity, 128)
        wp.launch(assemble, (lanes, 6, 6), [
            frame.cell_ids, frame.head, frame.cell_h, frame.pair_count, frame.invalid,
            coupled.owner, coupled.weights, coupled.s.model.particle_mass, coupled.n,
            matrix.offsets, matrix.columns, matrix.values, matrix.surface_error, lanes,
        ], device=matrix.device)
