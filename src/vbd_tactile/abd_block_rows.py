"""Scalar row access to small dense ABD blocks, without per-thread mat12 copies.

These kernels change storage access and thread assignment only. They retain
all coefficients, FP64 arithmetic and the original column summation order.
"""
import warp as wp

wp.set_module_options({'enable_backward': False})


@wp.kernel
def body_product(h: wp.array3d(dtype=wp.float64), x: wp.array(dtype=wp.vec3d),
                 n: int, add: bool, out: wp.array2d(dtype=wp.float64)):
    body, row = wp.tid()
    value = wp.float64(0.)
    for col in range(4):
        for j in range(3):
            value += h[body, row, 3*col+j]*x[n+4*body+col][j]
    node, component = n+4*body+row//3, row%3
    if add:
        value += out[node, component]
    out[node, component] = value


@wp.kernel
def finish_precondition(x: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d),
        inverse: wp.array(dtype=wp.mat33d), body_work: wp.array(dtype=wp.vec3d),
        mass: wp.array(dtype=float), n: int, alpha: wp.float64, beta: wp.float64,
        out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    value = wp.vec3d(0.)
    if i < n:
        value = x[i]
        if mass[i] > 0.:
            value = inverse[i]*x[i]
    else:
        value = body_work[i]
    out[i] = alpha*value+beta*y[i]


@wp.kernel
def copy_bodies(h: wp.array3d(dtype=wp.float64), slots: wp.array3d(dtype=int),
                out: wp.array(dtype=wp.mat33d)):
    body, i, j = wp.tid()
    value = wp.mat33d(0.)
    for k in range(3):
        for l in range(3):
            value[k,l] = h[body,3*i+k,3*j+l]
    out[slots[body,i,j]] = value


@wp.kernel
def mark_blocks(rows: wp.array(dtype=int), columns: wp.array(dtype=int),
                values: wp.array2d(dtype=wp.float64), used: wp.array(dtype=int)):
    p = wp.tid()
    keep = rows[p] == columns[p]
    for k in range(9):
        keep = keep or values[p,k] != wp.float64(0.)
    used[p] = int(keep)


@wp.kernel
def pack_marked(offsets: wp.array(dtype=int), columns: wp.array(dtype=int),
        values: wp.array(dtype=wp.mat33d), used: wp.array(dtype=int),
        counts: wp.array(dtype=int), packed_columns: wp.array(dtype=int),
        packed_values: wp.array(dtype=wp.mat33d)):
    row = wp.tid(); start = offsets[row]; size = int(0)
    for p in range(start,offsets[row+1]):
        if used[p] != 0:
            packed_columns[start+size] = columns[p]
            packed_values[start+size] = values[p]
            size += 1
    counts[row] = size
