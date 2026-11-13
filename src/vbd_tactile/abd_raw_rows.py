"""Elementwise assembly of the exact affine shape and inertia tangent.

The 144 entries per body are independent. Keep the matrix products and their
evaluation order identical to the scalar reference in abd_coupled_tangent;
only the outer entry loops are distributed over CUDA threads.
"""
import warp as wp
from .abd_gpu import affine

wp.set_module_options({'enable_backward': False})


@wp.kernel
def body_hessian_rows(u: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64),
                      mass: wp.array(dtype=wp.mat44d), kv: wp.array(dtype=wp.float64),
                      dt: wp.float64, n: int, h: wp.array3d(dtype=wp.float64),
                      diagonal: wp.array3d(dtype=wp.float64)):
    b, p, q = wp.tid()
    value = wp.float64(0.)
    if p % 3 == q % 3:
        value = mass[b][p // 3, q // 3] / (dt * dt)
    if p >= 3 and q >= 3:
        r = radius[b]
        a = affine(u, n + 4 * b, r)
        c = wp.transpose(a) * a - wp.identity(n=3, dtype=wp.float64)
        aat = a * wp.transpose(a)
        i, j = p % 3, q % 3
        col, other = p // 3 - 1, q // 3 - 1
        shape = a[i, other] * a[j, col]
        if i == j:
            shape += c[col, other]
        if col == other:
            shape += aat[i, j]
        value += wp.float64(4.) * kv[b] / (r * r) * shape
    h[b, p, q] = value
    diagonal[b, p, q] = value
