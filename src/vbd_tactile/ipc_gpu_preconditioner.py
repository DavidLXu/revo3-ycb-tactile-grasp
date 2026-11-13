"""Scaled Cholesky inversion for positive definite 3x3 IPC diagonal blocks."""
import warp as wp
wp.set_module_options({"enable_backward": False})


@wp.kernel
def diagonal_inverse(diagonal: wp.array(dtype=wp.mat33d),
                     inverse: wp.array(dtype=wp.mat33d),
                     stats: wp.array(dtype=wp.float64)):
    i = wp.tid()
    d = diagonal[i]
    scale = wp.float64(0)
    finite = True
    for row in range(3):
        for col in range(3):
            finite = finite and wp.isfinite(d[row, col])
            scale = wp.max(scale, wp.abs(d[row, col]))
    result = wp.mat33d(0)
    valid = False
    if finite and scale > wp.float64(0):
        # A Hessian block is symmetric; average roundoff from the two scatter
        # orders. Scaling avoids overflow in the factorization and inverse.
        a = (d / scale + wp.transpose(d) / scale) * wp.float64(.5)
        if a[0, 0] > wp.float64(0):
            l00 = wp.sqrt(a[0, 0])
            l10, l20 = a[1, 0] / l00, a[2, 0] / l00
            p11 = a[1, 1] - l10 * l10
            if p11 > wp.float64(0):
                l11 = wp.sqrt(p11)
                l21 = (a[2, 1] - l20 * l10) / l11
                p22 = a[2, 2] - l20 * l20 - l21 * l21
                if p22 > wp.float64(0):
                    l22 = wp.sqrt(p22)
                    r00, r11, r22 = wp.float64(1) / l00, wp.float64(1) / l11, wp.float64(1) / l22
                    r10 = -l10 * r00 / l11
                    r20 = -(l20 * r00 + l21 * r10) / l22
                    r21 = -l21 * r11 / l22
                    r = wp.mat33d(r00, 0., 0., r10, r11, 0., r20, r21, r22)
                    result = (wp.transpose(r) * r) / scale
                    valid = True
                    for row in range(3):
                        for col in range(3):
                            valid = valid and wp.isfinite(result[row, col])
    if not valid:
        wp.atomic_add(stats, 2, wp.float64(1))
    inverse[i] = result
