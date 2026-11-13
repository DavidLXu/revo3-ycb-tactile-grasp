"""Float64 device math for the existing unified IPC potential.

Independent analytical implementation. IPC physical area scaling and EE
mollification follow IPC Toolkit 1.6.0 (MIT); no host contact derivatives.
"""

import warp as wp
from .ipc_primitives_warp import _weights, _delta

wp.set_module_options({"enable_backward": False})
mat12 = wp.types.matrix((12, 12), wp.float64)
mat9 = wp.types.matrix((9, 9), wp.float64)
vec12 = wp.types.vector(12, wp.float64)
vec9 = wp.types.vector(9, wp.float64)


@wp.func
def barrier(d: wp.float64, dhat: wp.float64):
    e, first, second = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    if d <= wp.float64(0.0):
        e = wp.float64(wp.inf)
    elif d < dhat:
        a = d - dhat
        log = wp.log(d / dhat)
        e = -a * a * log
        first = -wp.float64(2.0) * a * log - a * a / d
        second = -wp.float64(2.0) * log - wp.float64(4.0) * a / d + a * a / (d * d)
    return e, first, second


@wp.func
def squared_distance(
    a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, kind: int, need_hessian: bool
):
    w = _weights(a, b, c, d, kind)
    r = _delta(w, a, b, c, d)
    q = wp.dot(r, r)
    g = vec12(0.0)
    h = mat12(0.0)
    for i in range(4):
        for k in range(3):
            g[3 * i + k] = wp.float64(2.0) * w[i] * r[k]
    if need_hessian:
        # Eliminate the active closest-point coordinates with a Schur complement.
        u, v = wp.vec4d(0.0), wp.vec4d(0.0)
        rank = int(0)
        if kind == 0:
            if (
                w[1] != wp.float64(0.0)
                and w[2] != wp.float64(0.0)
                and w[3] != wp.float64(0.0)
            ):
                u = wp.vec4d(0.0, 1.0, -1.0, 0.0)
                v = wp.vec4d(0.0, 1.0, 0.0, -1.0)
                rank = 2
            else:
                lo, hi = int(-1), int(-1)
                for i in range(1, 4):
                    if w[i] != wp.float64(0.0):
                        if lo < 0:
                            lo = i
                        else:
                            hi = i
                if hi >= 0:
                    u[lo] = wp.float64(1.0)
                    u[hi] = wp.float64(-1.0)
                    rank = 1
        else:
            if w[0] != wp.float64(0.0) and w[1] != wp.float64(0.0):
                u = wp.vec4d(-1.0, 1.0, 0.0, 0.0)
                rank = 1
            if w[2] != wp.float64(0.0) and w[3] != wp.float64(0.0):
                if rank == 1:
                    v = wp.vec4d(0.0, 0.0, 1.0, -1.0)
                    rank = 2
                else:
                    u = wp.vec4d(0.0, 0.0, 1.0, -1.0)
                    rank = 1
        bu, bv = _delta(u, a, b, c, d), _delta(v, a, b, c, d)
        iu, iv, iuv = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
        if rank == 1:
            iu = wp.float64(0.5) / wp.dot(bu, bu)
        elif rank == 2:
            cross = wp.cross(bu, bv)
            det = wp.dot(cross, cross)
            iu = wp.float64(0.5) * wp.dot(bv, bv) / det
            iv = wp.float64(0.5) * wp.dot(bu, bu) / det
            iuv = -wp.float64(0.5) * wp.dot(bu, bv) / det
        cu, cv = vec12(0.0), vec12(0.0)
        for i in range(4):
            for k in range(3):
                cu[3 * i + k] = wp.float64(2.0) * (bu[k] * w[i] + u[i] * r[k])
                cv[3 * i + k] = wp.float64(2.0) * (bv[k] * w[i] + v[i] * r[k])
        for i in range(12):
            for j in range(12):
                value = (
                    -iu * cu[i] * cu[j]
                    - iv * cv[i] * cv[j]
                    - iuv * (cu[i] * cv[j] + cv[i] * cu[j])
                )
                if i % 3 == j % 3:
                    value += wp.float64(2.0) * w[i // 3] * w[j // 3]
                h[i, j] = value
    return q, g, h, w


@wp.func
def mollifier(
    a: wp.vec3d,
    b: wp.vec3d,
    c: wp.vec3d,
    d: wp.vec3d,
    eps: wp.float64,
    need_hessian: bool,
):
    value = wp.float64(1.0)
    g, h = vec12(0.0), mat12(0.0)
    e, f = b - a, d - c
    n = wp.cross(e, f)
    s = wp.dot(n, n)
    if eps > wp.float64(0.0) and s < eps:
        t = s / eps
        value = (wp.float64(2.0) - t) * t
        first = wp.float64(2.0) * (wp.float64(1.0) - t) / eps
        second = -wp.float64(2.0) / (eps * eps)
        ge, gf = wp.float64(2.0) * wp.cross(f, n), wp.float64(2.0) * wp.cross(n, e)
        signs = wp.vec4d(-1.0, 1.0, -1.0, 1.0)
        gs = vec12(0.0)
        for i in range(4):
            for k in range(3):
                component = gf[k]
                if i < 2:
                    component = ge[k]
                gs[3 * i + k] = signs[i] * component
                g[3 * i + k] = first * gs[3 * i + k]
        if need_hessian:
            ee, ff, ef = wp.dot(e, e), wp.dot(f, f), wp.dot(e, f)
            for i in range(12):
                for j in range(12):
                    u, v = i % 3, j % 3
                    delta = wp.float64(0.0)
                    if u == v:
                        delta = wp.float64(1.0)
                    hs = wp.float64(0.0)
                    if i < 6 and j < 6:
                        hs = wp.float64(2.0) * (ff * delta - f[u] * f[v])
                    elif i >= 6 and j >= 6:
                        hs = wp.float64(2.0) * (ee * delta - e[u] * e[v])
                    elif i < 6:
                        hs = wp.float64(4.0) * e[u] * f[v] - wp.float64(2.0) * (
                            f[u] * e[v] + ef * delta
                        )
                    else:
                        hs = wp.float64(4.0) * e[v] * f[u] - wp.float64(2.0) * (
                            f[v] * e[u] + ef * delta
                        )
                    hs *= signs[i // 3] * signs[j // 3]
                    h[i, j] = first * hs + second * gs[i] * gs[j]
    return value, g, h


@wp.func
def contact_terms(
    a: wp.vec3d,
    b: wp.vec3d,
    c: wp.vec3d,
    d: wp.vec3d,
    kind: int,
    dhat: wp.float64,
    dmin: wp.float64,
    stiffness: wp.float64,
    area: wp.float64,
    sphere_offset: wp.float64,
    eps: wp.float64,
    need_hessian: bool,
):
    q, gq, hq, w = squared_distance(a, b, c, d, kind, need_hessian)
    dist = wp.sqrt(q)
    energy, first, second = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    if sphere_offset > wp.float64(0.0):
        gap = dist - sphere_offset
        energy, first, second = barrier(gap, dhat)
        # Sphere uses linear surface gap and rest area/dhat^2, as the CPU model.
        scale = stiffness * area / (dhat * dhat)
        if gap <= dmin:
            energy = wp.float64(wp.inf)
        if dist > wp.float64(0.0):
            second = second / (wp.float64(4.0) * q) - first / (
                wp.float64(4.0) * q * dist
            )
            first = first / (wp.float64(2.0) * dist)
        energy *= scale
        first *= scale
        second *= scale
    else:
        threshold = (wp.float64(2.0) * dmin + dhat) * dhat
        energy, first, second = barrier(q - dmin * dmin, threshold)
        scale = stiffness * area * dhat / (threshold * threshold)
        energy *= scale
        first *= scale
        second *= scale
    g = first * gq
    h = mat12(0.0)
    if need_hessian:
        h = second * wp.outer(gq, gq) + first * hq
    if kind != 0:
        m, gm, hm = mollifier(a, b, c, d, eps, need_hessian)
        if need_hessian:
            h = m * h + energy * hm + wp.outer(g, gm) + wp.outer(gm, g)
        g = m * g + energy * gm
        energy *= m
    return energy, g, h, w


@wp.func
def friction_terms(
    displacement: wp.vec3d, normal: wp.vec3d, scale: wp.float64, eps: wp.float64
):
    tangent = displacement - normal * wp.dot(normal, displacement)
    r = wp.length(tangent)
    e, first_over_r, second = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    if r < eps:
        e = r * r / eps - r * r * r / (wp.float64(3.0) * eps * eps)
        first_over_r = wp.float64(2.0) / eps - r / (eps * eps)
        second = wp.float64(2.0) / eps - wp.float64(2.0) * r / (eps * eps)
    else:
        e = r - eps / wp.float64(3.0)
        first_over_r = wp.float64(1.0) / r
    unit = wp.vec3d(0.0)
    if r > wp.float64(0.0):
        unit = tangent / r
    h = scale * (
        first_over_r * (wp.identity(n=3, dtype=wp.float64) - wp.outer(normal, normal))
        + (second - first_over_r) * wp.outer(unit, unit)
    )
    return scale * e, scale * first_over_r * tangent, h


@wp.func
def psd9(input_h: mat9):
    """Cyclic Jacobi eigensystem; positive reconstruction is PSD even at the iteration cap."""
    a = (input_h + wp.transpose(input_h)) * wp.float64(0.5)
    vectors = wp.identity(n=9, dtype=wp.float64)
    for sweep in range(16):
        scale, off = wp.float64(0.0), wp.float64(0.0)
        for i in range(9):
            scale = wp.max(scale, wp.abs(a[i, i]))
            for j in range(i + 1, 9):
                off = wp.max(off, wp.abs(a[i, j]))
        if off <= wp.float64(1e-13) * wp.max(scale, wp.float64(1e-30)):
            break
        for p in range(8):
            for q in range(p + 1, 9):
                apq = a[p, q]
                if wp.abs(apq) > wp.float64(1e-15) * wp.max(scale, wp.float64(1e-30)):
                    tau = (a[q, q] - a[p, p]) / (wp.float64(2.0) * apq)
                    sign = wp.float64(1.0)
                    if tau < wp.float64(0.0):
                        sign = wp.float64(-1.0)
                    t = sign / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
                    c = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
                    s = t * c
                    a[p, p] = a[p, p] - t * apq
                    a[q, q] = a[q, q] + t * apq
                    a[p, q] = wp.float64(0.0)
                    a[q, p] = wp.float64(0.0)
                    for k in range(9):
                        if k != p and k != q:
                            kp, kq = a[k, p], a[k, q]
                            a[k, p] = c * kp - s * kq
                            a[p, k] = a[k, p]
                            a[k, q] = s * kp + c * kq
                            a[q, k] = a[k, q]
                        vp, vq = vectors[k, p], vectors[k, q]
                        vectors[k, p] = c * vp - s * vq
                        vectors[k, q] = s * vp + c * vq
    result = mat9(0.0)
    for i in range(9):
        for j in range(9):
            value = wp.float64(0.0)
            for k in range(9):
                value += (
                    wp.max(a[k, k], wp.float64(0.0)) * vectors[i, k] * vectors[j, k]
                )
            result[i, j] = value
    return result


@wp.func
def psd12(input_h: mat12):
    """Cyclic Jacobi eigensystem; positive reconstruction is PSD even at the iteration cap."""
    a = (input_h + wp.transpose(input_h)) * wp.float64(0.5)
    vectors = wp.identity(n=12, dtype=wp.float64)
    for sweep in range(16):
        scale, off = wp.float64(0.0), wp.float64(0.0)
        for i in range(12):
            scale = wp.max(scale, wp.abs(a[i, i]))
            for j in range(i + 1, 12):
                off = wp.max(off, wp.abs(a[i, j]))
        if off <= wp.float64(1e-13) * wp.max(scale, wp.float64(1e-30)):
            break
        for p in range(11):
            for q in range(p + 1, 12):
                apq = a[p, q]
                if wp.abs(apq) > wp.float64(1e-15) * wp.max(scale, wp.float64(1e-30)):
                    tau = (a[q, q] - a[p, p]) / (wp.float64(2.0) * apq)
                    sign = wp.float64(1.0)
                    if tau < wp.float64(0.0):
                        sign = wp.float64(-1.0)
                    t = sign / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
                    c = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
                    s = t * c
                    a[p, p] = a[p, p] - t * apq
                    a[q, q] = a[q, q] + t * apq
                    a[p, q] = wp.float64(0.0)
                    a[q, p] = wp.float64(0.0)
                    for k in range(12):
                        if k != p and k != q:
                            kp, kq = a[k, p], a[k, q]
                            a[k, p] = c * kp - s * kq
                            a[p, k] = a[k, p]
                            a[k, q] = s * kp + c * kq
                            a[q, k] = a[k, q]
                        vp, vq = vectors[k, p], vectors[k, q]
                        vectors[k, p] = c * vp - s * vq
                        vectors[k, q] = s * vp + c * vq
    result = mat12(0.0)
    for i in range(12):
        for j in range(12):
            value = wp.float64(0.0)
            for k in range(12):
                value += (
                    wp.max(a[k, k], wp.float64(0.0)) * vectors[i, k] * vectors[j, k]
                )
            result[i, j] = value
    return result
