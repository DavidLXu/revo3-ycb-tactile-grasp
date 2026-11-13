"""GPU assembly and matrix-free products for unified IPC (all arrays device-resident)."""

from typing import Any
from .ipc_precision import position_overloads, pose_value
import warp as wp
from .ipc_gpu_math import (
    mat12,
    mat9,
    vec12,
    contact_terms,
    friction_terms,
    squared_distance,
    barrier,
    psd9,
    psd12,
)
from .ipc_primitives_warp import _delta, _finite, _weights

wp.set_module_options({"enable_backward": False})
mat43 = wp.types.matrix((4, 3), wp.float64)


@position_overloads("old", poses=('center', 'displacement'))
@wp.kernel
def sphere_seed_face_weights(
    old: wp.array(dtype=Any),
    faces: wp.array2d(dtype=int),
    center: Any,
    displacement: Any,
    offset: wp.float64,
    dhat: wp.float64,
    min_gap: wp.float64,
    weights: wp.array(dtype=wp.float64),
):
    """Translate vertices of near-contact FACES when seeding a sphere step.

    A coarse face may contact the sphere while all three vertices are far away.
    Vertex-distance-only seeding misses that contact and can start outside the
    barrier domain. These weights affect only the optimizer's initial guess.
    """
    _center = pose_value(center)
    _displacement = pose_value(displacement)
    f = wp.tid()
    a, b, c = wp.vec3d(old[faces[f, 0]]), wp.vec3d(old[faces[f, 1]]), wp.vec3d(old[faces[f, 2]])
    w = _weights(_center, a, b, c, 0)
    gap = wp.length(_delta(w, _center, a, b, c)) - offset
    inner = min_gap + wp.float64(2.0) * wp.length(_displacement)
    weight = wp.clamp((inner + wp.float64(2.0) * dhat - gap) / (wp.float64(2.0) * dhat),
                      wp.float64(0.0), wp.float64(1.0))
    for j in range(3):
        wp.atomic_max(weights, faces[f, j], weight)


@position_overloads("old", "positions", poses=('center', 'displacement'))
@wp.kernel
def sphere_seed_apply(
    old: wp.array(dtype=Any),
    mass: wp.array(dtype=float),
    center: Any,
    displacement: Any,
    offset: wp.float64,
    dhat: wp.float64,
    min_gap: wp.float64,
    weights: wp.array(dtype=wp.float64),
    positions: wp.array(dtype=Any),
):
    _center = pose_value(center)
    _displacement = pose_value(displacement)
    i = wp.tid()
    if mass[i] > 0.0:
        x = wp.vec3d(old[i])
        gap = wp.length(x - _center) - offset
        inner = min_gap + wp.float64(2.0) * wp.length(_displacement)
        vertex_weight = wp.clamp((inner + wp.float64(2.0) * dhat - gap) / (wp.float64(2.0) * dhat),
                                 wp.float64(0.0), wp.float64(1.0))
        positions[i] = type(old[i])(x + wp.max(vertex_weight, weights[i]) * _displacement)


@wp.func
def point(x: wp.array(dtype=Any), i: int, center: wp.vec3d):
    p = center
    if i >= 0:
        p = wp.vec3d(x[i])
    return p


@wp.func
def scatter(
    ids: wp.vec4i,
    g: vec12,
    energy: wp.float64,
    gradient: wp.array(dtype=wp.vec3d),
    force: wp.array(dtype=wp.vec3d),
    stats: wp.array(dtype=wp.float64),
    external: bool,
):
    if not wp.isfinite(energy):
        wp.atomic_add(stats, 2, wp.float64(1.0))
    wp.atomic_add(stats, 0, energy)
    for a in range(4):
        if ids[a] >= 0:
            v = wp.vec3d(g[3 * a], g[3 * a + 1], g[3 * a + 2])
            if not _finite(v):
                wp.atomic_add(stats, 2, wp.float64(1.0))
            wp.atomic_add(gradient, ids[a], v)
            if external:
                wp.atomic_add(force, ids[a], -v)


@wp.func
def signed_eps(i: int, j: int, k: int):
    v = wp.float64(0.0)
    if i != j and j != k and k != i:
        v = wp.float64(1.0)
        if (i - j) * (j - k) * (k - i) < 0:
            v = wp.float64(-1.0)
    return v


@position_overloads("x")
@wp.kernel
def inertia(
    x: wp.array(dtype=Any),
    predictor: wp.array(dtype=wp.vec3d),
    mass: wp.array(dtype=float),
    dt: wp.float64,
    gradient: wp.array(dtype=wp.vec3d),
    stats: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    dx = wp.vec3d(x[i]) - predictor[i]
    scale = wp.float64(mass[i]) / (dt * dt)
    gradient[i] = scale * dx
    energy = wp.float64(0.5) * scale * wp.dot(dx, dx)
    wp.atomic_add(stats, 0, energy)
    if not wp.isfinite(energy) or not _finite(gradient[i]):
        wp.atomic_add(stats, 2, wp.float64(1.0))


@wp.func
def tet_hessian(f: wp.mat33d, co: wp.mat33d, j: wp.float64, shape: mat43,
                volume: wp.float64, mu: wp.float64, lam: wp.float64,
                alpha: wp.float64, damp: wp.float64, dt: wp.float64):
    hf = mat9(0.0)
    for p in range(9):
        i, k = p // 3, p % 3
        for q in range(9):
            a, b = q // 3, q % 3
            hj = wp.float64(0.0)
            for u in range(3):
                for v in range(3):
                    hj += signed_eps(i, a, u) * signed_eps(k, b, v) * f[u, v]
            value = lam * co[i, k] * co[a, b] + lam * (j - alpha) * hj
            if p == q:
                value += mu
            hf[p, q] = value
    hf = psd9(hf)
    # PSD Gauss-Newton Hessian of the same exact strain-rate energy.
    for p in range(9):
        i, k = p // 3, p % 3
        for q in range(9):
            a, b = q // 3, q % 3
            value = wp.float64(0.0)
            for u in range(3):
                for v in range(3):
                    dp, dq = wp.float64(0.0), wp.float64(0.0)
                    if v == k:
                        dp += f[i, u]
                    if u == k:
                        dp += f[i, v]
                    if v == b:
                        dq += f[a, u]
                    if u == b:
                        dq += f[a, v]
                    value += dp * dq
            hf[p, q] += damp / dt * value
    h = mat12(0.0)
    for p in range(12):
        a, i = p // 3, p % 3
        for q in range(12):
            b, jj = q // 3, q % 3
            value = wp.float64(0.0)
            for k in range(3):
                for l in range(3):
                    value += shape[a, k] * hf[3 * i + k, 3 * jj + l] * shape[b, l]
            h[p, q] = volume * value
    return h


@position_overloads("x", "old")
@wp.kernel
def tetrahedra(
    x: wp.array(dtype=Any),
    old: wp.array(dtype=Any),
    ids: wp.array(dtype=wp.vec4i),
    poses: wp.array(dtype=wp.mat33),
    materials: wp.array2d(dtype=float),
    dt: wp.float64,
    need_hessian: bool,
    gradient: wp.array(dtype=wp.vec3d),
    stats: wp.array(dtype=wp.float64),
    hessians: wp.array(dtype=mat12),
):
    t = wp.tid()
    ind = ids[t]
    dm = wp.mat33d(poses[t])
    ds = wp.matrix_from_cols(
        wp.vec3d(x[ind[1]]) - wp.vec3d(x[ind[0]]),
        wp.vec3d(x[ind[2]]) - wp.vec3d(x[ind[0]]),
        wp.vec3d(x[ind[3]]) - wp.vec3d(x[ind[0]]),
    )
    ds0 = wp.matrix_from_cols(
        wp.vec3d(old[ind[1]]) - wp.vec3d(old[ind[0]]),
        wp.vec3d(old[ind[2]]) - wp.vec3d(old[ind[0]]),
        wp.vec3d(old[ind[3]]) - wp.vec3d(old[ind[0]]),
    )
    f, f0 = ds * dm, ds0 * dm
    mu = wp.float64(materials[t, 0])
    lam = wp.float64(materials[t, 1]) + mu
    damp = wp.float64(materials[t, 2])
    alpha = wp.float64(1.0) + mu / lam
    j = wp.determinant(f)
    volume = wp.float64(1.0) / (wp.float64(6.0) * wp.determinant(dm))
    ft = wp.transpose(f)
    co = wp.matrix_from_cols(
        wp.cross(ft[1], ft[2]), wp.cross(ft[2], ft[0]), wp.cross(ft[0], ft[1])
    )
    strain = wp.transpose(f) * f - wp.transpose(f0) * f0
    fnorm, cnorm = wp.float64(0.0), wp.float64(0.0)
    for i in range(3):
        for k in range(3):
            fnorm += f[i, k] * f[i, k]
            cnorm += strain[i, k] * strain[i, k]
    energy = volume * (
        wp.float64(0.5) * mu * (fnorm - wp.float64(3.0))
        + wp.float64(0.5)
        * lam
        * (
            (j - alpha) * (j - alpha)
            - (wp.float64(1.0) - alpha) * (wp.float64(1.0) - alpha)
        )
        + damp / (wp.float64(2.0) * dt) * cnorm
    )
    stress = (
        mu * f + lam * (j - alpha) * co + wp.float64(2.0) * damp / dt * (f * strain)
    )
    shape = mat43(0.0)
    for k in range(3):
        shape[0, k] = -(dm[0, k] + dm[1, k] + dm[2, k])
        for a in range(1, 4):
            shape[a, k] = dm[a - 1, k]
    g = vec12(0.0)
    for a in range(4):
        grad = stress * wp.vec3d(shape[a, 0], shape[a, 1], shape[a, 2]) * volume
        for k in range(3):
            g[3 * a + k] = grad[k]
    scatter(ind, g, energy, gradient, gradient, stats, False)
    if need_hessian:
        hessians[t] = tet_hessian(f, co, j, shape, volume, mu, lam, alpha, damp, dt)


@position_overloads("x", poses=('center',))
@wp.kernel
def freeze_contact(
    x: wp.array(dtype=Any),
    ids: wp.array(dtype=wp.vec4i),
    kinds: wp.array(dtype=int),
    areas: wp.array(dtype=wp.float64),
    moll_eps: wp.array(dtype=wp.float64),
    count: wp.array(dtype=int),
    center: Any,
    dhat: wp.float64,
    dmin: wp.float64,
    stiffness: wp.float64,
    offset: wp.float64,
    mu: wp.float64,
    weights: wp.array(dtype=wp.vec4d),
    normals: wp.array(dtype=wp.vec3d),
    loads: wp.array(dtype=wp.float64),
):
    _center = pose_value(center)
    i = wp.tid()
    if i < count[0]:
        ind = ids[i]
        a, b, c, d = (
            point(x, ind[0], _center),
            point(x, ind[1], _center),
            point(x, ind[2], _center),
            point(x, ind[3], _center),
        )
        q, gq, hq, w = squared_distance(a, b, c, d, kinds[i], False)
        dist = wp.sqrt(q)
        load = wp.float64(0.0)
        if offset > wp.float64(0.0):
            e, first, second = barrier(dist - offset, dhat)
            load = -first * stiffness * areas[i] / (dhat * dhat)
        else:
            threshold = (wp.float64(2.0) * dmin + dhat) * dhat
            e, first, second = barrier(q - dmin * dmin, threshold)
            load = (
                -wp.float64(2.0)
                * dist
                * first
                * stiffness
                * areas[i]
                * dhat
                / (threshold * threshold)
            )
            if kinds[i] != 0:
                n = wp.cross(b - a, d - c)
                if wp.dot(n, n) < moll_eps[i]:
                    # Matches IPC Toolkit: omit lagged friction on mollified EE.
                    load = wp.float64(0.0)
        normal = wp.vec3d(0.0)
        if dist > wp.float64(0.0):
            normal = _delta(w, a, b, c, d) / dist
        weights[i] = w
        normals[i] = normal
        loads[i] = mu * load


@position_overloads("x", poses=('center',))
@wp.kernel
def normals(
    x: wp.array(dtype=Any),
    ids: wp.array(dtype=wp.vec4i),
    kinds: wp.array(dtype=int),
    areas: wp.array(dtype=wp.float64),
    moll_eps: wp.array(dtype=wp.float64),
    count: wp.array(dtype=int),
    center: Any,
    dhat: wp.float64,
    dmin: wp.float64,
    stiffness: wp.float64,
    offset: wp.float64,
    need_hessian: bool,
    gradient: wp.array(dtype=wp.vec3d),
    forces: wp.array(dtype=wp.vec3d),
    stats: wp.array(dtype=wp.float64),
    hessians: wp.array(dtype=mat12),
):
    _center = pose_value(center)
    i = wp.tid()
    if i < count[0]:
        ind = ids[i]
        a, b, c, d = (
            point(x, ind[0], _center),
            point(x, ind[1], _center),
            point(x, ind[2], _center),
            point(x, ind[3], _center),
        )
        # Exclude inactive sphere faces before the more expensive Hessian.
        q, gq, hq, w = squared_distance(a, b, c, d, kinds[i], False)
        active = wp.sqrt(q) < offset + dmin + dhat
        h = mat12(0.0)
        if active:
            e, g, h, w = contact_terms(
                a,
                b,
                c,
                d,
                kinds[i],
                dhat,
                dmin,
                stiffness,
                areas[i],
                offset,
                moll_eps[i],
                need_hessian,
            )
            scatter(ind, g, e, gradient, forces, stats, True)
            if need_hessian:
                h = psd12(h)
        if need_hessian:
            hessians[i] = h


@position_overloads("x", "old", poses=('center', 'old_center'))
@wp.kernel
def friction(
    x: wp.array(dtype=Any),
    old: wp.array(dtype=Any),
    ids: wp.array(dtype=wp.vec4i),
    count: wp.array(dtype=int),
    center: Any,
    old_center: Any,
    weights: wp.array(dtype=wp.vec4d),
    normals: wp.array(dtype=wp.vec3d),
    loads: wp.array(dtype=wp.float64),
    eps: wp.float64,
    toolkit_constant: bool,
    need_hessian: bool,
    gradient: wp.array(dtype=wp.vec3d),
    forces: wp.array(dtype=wp.vec3d),
    stats: wp.array(dtype=wp.float64),
    hessians: wp.array(dtype=mat12),
):
    _center = pose_value(center)
    _old_center = pose_value(old_center)
    i = wp.tid()
    if i < count[0]:
        h = mat12(0.0)
        if loads[i] != wp.float64(0.0):
            ind, w = ids[i], weights[i]
            slip = wp.vec3d(0.0)
            for a in range(4):
                slip += w[a] * (
                    point(x, ind[a], _center) - point(old, ind[a], _old_center)
                )
            e, gv, hv = friction_terms(slip, normals[i], loads[i], eps)
            if toolkit_constant:
                e += loads[i] * eps / wp.float64(3.0)
            g = vec12(0.0)
            for a in range(4):
                for k in range(3):
                    g[3 * a + k] = w[a] * gv[k]
            scatter(ind, g, e, gradient, forces, stats, True)
            if need_hessian and loads[i] > wp.float64(0.):
                for a in range(4):
                    for b in range(4):
                        for k in range(3):
                            for l in range(3):
                                h[3 * a + k, 3 * b + l] = w[a] * w[b] * hv[k, l]
        if need_hessian:
            hessians[i] = h


@wp.kernel
def residual(
    gradient: wp.array(dtype=wp.vec3d),
    mass: wp.array(dtype=float),
    stats: wp.array(dtype=wp.float64),
    rhs: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    rhs[i] = wp.vec3d(0.0)
    if mass[i] > 0.0:
        norm = wp.length(gradient[i])
        wp.atomic_max(stats, 1, norm)
        rhs[i] = -gradient[i]
        if not wp.isfinite(norm):
            wp.atomic_add(stats, 2, wp.float64(1.0))


@wp.kernel
def diagonal_init(
    mass: wp.array(dtype=float), dt: wp.float64, diagonal: wp.array(dtype=wp.mat33d)
):
    i = wp.tid()
    scale = wp.float64(1.0)
    if mass[i] > 0.0:
        scale = wp.float64(mass[i]) / (dt * dt)
    diagonal[i] = scale * wp.identity(n=3, dtype=wp.float64)


@wp.kernel
def diagonal_add(
    ids: wp.array(dtype=wp.vec4i),
    count: wp.array(dtype=int),
    hessians: wp.array(dtype=mat12),
    mass: wp.array(dtype=float),
    diagonal: wp.array(dtype=wp.mat33d),
):
    i = wp.tid()
    if i < count[0]:
        ind, h = ids[i], hessians[i]
        for a in range(4):
            j = ind[a]
            if j >= 0:
                if mass[j] > 0.0:
                    block = wp.mat33d(0.0)
                    for k in range(3):
                        for l in range(3):
                            block[k, l] = h[3 * a + k, 3 * a + l]
                    wp.atomic_add(diagonal, j, block)


@wp.kernel
def diagonal_inverse(
    diagonal: wp.array(dtype=wp.mat33d),
    inverse: wp.array(dtype=wp.mat33d),
    stats: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    d = diagonal[i]
    inv = wp.inverse(d)
    for k in range(3):
        for l in range(3):
            if not wp.isfinite(inv[k, l]):
                wp.atomic_add(stats, 2, wp.float64(1.0))
    if wp.determinant(d) <= wp.float64(0.0):
        wp.atomic_add(stats, 2, wp.float64(1.0))
    inverse[i] = inv


@wp.kernel
def mv_init(
    x: wp.array(dtype=wp.vec3d),
    mass: wp.array(dtype=float),
    dt: wp.array(dtype=wp.float64),
    result: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    scale = wp.float64(1.0)
    if mass[i] > 0.0:
        scale = wp.float64(mass[i]) / (dt[0] * dt[0])
    result[i] = scale * x[i]


@wp.kernel
def mv_add(
    x: wp.array(dtype=wp.vec3d),
    ids: wp.array(dtype=wp.vec4i),
    count: wp.array(dtype=int),
    hessians: wp.array(dtype=mat12),
    mass: wp.array(dtype=float),
    result: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    if i < count[0]:
        ind, h = ids[i], hessians[i]
        v = vec12(0.0)
        for a in range(4):
            j = ind[a]
            if j >= 0:
                if mass[j] > 0.0:
                    for k in range(3):
                        v[3 * a + k] = x[j][k]
        out = h * v
        for a in range(4):
            j = ind[a]
            if j >= 0:
                if mass[j] > 0.0:
                    wp.atomic_add(
                        result, j, wp.vec3d(out[3 * a], out[3 * a + 1], out[3 * a + 2])
                    )


@wp.kernel
def axpby(
    x: wp.array(dtype=wp.vec3d),
    y: wp.array(dtype=wp.vec3d),
    alpha: wp.float64,
    beta: wp.float64,
    z: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    value = alpha * x[i]
    if beta != wp.float64(0.0):
        value += beta * y[i]
    z[i] = value


@position_overloads("x", "out")
@wp.kernel
def trial(
    x: wp.array(dtype=Any),
    direction: wp.array(dtype=wp.vec3d),
    gradient: wp.array(dtype=wp.vec3d),
    mass: wp.array(dtype=float),
    alpha: wp.float64,
    out: wp.array(dtype=Any),
    stats: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    out[i] = x[i]
    if mass[i] > 0.0:
        out[i] = type(x[i])(wp.vec3d(x[i]) + alpha * direction[i])
    dx = wp.vec3d(out[i]) - wp.vec3d(x[i])
    wp.atomic_add(stats, 0, wp.dot(gradient[i], dx))
    wp.atomic_add(stats, 1, wp.dot(dx, dx))
    if not _finite(wp.vec3d(out[i])) or not _finite(direction[i]):
        wp.atomic_add(stats, 2, wp.float64(1.0))


@wp.kernel
def check_linear(
    rhs: wp.array(dtype=wp.vec3d),
    applied: wp.array(dtype=wp.vec3d),
    direction: wp.array(dtype=wp.vec3d),
    stats: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    r = rhs[i] - applied[i]
    wp.atomic_add(stats, 0, wp.dot(r, r))
    wp.atomic_add(stats, 1, wp.dot(rhs[i], rhs[i]))
    wp.atomic_add(stats, 2, -wp.dot(rhs[i], direction[i]))


@wp.kernel
def publish_force(
    force: wp.array(dtype=wp.vec3d),
    output: wp.array(dtype=wp.vec3),
    stats: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    f = wp.vec3(force[i])
    output[i] = f
    if not _finite(wp.vec3d(f)):
        wp.atomic_add(stats, 2, wp.float64(1.0))
    wp.atomic_max(stats, 1, wp.length(force[i]))
    for k in range(3):
        wp.atomic_add(stats, 3 + k, force[i][k])
