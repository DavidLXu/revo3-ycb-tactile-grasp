"""Spectral Stable Neo-Hookean Hessian, in row-major deformation coordinates.

The six twist/flip modes are analytical; only the 3x3 stretch block needs an
eigensolve. This is the SNH decomposition used by GIPC, independently expressed
for this solver's shifted Lame parameter and existing strain-rate damping.
"""
import warp as wp
from .ipc_gpu_math import mat9, mat12

wp.set_module_options({"enable_backward": False})
mat43 = wp.types.matrix((4, 3), wp.float64)


@wp.func
def eigen3(a_in: wp.mat33d):
    # Cyclic Jacobi on a small symmetric matrix. Unlike a fixed approximate
    # quaternion sweep, the stopping criterion is relative to float64 scale.
    a = (a_in + wp.transpose(a_in)) * wp.float64(0.5)
    q = wp.identity(n=3, dtype=wp.float64)
    for sweep in range(16):
        scale = wp.max(wp.abs(a[0, 0]), wp.max(wp.abs(a[1, 1]), wp.abs(a[2, 2])))
        off = wp.max(wp.abs(a[0, 1]), wp.max(wp.abs(a[0, 2]), wp.abs(a[1, 2])))
        if off <= wp.float64(2e-15) * wp.max(scale, wp.float64(1e-300)):
            break
        for p in range(2):
            for r in range(p + 1, 3):
                ar = a[p, r]
                if wp.abs(ar) > wp.float64(1e-16) * wp.max(scale, wp.float64(1e-300)):
                    tau = (a[r, r] - a[p, p]) / (wp.float64(2.) * ar)
                    sign = wp.float64(1.)
                    if tau < wp.float64(0.):
                        sign = wp.float64(-1.)
                    t = sign / (wp.abs(tau) + wp.sqrt(wp.float64(1.) + tau * tau))
                    c = wp.float64(1.) / wp.sqrt(wp.float64(1.) + t * t)
                    s = t * c
                    a[p, p] -= t * ar
                    a[r, r] += t * ar
                    a[p, r] = wp.float64(0.)
                    a[r, p] = wp.float64(0.)
                    for k in range(3):
                        if k != p and k != r:
                            kp, kr = a[k, p], a[k, r]
                            a[k, p] = c * kp - s * kr
                            a[p, k] = a[k, p]
                            a[k, r] = s * kp + c * kr
                            a[r, k] = a[k, r]
                        qp, qr = q[k, p], q[k, r]
                        q[k, p] = c * qp - s * qr
                        q[k, r] = s * qp + c * qr
    return wp.vec3d(a[0, 0], a[1, 1], a[2, 2]), q


@wp.func
def add_mode(h: mat9, mode: wp.mat33d, eigenvalue: wp.float64):
    value = wp.max(eigenvalue, wp.float64(0.))
    out = h
    for p in range(9):
        for r in range(9):
            out[p, r] += value * mode[p // 3, p % 3] * mode[r // 3, r % 3]
    return out


@wp.func
def elastic_psd(f: wp.mat33d, mu: wp.float64, lam: wp.float64):
    u, sigma, v = wp.svd3(f)
    j = wp.determinant(f)
    c = lam * (j - wp.float64(1.)) - mu
    h = mat9(0.)
    invsqrt2 = wp.float64(0.7071067811865475244)
    for k in range(3):
        a, b = (k + 1) % 3, (k + 2) % 3
        ua = wp.vec3d(u[0, a], u[1, a], u[2, a])
        ub = wp.vec3d(u[0, b], u[1, b], u[2, b])
        va = wp.vec3d(v[0, a], v[1, a], v[2, a])
        vb = wp.vec3d(v[0, b], v[1, b], v[2, b])
        ab, ba = wp.outer(ua, vb), wp.outer(ub, va)
        h = add_mode(h, (ab - ba) * invsqrt2, mu + c * sigma[k])
        h = add_mode(h, (ab + ba) * invsqrt2, mu - c * sigma[k])
    stretch = wp.mat33d(0.)
    for k in range(3):
        a, b = (k + 1) % 3, (k + 2) % 3
        product = sigma[a] * sigma[b]
        stretch[k, k] = mu + lam * product * product
        stretch[a, b] = (lam * (wp.float64(2.) * j - wp.float64(1.)) - mu) * sigma[k]
        stretch[b, a] = stretch[a, b]
    values, vectors = eigen3(stretch)
    for k in range(3):
        mode = wp.mat33d(0.)
        for a in range(3):
            for b in range(3):
                for r in range(3):
                    mode[a, b] += u[a, r] * vectors[r, k] * v[b, r]
        h = add_mode(h, mode, values[k])
    return h


@wp.func
def tet_hessian_spectral(f: wp.mat33d, shape: mat43, volume: wp.float64,
                         mu: wp.float64, lam: wp.float64, damp: wp.float64, dt: wp.float64):
    hf = elastic_psd(f, mu, lam)
    # Closed-form J_C^T J_C for C=F^T F. Identical to the old nested sums.
    fft = f * wp.transpose(f)
    for p in range(9):
        i, k = p // 3, p % 3
        for q in range(9):
            a, b = q // 3, q % 3
            value = f[i, b] * f[a, k]
            if k == b:
                value += fft[i, a]
            hf[p, q] += wp.float64(2.) * damp / dt * value
    h = mat12(0.)
    for p in range(12):
        a, i = p // 3, p % 3
        for q in range(12):
            b, j = q // 3, q % 3
            value = wp.float64(0.)
            for k in range(3):
                for l in range(3):
                    value += shape[a, k] * hf[3*i+k, 3*j+l] * shape[b, l]
            h[p, q] = volume * value
    return h
