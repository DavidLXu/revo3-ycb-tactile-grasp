"""Hessian-only kernels for an already evaluated, immutable configuration.

Do not accumulate the gradient or forces again. The exact same material and
contact derivatives/PSD projection functions are shared with full evaluation.
"""
from typing import Any
import warp as wp

from .ipc_precision import position_overloads, pose_value
from .ipc_gpu_kernels import mat43, tet_hessian, point
from .ipc_gpu_math import mat12, contact_terms, squared_distance, friction_terms, psd12
from .ipc_gpu_material import tet_hessian_spectral

wp.set_module_options({"enable_backward": False})


@position_overloads("x")
@wp.kernel
def tetrahedra(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
               poses: wp.array(dtype=wp.mat33), materials: wp.array2d(dtype=float),
               dt: wp.float64, hessians: wp.array(dtype=mat12)):
    t = wp.tid()
    ind = ids[t]
    dm = wp.mat33d(poses[t])
    ds = wp.matrix_from_cols(wp.vec3d(x[ind[1]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[2]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[3]])-wp.vec3d(x[ind[0]]))
    f = ds*dm
    mu = wp.float64(materials[t, 0])
    lam = wp.float64(materials[t, 1])+mu
    damp = wp.float64(materials[t, 2])
    alpha = wp.float64(1.)+mu/lam
    volume = wp.float64(1.)/(wp.float64(6.)*wp.determinant(dm))
    ft = wp.transpose(f)
    co = wp.matrix_from_cols(wp.cross(ft[1], ft[2]), wp.cross(ft[2], ft[0]), wp.cross(ft[0], ft[1]))
    shape = mat43(0.)
    for k in range(3):
        shape[0, k] = -(dm[0, k]+dm[1, k]+dm[2, k])
        for a in range(1, 4):
            shape[a, k] = dm[a-1, k]
    hessians[t] = tet_hessian(f, co, wp.determinant(f), shape, volume, mu, lam, alpha, damp, dt)


@position_overloads("x")
@wp.kernel
def tetrahedra_spectral(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
                       poses: wp.array(dtype=wp.mat33), materials: wp.array2d(dtype=float),
                       dt: wp.float64, hessians: wp.array(dtype=mat12)):
    t = wp.tid()
    ind = ids[t]
    dm = wp.mat33d(poses[t])
    ds = wp.matrix_from_cols(wp.vec3d(x[ind[1]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[2]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[3]])-wp.vec3d(x[ind[0]]))
    mu = wp.float64(materials[t, 0])
    lam = wp.float64(materials[t, 1]) + mu
    shape = mat43(0.)
    for k in range(3):
        shape[0, k] = -(dm[0, k]+dm[1, k]+dm[2, k])
        for a in range(1, 4):
            shape[a, k] = dm[a-1, k]
    volume = wp.float64(1.) / (wp.float64(6.) * wp.determinant(dm))
    hessians[t] = tet_hessian_spectral(ds*dm, shape, volume, mu, lam,
                                     wp.float64(materials[t, 2]), dt)


@position_overloads("x", poses=('center',))
@wp.kernel
def normals(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
            kinds: wp.array(dtype=int), areas: wp.array(dtype=wp.float64),
            moll_eps: wp.array(dtype=wp.float64), count: wp.array(dtype=int),
            center: Any, dhat: wp.float64, dmin: wp.float64,
            stiffness: wp.float64, offset: wp.float64, hessians: wp.array(dtype=mat12)):
    _center = pose_value(center)
    i = wp.tid()
    if i < count[0]:
        ind = ids[i]
        a, b = point(x, ind[0], _center), point(x, ind[1], _center)
        c, d = point(x, ind[2], _center), point(x, ind[3], _center)
        q, gq, hq, w = squared_distance(a, b, c, d, kinds[i], False)
        h = mat12(0.)
        if wp.sqrt(q) < offset+dmin+dhat:
            e, g, h, w = contact_terms(a, b, c, d, kinds[i], dhat, dmin,
                                       stiffness, areas[i], offset, moll_eps[i], True)
            h = psd12(h)
        hessians[i] = h


@position_overloads("x", "old", poses=('center', 'old_center'))
@wp.kernel
def friction(x: wp.array(dtype=Any), old: wp.array(dtype=Any),
             ids: wp.array(dtype=wp.vec4i), count: wp.array(dtype=int),
             center: Any, old_center: Any, weights: wp.array(dtype=wp.vec4d),
             normals: wp.array(dtype=wp.vec3d), loads: wp.array(dtype=wp.float64),
             eps: wp.float64, hessians: wp.array(dtype=mat12)):
    _center = pose_value(center)
    _old_center = pose_value(old_center)
    i = wp.tid()
    if i < count[0]:
        h = mat12(0.)
        if loads[i] > wp.float64(0.):
            ind, w = ids[i], weights[i]
            slip = wp.vec3d(0.)
            for a in range(4):
                slip += w[a]*(point(x, ind[a], _center)-point(old, ind[a], _old_center))
            e, gv, hv = friction_terms(slip, normals[i], loads[i], eps)
            for a in range(4):
                for b in range(4):
                    for k in range(3):
                        for l in range(3):
                            h[3*a+k, 3*b+l] = w[a]*w[b]*hv[k, l]
        hessians[i] = h
