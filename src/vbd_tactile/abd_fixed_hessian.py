"""ABD-only Hessians: omit blocks eliminated by fixed-coordinate constraints.

Constitutive/contact derivative functions are shared with the original kernels.
Energy, gradient, forces and all feasibility/path checks are unchanged.
"""
from typing import Any
import warp as wp
from .ipc_precision import position_overloads, pose_value
from .ipc_gpu_kernels import mat43, tet_hessian, point
from .ipc_gpu_material import tet_hessian_spectral
from .ipc_gpu_math import mat12, vec12, friction_terms, barrier
from .ipc_volume_barrier import jacobian_gradients
wp.set_module_options({'enable_backward':False})

@wp.func
def mobile(ids:wp.vec4i,mass:wp.array(dtype=float)):
    result=bool(False)
    for i in range(4):
        if ids[i]>=0:
            result=result or mass[ids[i]]>0.
    return result

@position_overloads("x")
@wp.kernel
def tetrahedra(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
               poses: wp.array(dtype=wp.mat33), materials: wp.array2d(dtype=float),
               dt: wp.float64, hessians: wp.array(dtype=mat12), mass: wp.array(dtype=float)):
    t = wp.tid()
    ind = ids[t]
    if not mobile(ind,mass):
        hessians[t]=mat12(0.)
        return
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
                       dt: wp.float64, hessians: wp.array(dtype=mat12), mass: wp.array(dtype=float)):
    t = wp.tid()
    ind = ids[t]
    if not mobile(ind,mass):
        hessians[t]=mat12(0.)
        return
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


@position_overloads("x", "old", poses=('center', 'old_center'))
@wp.kernel
def friction(x: wp.array(dtype=Any), old: wp.array(dtype=Any),
             ids: wp.array(dtype=wp.vec4i), count: wp.array(dtype=int),
             center: Any, old_center: Any, weights: wp.array(dtype=wp.vec4d),
             normals: wp.array(dtype=wp.vec3d), loads: wp.array(dtype=wp.float64),
             eps: wp.float64, hessians: wp.array(dtype=mat12), mass: wp.array(dtype=float)):
    _center = pose_value(center)
    _old_center = pose_value(old_center)
    i = wp.tid()
    if i < count[0]:
        h = mat12(0.)
        if not mobile(ids[i],mass):
            hessians[i]=h
            return
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
@position_overloads('x')
@wp.kernel
def add_hessian(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
                poses: wp.array(dtype=wp.mat33), materials: wp.array2d(dtype=float),
                floor: wp.float64, activation: wp.float64, scale: wp.float64,
                hessians: wp.array(dtype=mat12), mass: wp.array(dtype=float)):
    t=wp.tid(); ind=ids[t]
    if not mobile(ind,mass):return
    dm=wp.mat33d(poses[t])
    f=wp.matrix_from_cols(wp.vec3d(x[ind[1]])-wp.vec3d(x[ind[0]]),
        wp.vec3d(x[ind[2]])-wp.vec3d(x[ind[0]]), wp.vec3d(x[ind[3]])-wp.vec3d(x[ind[0]]))*dm
    j=wp.determinant(f)
    if j > floor and j < activation:
        width=activation-floor
        e,first,second=barrier(j-floor,width)
        volume=wp.float64(1.0)/(wp.float64(6.0)*wp.determinant(dm))
        k=scale*(wp.float64(materials[t,0])+wp.float64(materials[t,1]))*volume/width
        a,b,c,d=jacobian_gradients(f,dm)
        g=vec12(a[0],a[1],a[2],b[0],b[1],b[2],c[0],c[1],c[2],d[0],d[1],d[2])
        h=hessians[t]
        for p in range(12):
            for q in range(12):
                h[p,q]+=k*second*g[p]*g[q]
        hessians[t]=h
