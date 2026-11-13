"""Optional C2 volume barrier, distinct from Stable Neo-Hookean elasticity.

Energy per rest volume: scale * (lambda + mu) * b(J-Jmin, h) / h,
where h=Jactivation-Jmin. It is exactly zero above activation and diverges at
the existing feasibility floor. Forces are the exact energy gradient. The
positive generalized Gauss-Newton Hessian uses b'' grad(J) grad(J)^T; the
line search and force residual always evaluate the full nonlinear energy.
"""
from typing import Any
import warp as wp
from .ipc_precision import position_overloads
from .ipc_gpu_math import barrier, mat12, vec12

wp.set_module_options({"enable_backward": False})


@wp.func
def jacobian_gradients(f: wp.mat33d, dm: wp.mat33d):
    ft = wp.transpose(f)
    co = wp.matrix_from_cols(wp.cross(ft[1],ft[2]), wp.cross(ft[2],ft[0]), wp.cross(ft[0],ft[1]))
    g1 = co*wp.vec3d(dm[0,0],dm[0,1],dm[0,2])
    g2 = co*wp.vec3d(dm[1,0],dm[1,1],dm[1,2])
    g3 = co*wp.vec3d(dm[2,0],dm[2,1],dm[2,2])
    return -(g1+g2+g3), g1, g2, g3


@position_overloads('x')
@wp.kernel
def evaluate(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
             poses: wp.array(dtype=wp.mat33), materials: wp.array2d(dtype=float),
             floor: wp.float64, activation: wp.float64, scale: wp.float64,
             gradient: wp.array(dtype=wp.vec3d), stats: wp.array(dtype=wp.float64)):
    t = wp.tid(); ind = ids[t]; dm = wp.mat33d(poses[t])
    f = wp.matrix_from_cols(wp.vec3d(x[ind[1]])-wp.vec3d(x[ind[0]]),
        wp.vec3d(x[ind[2]])-wp.vec3d(x[ind[0]]), wp.vec3d(x[ind[3]])-wp.vec3d(x[ind[0]]))*dm
    j = wp.determinant(f)
    if not wp.isfinite(j) or j <= floor:
        wp.atomic_add(stats, 2, wp.float64(1.0))
    elif j < activation:
        h = activation-floor
        e, first, second = barrier(j-floor, h)
        volume = wp.float64(1.0)/(wp.float64(6.0)*wp.determinant(dm))
        k = scale*(wp.float64(materials[t,0])+wp.float64(materials[t,1]))*volume/h
        a,b,c,d = jacobian_gradients(f,dm)
        wp.atomic_add(stats,0,k*e)
        wp.atomic_add(gradient,ind[0],k*first*a)
        wp.atomic_add(gradient,ind[1],k*first*b)
        wp.atomic_add(gradient,ind[2],k*first*c)
        wp.atomic_add(gradient,ind[3],k*first*d)


@position_overloads('x')
@wp.kernel
def add_hessian(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
                poses: wp.array(dtype=wp.mat33), materials: wp.array2d(dtype=float),
                floor: wp.float64, activation: wp.float64, scale: wp.float64,
                hessians: wp.array(dtype=mat12)):
    t=wp.tid(); ind=ids[t]; dm=wp.mat33d(poses[t])
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
