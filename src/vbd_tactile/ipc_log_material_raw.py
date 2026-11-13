"""Unprojected log-NH elastic tangent with the existing GN strain damping.

Kept in a separate Warp module so the optional raw path does not recompile the
large SVD/Jacobi kernels. The tensor-to-tet mapping is checked independently.
"""
from typing import Any
import warp as wp
from .ipc_precision import position_overloads
from .ipc_gpu_math import mat9, mat12
from .ipc_gpu_kernels import mat43
from .ipc_log_material import elastic_raw_hessian
wp.set_module_options({'enable_backward':False})


@wp.func
def raw_tet_tangent(f: wp.mat33d, dm: wp.mat33d, mu: wp.float64, lam: wp.float64,
                damp: wp.float64, dt: wp.float64):
    hf = elastic_raw_hessian(f,mu,lam)
    # The same Gauss-Newton strain-rate Hessian as the stable material.
    fft = f*wp.transpose(f)
    for p in range(9):
        i, k = p//3, p%3
        for q in range(9):
            a, b = q//3, q%3
            value = f[i,b]*f[a,k]
            if k == b:
                value += fft[i,a]
            hf[p,q] += wp.float64(2.)*damp/dt*value
    shape = mat43(0.)
    for k in range(3):
        shape[0,k] = -(dm[0,k]+dm[1,k]+dm[2,k])
        for a in range(1,4):
            shape[a,k] = dm[a-1,k]
    volume = wp.float64(1.)/(wp.float64(6.)*wp.determinant(dm))
    h = mat12(0.)
    for p in range(12):
        a, i = p//3, p%3
        for q in range(12):
            b, j = q//3, q%3
            value = wp.float64(0.)
            for k in range(3):
                for l in range(3):
                    value += shape[a,k]*hf[3*i+k,3*j+l]*shape[b,l]
            h[p,q] = volume*value
    return h


@position_overloads('x')
@wp.kernel
def hessian_unprojected(x:wp.array(dtype=Any),ids:wp.array(dtype=wp.vec4i),
        poses:wp.array(dtype=wp.mat33),materials:wp.array2d(dtype=float),
        dt:wp.float64,out:wp.array(dtype=mat12),mass:wp.array(dtype=float),skip_fixed:bool):
    """Raw elastic curvature plus unchanged GN strain damping, before constraints.

    This tangent can be indefinite. The ABD on-demand policy must detect a
    failed preconditioner/PCG direction and retry the existing PSD tangent.
    Energy, stress, contact curvature, mass and damping are not changed.
    """
    t=wp.tid();ind=ids[t];dm=wp.mat33d(poses[t])
    mobile=not skip_fixed
    for a in range(4):mobile=mobile or mass[ind[a]]>0.
    if not mobile:
        out[t]=mat12(0.)
        return
    ds=wp.matrix_from_cols(wp.vec3d(x[ind[1]])-wp.vec3d(x[ind[0]]),
        wp.vec3d(x[ind[2]])-wp.vec3d(x[ind[0]]),wp.vec3d(x[ind[3]])-wp.vec3d(x[ind[0]]))
    out[t]=raw_tet_tangent(ds*dm,dm,wp.float64(materials[t,0]),wp.float64(materials[t,1]),
        wp.float64(materials[t,2]),dt)
