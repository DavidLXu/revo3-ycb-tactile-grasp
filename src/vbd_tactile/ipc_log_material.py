"""Optional log-J Neo-Hookean elasticity; unchanged strain damping and PSD policy.

psi = mu/2 (F:F-3) - mu log J + lambda/2 (log J)^2, J>0.
Lambda here is the physical Lame coefficient, NOT the SNH lambda+mu shift.
This matches the published Taccel elastic energy, not its complete solver.
"""
from typing import Any
import warp as wp
from .ipc_precision import position_overloads
from .ipc_gpu_math import mat9, mat12, vec12, psd9
from .ipc_gpu_kernels import mat43, signed_eps, scatter
from .ipc_gpu_material import eigen3, add_mode

wp.set_module_options({'enable_backward': False})


@wp.func
def cofactor(f: wp.mat33d):
    ft = wp.transpose(f)
    return wp.matrix_from_cols(wp.cross(ft[1], ft[2]), wp.cross(ft[2], ft[0]), wp.cross(ft[0], ft[1]))


@wp.func
def elastic(f: wp.mat33d, mu: wp.float64, lam: wp.float64):
    j = wp.determinant(f)
    if not wp.isfinite(j) or j <= wp.float64(0.):
        return wp.float64(wp.inf), wp.mat33d(wp.nan)
    logj = wp.log(j)
    norm = wp.float64(0.)
    for i in range(3):
        for k in range(3):
            norm += f[i, k]*f[i, k]
    energy = mu*wp.float64(.5)*(norm-wp.float64(3.)) - mu*logj + lam*wp.float64(.5)*logj*logj
    stress = mu*f + ((lam*logj-mu)/j)*cofactor(f)
    return energy, stress


@wp.func
def elastic_raw_hessian(f: wp.mat33d, mu: wp.float64, lam: wp.float64):
    j = wp.determinant(f)
    if not wp.isfinite(j) or j <= wp.float64(0.):
        return mat9(wp.nan)
    co = cofactor(f)
    first = (lam*wp.log(j)-mu)/j
    second = (lam+mu-lam*wp.log(j))/(j*j)
    h = mat9(0.)
    for p in range(9):
        i, k = p//3, p%3
        for q in range(9):
            a, b = q//3, q%3
            hj = wp.float64(0.)
            for u in range(3):
                for v in range(3):
                    hj += signed_eps(i,a,u)*signed_eps(k,b,v)*f[u,v]
            h[p,q] = second*co[i,k]*co[a,b] + first*hj
            if p == q:
                h[p,q] += mu
    return h


@wp.func
def elastic_psd(f: wp.mat33d, mu: wp.float64, lam: wp.float64):
    j = wp.determinant(f)
    if not wp.isfinite(j) or j <= wp.float64(0.):
        return mat9(wp.nan)
    first = (lam*wp.log(j)-mu)/j
    second = (lam+mu-lam*wp.log(j))/(j*j)
    u, sigma, v = wp.svd3(f)
    h = mat9(0.)
    invsqrt2 = wp.float64(0.7071067811865475244)
    for k in range(3):
        a, b = (k+1)%3, (k+2)%3
        ua, ub = wp.vec3d(u[0,a],u[1,a],u[2,a]), wp.vec3d(u[0,b],u[1,b],u[2,b])
        va, vb = wp.vec3d(v[0,a],v[1,a],v[2,a]), wp.vec3d(v[0,b],v[1,b],v[2,b])
        ab, ba = wp.outer(ua,vb), wp.outer(ub,va)
        h = add_mode(h,(ab-ba)*invsqrt2,mu+first*sigma[k])
        h = add_mode(h,(ab+ba)*invsqrt2,mu-first*sigma[k])
    stretch = wp.mat33d(0.)
    for k in range(3):
        a, b = (k+1)%3, (k+2)%3
        product = sigma[a]*sigma[b]
        stretch[k,k] = mu+second*product*product
        stretch[a,b] = (second*j+first)*sigma[k]
        stretch[b,a] = stretch[a,b]
    values, vectors = eigen3(stretch)
    for k in range(3):
        mode = wp.mat33d(0.)
        for a in range(3):
            for b in range(3):
                for r in range(3):
                    mode[a,b] += u[a,r]*vectors[r,k]*v[b,r]
        h = add_mode(h,mode,values[k])
    return h


@wp.func
def tet_hessian(f: wp.mat33d, dm: wp.mat33d, mu: wp.float64, lam: wp.float64,
                damp: wp.float64, dt: wp.float64, spectral: bool):
    hf = mat9(0.)
    if spectral:
        hf = elastic_psd(f,mu,lam)
    else:
        hf = psd9(elastic_raw_hessian(f,mu,lam))
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


@position_overloads('x','old')
@wp.kernel
def tetrahedra(x: wp.array(dtype=Any), old: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
               poses: wp.array(dtype=wp.mat33), materials: wp.array2d(dtype=float),
               dt: wp.float64, need_hessian: bool, gradient: wp.array(dtype=wp.vec3d),
               stats: wp.array(dtype=wp.float64), hessians: wp.array(dtype=mat12)):
    t = wp.tid()
    ind, dm = ids[t], wp.mat33d(poses[t])
    ds = wp.matrix_from_cols(wp.vec3d(x[ind[1]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[2]])-wp.vec3d(x[ind[0]]),wp.vec3d(x[ind[3]])-wp.vec3d(x[ind[0]]))
    ds0 = wp.matrix_from_cols(wp.vec3d(old[ind[1]])-wp.vec3d(old[ind[0]]),
                             wp.vec3d(old[ind[2]])-wp.vec3d(old[ind[0]]),wp.vec3d(old[ind[3]])-wp.vec3d(old[ind[0]]))
    f, f0 = ds*dm, ds0*dm
    mu, lam, damp = wp.float64(materials[t,0]), wp.float64(materials[t,1]), wp.float64(materials[t,2])
    energy, stress = elastic(f,mu,lam)
    volume = wp.float64(1.)/(wp.float64(6.)*wp.determinant(dm))
    strain = wp.transpose(f)*f-wp.transpose(f0)*f0
    cnorm = wp.float64(0.)
    for i in range(3):
        for k in range(3):
            cnorm += strain[i,k]*strain[i,k]
    energy = volume*(energy+damp/(wp.float64(2.)*dt)*cnorm)
    stress += wp.float64(2.)*damp/dt*(f*strain)
    g = vec12(0.)
    for a in range(4):
        shape = wp.vec3d(0.)
        for k in range(3):
            if a == 0:
                shape[k] = -(dm[0,k]+dm[1,k]+dm[2,k])
            else:
                shape[k] = dm[a-1,k]
        grad = volume*(stress*shape)
        for k in range(3):
            g[3*a+k] = grad[k]
    scatter(ind,g,energy,gradient,gradient,stats,False)
    if need_hessian:
        hessians[t] = tet_hessian(f,dm,mu,lam,damp,dt,False)


@position_overloads("x")
@wp.kernel
def hessian(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
    poses: wp.array(dtype=wp.mat33), materials: wp.array2d(dtype=float),
    dt: wp.float64, out: wp.array(dtype=mat12)):
    t = wp.tid()
    ind, dm = ids[t], wp.mat33d(poses[t])
    ds = wp.matrix_from_cols(wp.vec3d(x[ind[1]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[2]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[3]])-wp.vec3d(x[ind[0]]))
    out[t] = tet_hessian(ds*dm,dm,wp.float64(materials[t,0]),wp.float64(materials[t,1]),
                         wp.float64(materials[t,2]),dt,False)


@position_overloads("x")
@wp.kernel
def hessian_free(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
    poses: wp.array(dtype=wp.mat33), materials: wp.array2d(dtype=float),
    dt: wp.float64, out: wp.array(dtype=mat12), mass: wp.array(dtype=float)):
    t = wp.tid()
    ind, dm = ids[t], wp.mat33d(poses[t])
    mobile = bool(False)
    for a in range(4):
        mobile = mobile or mass[ind[a]] > 0.
    if not mobile:
        out[t] = mat12(0.)
        return
    ds = wp.matrix_from_cols(wp.vec3d(x[ind[1]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[2]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[3]])-wp.vec3d(x[ind[0]]))
    out[t] = tet_hessian(ds*dm,dm,wp.float64(materials[t,0]),wp.float64(materials[t,1]),
                         wp.float64(materials[t,2]),dt,False)


@position_overloads("x")
@wp.kernel
def hessian_spectral(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
    poses: wp.array(dtype=wp.mat33), materials: wp.array2d(dtype=float),
    dt: wp.float64, out: wp.array(dtype=mat12)):
    t = wp.tid()
    ind, dm = ids[t], wp.mat33d(poses[t])
    ds = wp.matrix_from_cols(wp.vec3d(x[ind[1]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[2]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[3]])-wp.vec3d(x[ind[0]]))
    out[t] = tet_hessian(ds*dm,dm,wp.float64(materials[t,0]),wp.float64(materials[t,1]),
                         wp.float64(materials[t,2]),dt,True)


@position_overloads("x")
@wp.kernel
def hessian_spectral_free(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
    poses: wp.array(dtype=wp.mat33), materials: wp.array2d(dtype=float),
    dt: wp.float64, out: wp.array(dtype=mat12), mass: wp.array(dtype=float)):
    t = wp.tid()
    ind, dm = ids[t], wp.mat33d(poses[t])
    mobile = bool(False)
    for a in range(4):
        mobile = mobile or mass[ind[a]] > 0.
    if not mobile:
        out[t] = mat12(0.)
        return
    ds = wp.matrix_from_cols(wp.vec3d(x[ind[1]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[2]])-wp.vec3d(x[ind[0]]),
                            wp.vec3d(x[ind[3]])-wp.vec3d(x[ind[0]]))
    out[t] = tet_hessian(ds*dm,dm,wp.float64(materials[t,0]),wp.float64(materials[t,1]),
                         wp.float64(materials[t,2]),dt,True)
