"""Contract the log-NH raw tangent directly into parallel 3x3 node blocks.

For rest shape gradients ga, gb, va=F^-T ga, vb=F^-T gb and
q=lambda log(J)-mu, the elastic block is
V [mu (ga.gb) I + lambda va vb^T - q vb va^T].
The unchanged GN strain-damping block is added before assembly. This is the
same indefinite tangent as ipc_log_material_raw, not a new material/projection.
"""
from typing import Any
import warp as wp
from .ipc_precision import position_overloads
from .ipc_log_material import cofactor

wp.set_module_options({'enable_backward':False})


@position_overloads('x')
@wp.kernel
def hessian_blocks(x:wp.array(dtype=Any),ids:wp.array(dtype=wp.vec4i),
        poses:wp.array(dtype=wp.mat33),materials:wp.array2d(dtype=float),
        dt:wp.float64,out:wp.array3d(dtype=wp.float64),mass:wp.array(dtype=float),skip_fixed:bool):
    t,a,b=wp.tid();ind=ids[t];dm=wp.mat33d(poses[t]);h=wp.mat33d(0.)
    mobile=not skip_fixed
    for node in range(4):mobile=mobile or mass[ind[node]]>0.
    if mobile:
        ds=wp.matrix_from_cols(wp.vec3d(x[ind[1]])-wp.vec3d(x[ind[0]]),
            wp.vec3d(x[ind[2]])-wp.vec3d(x[ind[0]]),wp.vec3d(x[ind[3]])-wp.vec3d(x[ind[0]]))
        f=ds*dm;j=wp.determinant(f)
        if j>wp.float64(0.) and wp.isfinite(j):
            ga=wp.vec3d(0.);gb=wp.vec3d(0.)
            for k in range(3):
                if a==0:ga[k]=-(dm[0,k]+dm[1,k]+dm[2,k])
                else:ga[k]=dm[a-1,k]
                if b==0:gb[k]=-(dm[0,k]+dm[1,k]+dm[2,k])
                else:gb[k]=dm[b-1,k]
            invt=cofactor(f)/j;va=invt*ga;vb=invt*gb
            mu=wp.float64(materials[t,0]);lam=wp.float64(materials[t,1]);damp=wp.float64(materials[t,2])
            dot=wp.dot(ga,gb);q=lam*wp.log(j)-mu
            h=mu*dot*wp.identity(n=3,dtype=wp.float64)+lam*wp.outer(va,vb)-q*wp.outer(vb,va)
            h+=wp.float64(2.)*damp/dt*(wp.outer(f*gb,f*ga)+dot*f*wp.transpose(f))
            h/=wp.float64(6.)*wp.determinant(dm)
        else:h=wp.mat33d(wp.nan)
    for i in range(3):
        for k in range(3):out[t,3*a+i,3*b+k]=h[i,k]
