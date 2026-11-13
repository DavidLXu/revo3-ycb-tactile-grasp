"""Cancellation-resistant energy increments for the float64 sphere IPC path.

These are algebraic differences of the same inertia, elastic, damping, volume,
normal contact and frozen friction energies. They do not change the objective
or Armijo condition. A free self-contact at either endpoint or in the frozen
frame disables this comparison; the original complete objective is then used.
"""
import warp as wp
from .ipc_gpu_math import barrier
from .ipc_primitives_warp import _weights,_delta
wp.set_module_options({'enable_backward':False})


@wp.func
def log1p_stable(x: wp.float64):
    # The atanh series avoids forming 1+x when x is below machine epsilon.
    if wp.abs(x) < wp.float64(.01):
        z=x/(wp.float64(2)+x); z2=z*z
        return wp.float64(2)*z*(wp.float64(1)+z2*(wp.float64(1)/wp.float64(3)
            +z2*(wp.float64(1)/wp.float64(5)+z2*(wp.float64(1)/wp.float64(7)
            +z2/wp.float64(9)))))
    return wp.log(wp.float64(1)+x)


@wp.func
def barrier_change(d: wp.float64, dd: wp.float64, h: wp.float64):
    end=d+dd
    if d>wp.float64(0) and d<h and end>wp.float64(0) and end<h:
        t=d-h
        return -dd*(wp.float64(2)*t+dd)*wp.log(d/h)-(t+dd)*(t+dd)*log1p_stable(dd/d)
    e0,g0,h0=barrier(d,h);e1,g1,h1=barrier(end,h)
    return e1-e0


@wp.func
def smooth_friction_change(r: wp.float64, dr: wp.float64, eps: wp.float64):
    end=r+dr
    value=dr
    if r<eps and end<eps:
        value=dr*((wp.float64(2)*r+dr)/eps
            -(wp.float64(3)*r*r+wp.float64(3)*r*dr+dr*dr)/(wp.float64(3)*eps*eps))
    elif r<eps:
        d=eps-r
        value=d*((wp.float64(2)*r+d)/eps
            -(wp.float64(3)*r*r+wp.float64(3)*r*d+d*d)/(wp.float64(3)*eps*eps))+(dr-d)
    elif end<eps:
        d=end-eps
        value=(eps-r)+d*((wp.float64(2)*eps+d)/eps
            -(wp.float64(3)*eps*eps+wp.float64(3)*eps*d+d*d)/(wp.float64(3)*eps*eps))
    return value


@wp.kernel
def free_contacts(ids:wp.array(dtype=wp.vec4i),count:wp.array(dtype=int),mass:wp.array(dtype=float),flag:wp.array(dtype=int)):
    i=wp.tid()
    if i<count[0]:
        for j in range(4):
            if ids[i][j]>=0 and mass[ids[i][j]]>0.0:wp.atomic_max(flag,0,1)


@wp.kernel
def inertia_change(a:wp.array(dtype=wp.vec3d),b:wp.array(dtype=wp.vec3d),predictor:wp.array(dtype=wp.vec3d),
                   mass:wp.array(dtype=float),dt:wp.float64,out:wp.array(dtype=wp.float64)):
    i=wp.tid();dx=b[i]-a[i]
    wp.atomic_add(out,0,wp.float64(.5)*wp.float64(mass[i])/(dt*dt)*wp.dot(dx,wp.float64(2)*(a[i]-predictor[i])+dx))


@wp.func
def inner(a:wp.mat33d,b:wp.mat33d):
    total=wp.float64(0)
    for i in range(3):
        for j in range(3):total+=a[i,j]*b[i,j]
    return total


@wp.func
def cofactor(a:wp.mat33d):
    t=wp.transpose(a)
    return wp.matrix_from_cols(wp.cross(t[1],t[2]),wp.cross(t[2],t[0]),wp.cross(t[0],t[1]))


@wp.kernel
def tet_change(a:wp.array(dtype=wp.vec3d),b:wp.array(dtype=wp.vec3d),old:wp.array(dtype=wp.vec3d),
               ids:wp.array2d(dtype=int),poses:wp.array(dtype=wp.mat33),material:wp.array2d(dtype=float),
               dt:wp.float64,floor:wp.float64,activation:wp.float64,scale:wp.float64,out:wp.array(dtype=wp.float64)):
    i=wp.tid();j,k,l,m=ids[i,0],ids[i,1],ids[i,2],ids[i,3];dm=wp.mat33d(poses[i])
    f=wp.matrix_from_cols(a[k]-a[j],a[l]-a[j],a[m]-a[j])*dm
    delta=wp.matrix_from_cols((b[k]-a[k])-(b[j]-a[j]),(b[l]-a[l])-(b[j]-a[j]),(b[m]-a[m])-(b[j]-a[j]))*dm
    f0=wp.matrix_from_cols(old[k]-old[j],old[l]-old[j],old[m]-old[j])*dm
    jac=wp.determinant(f);dj=inner(cofactor(f),delta)+inner(cofactor(delta),f)+wp.determinant(delta)
    mu=wp.float64(material[i,0]);lam=wp.float64(material[i,1])+mu;alpha=wp.float64(1)+mu/lam
    strain=wp.transpose(f)*f-wp.transpose(f0)*f0
    dc=wp.transpose(f)*delta+wp.transpose(delta)*f+wp.transpose(delta)*delta
    volume=wp.float64(1)/(wp.float64(6)*wp.determinant(dm))
    change=volume*(wp.float64(.5)*mu*inner(delta,wp.float64(2)*f+delta)
        +wp.float64(.5)*lam*dj*(wp.float64(2)*(jac-alpha)+dj)
        +wp.float64(material[i,2])/(wp.float64(2)*dt)*inner(dc,wp.float64(2)*strain+dc))
    if activation>floor:
        change+=volume*scale*lam/(activation-floor)*barrier_change(jac-floor,dj,activation-floor)
    wp.atomic_add(out,0,change)


@wp.kernel
def needed(stats: wp.array(dtype=wp.float64), values: wp.array(dtype=wp.float64),
           enabled: wp.array(dtype=int)):
    # Only spend the extra kernels when global subtraction is near its noise
    # floor. This gate selects arithmetic; it never accepts a Newton trial.
    error = wp.float64(64)*wp.float64(2.220446049250313e-16)*wp.max(wp.abs(stats[0]), wp.abs(values[0]))
    enabled[0] = int(wp.isfinite(stats[0]) and wp.isfinite(values[0])
                     and wp.abs(stats[0]-values[0]) <= error)


@wp.kernel
def sphere_change(a:wp.array(dtype=wp.vec3d),b:wp.array(dtype=wp.vec3d),old:wp.array(dtype=wp.vec3d),
                  ids:wp.array(dtype=wp.vec4i),areas:wp.array(dtype=wp.float64),center:wp.array(dtype=wp.vec3d),
                  old_center:wp.array(dtype=wp.vec3d),offset:wp.float64,dhat:wp.float64,stiff:wp.float64,
                  weights:wp.array(dtype=wp.vec4d),normals:wp.array(dtype=wp.vec3d),loads:wp.array(dtype=wp.float64),
                  eps:wp.float64,out:wp.array(dtype=wp.float64)):
    i=wp.tid();ii=ids[i];p=center[0]
    w0=_weights(p,a[ii[1]],a[ii[2]],a[ii[3]],0);w1=_weights(p,b[ii[1]],b[ii[2]],b[ii[3]],0)
    v0=_delta(w0,p,a[ii[1]],a[ii[2]],a[ii[3]])
    displacement=wp.vec3d(0)
    same=True
    for j in range(1,4):
        displacement+=w1[j]*(b[ii[j]]-a[ii[j]])
        same=same and ((w1[j]==wp.float64(0))==(w0[j]==wp.float64(0)))
    anchor=int(1)
    for j in range(2,4):
        if w0[j]!=wp.float64(0) or w1[j]!=wp.float64(0):anchor=j
    tangent=wp.vec3d(0)
    for j in range(1,4):
        tangent+=(w1[j]-w0[j])*(a[ii[j]]-a[ii[anchor]])
    dv=displacement+tangent
    r0=wp.length(v0);r1=wp.length(v0+dv)
    dq=wp.dot(dv,wp.float64(2)*v0+dv)
    if same:
        # Closest-point stationarity makes the old residual orthogonal to
        # changes of coordinates inside the same closest feature.
        dq=wp.dot(displacement,wp.float64(2)*(v0+tangent)+displacement)+wp.dot(tangent,tangent)
    dr=wp.float64(0)
    if r0+r1>wp.float64(0):dr=dq/(r0+r1)
    change=stiff*areas[i]/(dhat*dhat)*barrier_change(r0-offset,dr,dhat)
    if loads[i]>wp.float64(0):
        w=weights[i];s0=w[0]*(p-old_center[0]);ds=wp.vec3d(0)
        for j in range(1,4):
            s0+=w[j]*(a[ii[j]]-old[ii[j]]);ds+=w[j]*(b[ii[j]]-a[ii[j]])
        t=s0-normals[i]*wp.dot(normals[i],s0)
        dt=ds-normals[i]*wp.dot(normals[i],ds)
        r=wp.length(t);end=wp.length(t+dt);delta=wp.float64(0)
        if r+end>wp.float64(0):delta=wp.dot(dt,wp.float64(2)*t+dt)/(r+end)
        change+=loads[i]*smooth_friction_change(r,delta,eps)
    wp.atomic_add(out,0,change)
