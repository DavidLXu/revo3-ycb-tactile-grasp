"""Algebraic energy differences for the complete joint FEM/mesh/ABD objective.

No force tolerance or Armijo slack is introduced. Current active pairs supply
their change; base pairs which became inactive supply their old energy. This
partitions the union without matching or truncating changing contact lists.
"""
import warp as wp
from .ipc_primitives_warp import _weights, _delta
from .ipc_gpu_math import barrier
from .ipc_energy_increment import inner, cofactor, log1p_stable, barrier_change, smooth_friction_change

wp.set_module_options({'enable_backward': False})


@wp.func
def log1pmx(x: wp.float64):
    # log(1+x)-x, without subtracting its first-order part near zero.
    if wp.abs(x) < wp.float64(.01):
        z=x/(wp.float64(2)+x); z2=z*z
        return -x*z+wp.float64(2)*z*z2*(wp.float64(1)/wp.float64(3)
            +z2*(wp.float64(1)/wp.float64(5)+z2*(wp.float64(1)/wp.float64(7)
            +z2/wp.float64(9))))
    return wp.log(wp.float64(1)+x)-x


@wp.kernel
def log_tet_change(a:wp.array(dtype=wp.vec3d), b:wp.array(dtype=wp.vec3d), old:wp.array(dtype=wp.vec3d),
                   ids:wp.array2d(dtype=int), poses:wp.array(dtype=wp.mat33), material:wp.array2d(dtype=float),
                   dt:wp.float64, floor:wp.float64, activation:wp.float64, scale:wp.float64,
                   out:wp.array(dtype=wp.float64)):
    i=wp.tid(); j,k,l,m=ids[i,0],ids[i,1],ids[i,2],ids[i,3]; dm=wp.mat33d(poses[i])
    f=wp.matrix_from_cols(a[k]-a[j],a[l]-a[j],a[m]-a[j])*dm
    delta=wp.matrix_from_cols((b[k]-a[k])-(b[j]-a[j]),(b[l]-a[l])-(b[j]-a[j]),
        (b[m]-a[m])-(b[j]-a[j]))*dm
    f0=wp.matrix_from_cols(old[k]-old[j],old[l]-old[j],old[m]-old[j])*dm
    jac=wp.determinant(f); co=cofactor(f)
    nonlinear_dj=inner(cofactor(delta),f)+wp.determinant(delta)
    dj=inner(co,delta)+nonlinear_dj
    mu=wp.float64(material[i,0]); lam=wp.float64(material[i,1])
    change=wp.float64(wp.inf)
    if jac>wp.float64(0) and jac+dj>wp.float64(0):
        t=dj/jac; dl=log1p_stable(t); coefficient=lam*wp.log(jac)-mu
        stress=mu*f+(coefficient/jac)*co
        # Exact algebraic delta psi. Separate stress work from the quadratic
        # remainder so the two O(mu*delta) terms do not cancel near rest.
        change=inner(stress,delta)+wp.float64(.5)*mu*inner(delta,delta)
        change+=coefficient*(log1pmx(t)+nonlinear_dj/jac)+wp.float64(.5)*lam*dl*dl
        strain=wp.transpose(f)*f-wp.transpose(f0)*f0
        dc=wp.transpose(f)*delta+wp.transpose(delta)*f+wp.transpose(delta)*delta
        change+=wp.float64(material[i,2])/(wp.float64(2)*dt)*inner(dc,wp.float64(2)*strain+dc)
        if activation>floor:
            change+=scale*(lam+mu)/(activation-floor)*barrier_change(jac-floor,dj,activation-floor)
    volume=wp.float64(1)/(wp.float64(6)*wp.determinant(dm))
    wp.atomic_add(out,0,volume*change)


@wp.kernel
def increment_needed(stats:wp.array(dtype=wp.float64), values:wp.array(dtype=wp.float64),
                     line:wp.array(dtype=wp.float64), enabled:wp.array(dtype=int)):
    # World-to-affine mapping and local constitutive evaluation can introduce
    # more roundoff than a bound based only on the final summed energy. In
    # particular, two almost cancelling objectives need not have close stored
    # values. Verify every otherwise rejected finite trial with the algebraic
    # increment; this selects arithmetic, never relaxes the Armijo inequality.
    error=wp.float64(64)*wp.float64(2.220446049250313e-16)*wp.max(wp.abs(stats[0]),wp.abs(values[0]))
    rejected=stats[0]>values[0]+wp.float64(1e-4)*line[0]
    enabled[0]=int(wp.isfinite(stats[0]) and wp.isfinite(values[0])
                   and (rejected or wp.abs(stats[0]-values[0])<=error))


@wp.kernel
def body_change(a:wp.array(dtype=wp.vec3d), b:wp.array(dtype=wp.vec3d),
                predictor:wp.array(dtype=wp.vec3d), radius:wp.array(dtype=wp.float64),
                mass:wp.array(dtype=wp.mat44d), kv:wp.array(dtype=wp.float64),
                dt:wp.float64, n:int, out:wp.array(dtype=wp.float64)):
    body=wp.tid(); base=n+4*body; r=radius[body]
    f=wp.matrix_from_cols(a[base+1],a[base+2],a[base+3])/r
    df=wp.matrix_from_cols(b[base+1]-a[base+1],b[base+2]-a[base+2],b[base+3]-a[base+3])/r
    c=wp.transpose(f)*f-wp.identity(n=3,dtype=wp.float64)
    dc=wp.transpose(f)*df+wp.transpose(df)*f+wp.transpose(df)*df
    value=kv[body]*inner(dc,wp.float64(2)*c+dc)
    for i in range(4):
        du=b[base+i]-a[base+i]
        for j in range(4):
            value+=mass[body][i,j]/(dt*dt)*wp.dot(du,
                a[base+j]-predictor[4*body+j]+wp.float64(.5)*(b[base+j]-a[base+j]))
    wp.atomic_add(out,0,value)


@wp.func
def distance_change(a:wp.array(dtype=wp.vec3d),b:wp.array(dtype=wp.vec3d),ids:wp.vec4i,kind:int):
    w0=_weights(a[ids[0]],a[ids[1]],a[ids[2]],a[ids[3]],kind)
    w1=_weights(b[ids[0]],b[ids[1]],b[ids[2]],b[ids[3]],kind)
    v0=_delta(w0,a[ids[0]],a[ids[1]],a[ids[2]],a[ids[3]])
    v1=_delta(w1,b[ids[0]],b[ids[1]],b[ids[2]],b[ids[3]])
    displacement=wp.vec3d(0); tangent=wp.vec3d(0); same=True
    for j in range(4):
        displacement+=w1[j]*(b[ids[j]]-a[ids[j]])
        same=same and ((w1[j]==wp.float64(0))==(w0[j]==wp.float64(0)))
    # Anchor each primitive separately: the weight changes sum to zero on
    # each primitive. Never involve a distant inactive vertex in roundoff.
    first=int(1)
    if kind!=0:
        first=2; anchor=int(0)
        if w0[1]!=wp.float64(0) or w1[1]!=wp.float64(0):anchor=1
        for j in range(2):tangent+=(w1[j]-w0[j])*(a[ids[j]]-a[ids[anchor]])
    anchor=first
    for j in range(first+1,4):
        if w0[j]!=wp.float64(0) or w1[j]!=wp.float64(0):anchor=j
    for j in range(first,4):tangent+=(w1[j]-w0[j])*(a[ids[j]]-a[ids[anchor]])
    dv=displacement+tangent
    dq=wp.dot(dv,wp.float64(2)*v0+dv)
    if same:
        dq=wp.dot(displacement,wp.float64(2)*(v0+tangent)+displacement)+wp.dot(tangent,tangent)
    return wp.dot(v0,v0),wp.dot(v1,v1),dq


@wp.func
def mollifier_change(a:wp.array(dtype=wp.vec3d),b:wp.array(dtype=wp.vec3d),ids:wp.vec4i,eps:wp.float64):
    m=wp.float64(1); change=wp.float64(0)
    if eps>wp.float64(0):
        e=a[ids[1]]-a[ids[0]]; f=a[ids[3]]-a[ids[2]]
        de=(b[ids[1]]-a[ids[1]])-(b[ids[0]]-a[ids[0]])
        df=(b[ids[3]]-a[ids[3]])-(b[ids[2]]-a[ids[2]])
        normal=wp.cross(e,f); dn=wp.cross(e,df)+wp.cross(de,f)+wp.cross(de,df)
        t=wp.dot(normal,normal)/eps; delta=wp.dot(dn,wp.float64(2)*normal+dn)/eps
        end=t+delta
        if t<wp.float64(1):
            m=t*(wp.float64(2)-t)
            if end<wp.float64(1):change=delta*(wp.float64(2)-wp.float64(2)*t-delta)
            else:change=(wp.float64(1)-t)*(wp.float64(1)-t)
        elif end<wp.float64(1):change=-(wp.float64(1)-end)*(wp.float64(1)-end)
    return m,change


@wp.kernel
def normal_change(a:wp.array(dtype=wp.vec3d),b:wp.array(dtype=wp.vec3d),
                  ids:wp.array(dtype=wp.vec4i),kinds:wp.array(dtype=int),
                  areas:wp.array(dtype=wp.float64),eps:wp.array(dtype=wp.float64),
                  count:wp.array(dtype=int),dmin:wp.float64,dhat:wp.float64,stiff:wp.float64,
                  departed:bool,out:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if i<count[0]:
        q0,q1,dq=distance_change(a,b,ids[i],kinds[i])
        active=wp.sqrt(q1)<dmin+dhat
        if active!=departed:
            threshold=(wp.float64(2)*dmin+dhat)*dhat
            old,g,h=barrier(q0-dmin*dmin,threshold)
            m=wp.float64(1); dm=wp.float64(0)
            if kinds[i]!=0:m,dm=mollifier_change(a,b,ids[i],eps[i])
            value=-m*old
            if not departed:
                delta=barrier_change(q0-dmin*dmin,dq,threshold)
                value=m*delta+(old+delta)*dm
            wp.atomic_add(out,0,stiff*areas[i]*dhat/(threshold*threshold)*value)


@wp.kernel
def friction_change(a:wp.array(dtype=wp.vec3d),b:wp.array(dtype=wp.vec3d),old:wp.array(dtype=wp.vec3d),
                    ids:wp.array(dtype=wp.vec4i),weights:wp.array(dtype=wp.vec4d),
                    normals:wp.array(dtype=wp.vec3d),loads:wp.array(dtype=wp.float64),
                    count:wp.array(dtype=int),eps:wp.float64,out:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if i<count[0]:
        if loads[i]!=wp.float64(0):
            s=wp.vec3d(0); ds=wp.vec3d(0); w=weights[i]; n=normals[i]
            for j in range(4):
                s+=w[j]*(a[ids[i][j]]-old[ids[i][j]])
                ds+=w[j]*(b[ids[i][j]]-a[ids[i][j]])
            t=s-n*wp.dot(n,s); delta=ds-n*wp.dot(n,ds)
            r=wp.length(t); end=wp.length(t+delta); dr=wp.float64(0)
            if r+end>wp.float64(0):dr=wp.dot(delta,wp.float64(2)*t+delta)/(r+end)
            wp.atomic_add(out,0,loads[i]*smooth_friction_change(r,dr,eps))


class BaseContacts:
    """Only immutable pair metadata is needed; no gradient/Hessian duplicate."""
    def __init__(self,batch,device):
        self.capacity=batch.capacity
        for name in ('ids','kinds','areas','eps','count'):
            setattr(self,name,wp.empty_like(getattr(batch,name),device=device))

    def copy(self,batch):
        for name in ('ids','kinds','areas','eps','count'):wp.copy(getattr(self,name),getattr(batch,name))
