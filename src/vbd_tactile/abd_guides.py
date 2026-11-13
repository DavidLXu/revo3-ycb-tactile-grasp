"""Exact fixed-orientation translation guides, enforced in the joint solve.

A nonzero unit axis permits only COM motion along that world-space axis.
Zero axes leave bodies free. Projecting H as P H P + (I-P) preserves SPD;
constraint forces are excluded from the free equilibrium residual, not erased
from the physical contact output. No state is teleported after solving.
"""
import warp as wp

wp.set_module_options({'enable_backward': False})


@wp.func
def projected(value: wp.vec3d, i: int, n: int, axes: wp.array(dtype=wp.vec3d)):
    result=value
    if i>=n:
        b=(i-n)//4; column=(i-n)%4; axis=axes[b]
        if wp.dot(axis,axis)>wp.float64(0.):
            result=wp.vec3d(0.)
            if column==0:result=axis*wp.dot(axis,value)
    return result


@wp.kernel
def project(x:wp.array(dtype=wp.vec3d),axes:wp.array(dtype=wp.vec3d),n:int,
            out:wp.array(dtype=wp.vec3d)):
    i=wp.tid();out[i]=projected(x[i],i,n,axes)


@wp.kernel
def finish(x:wp.array(dtype=wp.vec3d),y:wp.array(dtype=wp.vec3d),
           product:wp.array(dtype=wp.vec3d),axes:wp.array(dtype=wp.vec3d),n:int,
           alpha:wp.float64,beta:wp.float64,out:wp.array(dtype=wp.vec3d)):
    i=wp.tid()
    complement=x[i]-projected(x[i],i,n,axes)
    value=projected(product[i],i,n,axes)+complement
    out[i]=alpha*value+beta*y[i]


@wp.func
def projector(i:int,n:int,axes:wp.array(dtype=wp.vec3d)):
    p=wp.identity(n=3,dtype=wp.float64)
    if i>=n:
        b=(i-n)//4;axis=axes[b]
        if wp.dot(axis,axis)>wp.float64(0.):
            p=wp.mat33d(0.)
            if (i-n)%4==0:p=wp.outer(axis,axis)
    return p


@wp.kernel
def project_blocks(rows:wp.array(dtype=int),columns:wp.array(dtype=int),
        values:wp.array(dtype=wp.mat33d),axes:wp.array(dtype=wp.vec3d),n:int):
    k=wp.tid();i=rows[k];j=columns[k]
    if i>=n or j>=n:
        pi=projector(i,n,axes);pj=projector(j,n,axes)
        value=pi*values[k]*pj
        if i==j:value+=wp.identity(n=3,dtype=wp.float64)-pi
        values[k]=value


@wp.kernel
def reduced_inverse(diagonal:wp.array3d(dtype=wp.float64),axes:wp.array(dtype=wp.vec3d),
        inverse:wp.array3d(dtype=wp.float64),stats:wp.array(dtype=wp.float64)):
    b=wp.tid();axis=axes[b]
    if wp.dot(axis,axis)>wp.float64(0.):
        stiffness=wp.float64(0.)
        for i in range(3):
            for j in range(3):stiffness+=axis[i]*diagonal[b,i,j]*axis[j]
        if not wp.isfinite(stiffness) or stiffness<=wp.float64(0.):
            wp.atomic_add(stats,2,wp.float64(1.));stiffness=wp.float64(1.)
        for i in range(12):
            for j in range(12):
                value=wp.float64(0.)
                if i==j:value=wp.float64(1.)
                if i<3 and j<3:
                    component=axis[i]*axis[j]
                    complement=value-component
                    value=complement+component/stiffness
                inverse[b,i,j]=value


@wp.kernel
def translation_mass_blocks(mass:wp.array(dtype=wp.float64),dt:wp.float64,
        h:wp.array3d(dtype=wp.float64),diagonal:wp.array3d(dtype=wp.float64)):
    # For a fixed affine shape, only the COM mass block survives P H P.
    # Contact contributions are added to diagonal/assembly afterwards.
    b,i,j=wp.tid();value=wp.float64(0.)
    if i<3 and i==j:value=mass[b]/(dt*dt)
    h[b,i,j]=value;diagonal[b,i,j]=value
