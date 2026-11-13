"""GPU affine-body coordinates, derivatives and joint FEM/ABD linear algebra.

Each body owns four vec3 coordinates: COM and the columns of radius*A.
Scaling affine coordinates into metres keeps generalized residuals in newtons.
The shape potential is kappa*volume*||A.T*A-I||_F^2 (ABD, Lan et al. 2022).
"""
import warp as wp
from .ipc_gpu_math import mat9, mat12, psd9
from .ipc_vbd_kernels import determinant_path_min

wp.set_module_options({'enable_backward': False})


@wp.func
def affine(u: wp.array(dtype=wp.vec3d), offset: int, radius: wp.float64):
    return wp.matrix_from_cols(u[offset+1], u[offset+2], u[offset+3]) / radius


@wp.kernel
def seed(x: wp.array(dtype=wp.vec3d), q: wp.array(dtype=wp.vec3d),
         qd: wp.array(dtype=wp.vec3d), force: wp.array(dtype=wp.vec3d),
         mass: wp.array(dtype=wp.float64), gravity: wp.vec3d, dt: wp.float64,
         n: int, u: wp.array(dtype=wp.vec3d), predictor: wp.array(dtype=wp.vec3d)):
    i=wp.tid()
    if i<n:
        u[i]=x[i]
    else:
        j=i-n
        u[i]=q[j]
        value=q[j]+dt*qd[j]
        if j%4==0:
            value+=dt*dt*(gravity+force[j//4]/mass[j//4])
        predictor[j]=value


@wp.kernel
def map_positions(u: wp.array(dtype=wp.vec3d), owner: wp.array(dtype=int),
                  weights: wp.array(dtype=wp.vec4d), n: int,
                  x: wp.array(dtype=wp.vec3d)):
    i=wp.tid(); b=owner[i]
    value=u[i]
    if b>=0:
        value=wp.vec3d(0.)
        for j in range(4):value+=weights[i][j]*u[n+4*b+j]
    x[i]=value


@wp.kernel
def check_bindings(x: wp.array(dtype=wp.vec3d), q: wp.array(dtype=wp.vec3d),
                   owner: wp.array(dtype=int), weights: wp.array(dtype=wp.vec4d),
                   status: wp.array(dtype=int)):
    i=wp.tid(); b=owner[i]
    if b>=0:
        value=wp.vec3d(0.)
        for j in range(4):value+=weights[i][j]*q[4*b+j]
        if not wp.isfinite(wp.length(value)) or wp.length(value-x[i])>wp.float64(1.e-12):
            wp.atomic_max(status,0,32)


@wp.kernel
def restrict_vector(full: wp.array(dtype=wp.vec3d), owner: wp.array(dtype=int),
                    weights: wp.array(dtype=wp.vec4d), mass: wp.array(dtype=float),
                    n: int, out: wp.array(dtype=wp.vec3d)):
    i=wp.tid(); b=owner[i]
    if mass[i]>0.:out[i]=full[i]
    if b>=0:
        for j in range(4):wp.atomic_add(out,n+4*b+j,weights[i][j]*full[i])


@wp.kernel
def clear_fixed(out: wp.array(dtype=wp.vec3d), mass: wp.array(dtype=float)):
    i=wp.tid()
    if mass[i]<=0.:out[i]=wp.vec3d(0.)


@wp.kernel
def body_energy(u: wp.array(dtype=wp.vec3d), predictor: wp.array(dtype=wp.vec3d),
                radius: wp.array(dtype=wp.float64), mass_matrix: wp.array(dtype=wp.mat44d),
                kv: wp.array(dtype=wp.float64), dt: wp.float64, n: int,
                gradient: wp.array(dtype=wp.vec3d), stats: wp.array(dtype=wp.float64)):
    b=wp.tid(); base=n+4*b; r=radius[b]
    a=affine(u,base,r); c=wp.transpose(a)*a-wp.identity(n=3,dtype=wp.float64)
    g=(wp.float64(4.)*kv[b]/r)*a*c
    energy=wp.float64(0.)
    for i in range(3):
        for j in range(3):energy+=kv[b]*c[i,j]*c[i,j]
    for i in range(4):
        inertial=wp.vec3d(0.)
        for j in range(4):inertial+=mass_matrix[b][i,j]*(u[base+j]-predictor[4*b+j])/(dt*dt)
        energy+=wp.float64(.5)*wp.dot(u[base+i]-predictor[4*b+i],inertial)
        value=inertial
        if i>0:value+=wp.vec3d(g[0,i-1],g[1,i-1],g[2,i-1])
        wp.atomic_add(gradient,base+i,value)
        if not wp.isfinite(wp.length(value)):wp.atomic_add(stats,2,wp.float64(1.))
    wp.atomic_add(stats,0,energy)
    if not wp.isfinite(energy):wp.atomic_add(stats,2,wp.float64(1.))


@wp.func
def shape_mode(h: mat9, mode: wp.mat33d, eigenvalue: wp.float64):
    result=h; value=wp.max(eigenvalue,wp.float64(0.))
    for p in range(9):
        for q in range(9):result[p,q]+=value*mode[p%3,p//3]*mode[q%3,q//3]
    return result


@wp.func
def shape_hessian_psd(a: wp.mat33d):
    # For E=||A^T A-I||^2, writing dA=U B V^T diagonalizes each
    # diagonal mode and each symmetric/skew pair of B. This is the exact
    # positive spectral part, without a general 9x9 Jacobi eigensolve.
    u,sigma,v=wp.svd3(a); h=mat9(0.)
    for i in range(3):
        ui=wp.vec3d(u[0,i],u[1,i],u[2,i]); vi=wp.vec3d(v[0,i],v[1,i],v[2,i])
        h=shape_mode(h,wp.outer(ui,vi),wp.float64(4.)*(wp.float64(3.)*sigma[i]*sigma[i]-wp.float64(1.)))
        for j in range(i+1,3):
            uj=wp.vec3d(u[0,j],u[1,j],u[2,j]); vj=wp.vec3d(v[0,j],v[1,j],v[2,j])
            ab=wp.outer(ui,vj); ba=wp.outer(uj,vi)
            diagonal=sigma[i]*sigma[i]+sigma[j]*sigma[j]-wp.float64(1.)
            cross=sigma[i]*sigma[j]
            h=shape_mode(h,(ab+ba)*wp.float64(0.7071067811865475244),wp.float64(4.)*(diagonal+cross))
            h=shape_mode(h,(ab-ba)*wp.float64(0.7071067811865475244),wp.float64(4.)*(diagonal-cross))
    return h


@wp.kernel
def body_hessian_spectral(u: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64),
                 mass_matrix: wp.array(dtype=wp.mat44d), kv: wp.array(dtype=wp.float64),
                 dt: wp.float64, n: int, h: wp.array(dtype=mat12), diagonal: wp.array(dtype=mat12)):
    b=wp.tid(); r=radius[b]; a=affine(u,n+4*b,r)
    hf=shape_hessian_psd(a)*(kv[b]/(r*r))
    result=mat12(0.)
    for i in range(12):
        for j in range(12):
            value=wp.float64(0.)
            if i%3==j%3:value=mass_matrix[b][i//3,j//3]/(dt*dt)
            if i>=3 and j>=3:value+=hf[i-3,j-3]
            result[i,j]=value
    h[b]=result; diagonal[b]=result


@wp.kernel
def body_hessian(u: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64),
                 mass_matrix: wp.array(dtype=wp.mat44d), kv: wp.array(dtype=wp.float64),
                 dt: wp.float64, n: int, h: wp.array(dtype=mat12), diagonal: wp.array(dtype=mat12)):
    b=wp.tid(); r=radius[b]; a=affine(u,n+4*b,r)
    c=wp.transpose(a)*a-wp.identity(n=3,dtype=wp.float64); aat=a*wp.transpose(a)
    hf=mat9(0.)
    for p in range(9):
        col,i=p//3,p%3
        for q in range(9):
            other,j=q//3,q%3
            v=a[i,other]*a[j,col]
            if i==j:v+=c[col,other]
            if col==other:v+=aat[i,j]
            hf[p,q]=wp.float64(4.)*kv[b]/(r*r)*v
    hf=psd9(hf)
    result=mat12(0.)
    for i in range(12):
        for j in range(12):
            value=wp.float64(0.)
            if i%3==j%3:value=mass_matrix[b][i//3,j//3]/(dt*dt)
            if i>=3 and j>=3:value+=hf[i-3,j-3]
            result[i,j]=value
    h[b]=result; diagonal[b]=result


@wp.kernel
def contact_body_diagonal(ids: wp.array(dtype=wp.vec4i), indices: wp.array(dtype=int),
                          count: wp.array(dtype=int), h: wp.array3d(dtype=wp.float64),
                          owner: wp.array(dtype=int), weights: wp.array(dtype=wp.vec4d),
                          diagonal: wp.array3d(dtype=wp.float64), lanes: int):
    lane,row,col=wp.tid()
    # Contact storage grows for exceptional broad phases. Do not launch 144
    # threads for every reserved slot on every Newton iteration. All active
    # slots are visited by a bounded grid, without a host count readback.
    for slot in range(lane,count[0],lanes):
        c=indices[slot]; ind=ids[c]
        for p in range(4):
            node=ind[p]
            if node>=0:
                body=owner[node]; first=True
                for previous in range(p):
                    if ind[previous]>=0:
                        if owner[ind[previous]]==body:first=False
                if body>=0 and first:
                    value=wp.float64(0.)
                    for a in range(4):
                        if ind[a]>=0:
                            if owner[ind[a]]==body:
                                for b in range(4):
                                    if ind[b]>=0:
                                        if owner[ind[b]]==body:
                                            value+=weights[ind[a]][row//3]*h[c,3*a+row%3,3*b+col%3]*weights[ind[b]][col//3]
                    wp.atomic_add(diagonal,body,row,col,value)


@wp.kernel
def contact_product(x: wp.array(dtype=wp.vec3d), ids: wp.array(dtype=wp.vec4i),
                    indices: wp.array(dtype=int), count: wp.array(dtype=int),
                    h: wp.array3d(dtype=wp.float64), mass: wp.array(dtype=float),
                    out: wp.array2d(dtype=wp.float64), lanes: int):
    lane,local=wp.tid()
    for slot in range(lane,count[0],lanes):
        i=indices[slot]; ind=ids[i]; row=ind[local//3]
        if row>=0:
            if mass[row]>0.:
                value=wp.float64(0.)
                for b in range(4):
                    col=ind[b]
                    if col>=0:
                        if mass[col]>0.:
                            for j in range(3):value+=h[i,local,3*b+j]*x[col][j]
                wp.atomic_add(out,row,local%3,value)


@wp.kernel
def invert_body(diagonal: wp.array(dtype=mat12), inverse: wp.array(dtype=mat12),
                stats: wp.array(dtype=wp.float64)):
    b=wp.tid(); h=diagonal[b]; scale=wp.float64(0.); valid=True
    for i in range(12):
        for j in range(12):
            valid=valid and wp.isfinite(h[i,j]); scale=wp.max(scale,wp.abs(h[i,j]))
    l=mat12(0.); result=mat12(0.)
    if valid and scale>wp.float64(0.):
        for i in range(12):
            for j in range(i+1):
                value=(h[i,j]+h[j,i])/(wp.float64(2.)*scale)
                for k in range(j):value-=l[i,k]*l[j,k]
                if i==j:
                    if value<=wp.float64(0.) or not wp.isfinite(value):valid=False
                    l[i,j]=wp.sqrt(wp.max(value,wp.float64(1.e-300)))
                else:l[i,j]=value/l[j,j]
        inv_l=mat12(0.)
        for col in range(12):
            for i in range(12):
                value=wp.float64(0.)
                if i==col:value=wp.float64(1.)
                for j in range(i):value-=l[i,j]*inv_l[j,col]
                inv_l[i,col]=value/l[i,i]
        for i in range(12):
            for j in range(12):
                value=wp.float64(0.)
                for k in range(12):value+=inv_l[k,i]*inv_l[k,j]
                result[i,j]=value/scale
                valid=valid and wp.isfinite(result[i,j])
    else:valid=False
    if not valid:wp.atomic_add(stats,2,wp.float64(1.))
    inverse[b]=result


@wp.kernel
def body_product(h: wp.array(dtype=mat12), x: wp.array(dtype=wp.vec3d), n: int,
                 out: wp.array(dtype=wp.vec3d)):
    b,row=wp.tid(); value=wp.vec3d(0.)
    for i in range(3):
        for col in range(4):
            for j in range(3):value[i]+=h[b][3*row+i,3*col+j]*x[n+4*b+col][j]
    out[n+4*b+row]+=value


@wp.kernel
def combine(x: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d),
            work: wp.array(dtype=wp.vec3d), mass: wp.array(dtype=float), n: int,
            alpha: wp.float64, beta: wp.float64, out: wp.array(dtype=wp.vec3d)):
    i=wp.tid(); value=work[i]
    if i<n:
        if mass[i]<=0.:value=x[i]
    out[i]=alpha*value+beta*y[i]


@wp.kernel
def precondition_soft(x: wp.array(dtype=wp.vec3d), inv: wp.array(dtype=wp.mat33d),
                      out: wp.array(dtype=wp.vec3d)):
    i=wp.tid(); out[i]=inv[i]*x[i]


@wp.kernel
def body_paths(old: wp.array(dtype=wp.vec3d), current: wp.array(dtype=wp.vec3d),
               radius: wp.array(dtype=wp.float64), n: int, diagnostic: wp.array(dtype=wp.float64)):
    b=wp.tid(); a=affine(old,n+4*b,radius[b]); change=affine(current,n+4*b,radius[b])-a
    minimum=determinant_path_min(a,change)
    wp.atomic_min(diagnostic,3,minimum)
    if not wp.isfinite(minimum):wp.atomic_add(diagnostic,0,wp.float64(1.))


@wp.kernel
def reset_shape(stats: wp.array(dtype=wp.float64)):
    stats[0]=wp.float64(0.); stats[1]=wp.float64(wp.inf)


@wp.kernel
def shape_check(u: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64), n: int,
                limit: wp.float64, stats: wp.array(dtype=wp.float64), status: wp.array(dtype=int)):
    b=wp.tid(); a=affine(u,n+4*b,radius[b]); c=wp.transpose(a)*a-wp.identity(n=3,dtype=wp.float64)
    error=wp.float64(0.)
    for i in range(3):
        for j in range(3):error+=c[i,j]*c[i,j]
    error=wp.sqrt(error); det=wp.determinant(a)
    wp.atomic_max(stats,0,error); wp.atomic_min(stats,1,det)
    if not wp.isfinite(error) or error>limit or not wp.isfinite(det) or det<=wp.float64(0.):
        wp.atomic_max(status,0,13)


@wp.kernel
def publish(u: wp.array(dtype=wp.vec3d), old: wp.array(dtype=wp.vec3d),
            radius: wp.array(dtype=wp.float64), com: wp.array(dtype=wp.vec3d), n: int,
            dt: wp.float64, q: wp.array(dtype=wp.vec3d), qd: wp.array(dtype=wp.vec3d),
            poses: wp.array(dtype=wp.mat44d), velocities: wp.array2d(dtype=wp.float64)):
    b=wp.tid(); base=n+4*b; a=affine(u,base,radius[b])
    da=(a-affine(old,base,radius[b]))/dt
    spin=da*wp.inverse(a); omega=wp.vec3d(spin[2,1]-spin[1,2],spin[0,2]-spin[2,0],spin[1,0]-spin[0,1])*wp.float64(.5)
    p=u[base]-a*com[b]; pose=wp.identity(n=4,dtype=wp.float64)
    for i in range(3):
        for j in range(3):pose[i,j]=a[i,j]
        pose[i,3]=p[i]
        velocities[b,i]=(u[base][i]-old[base][i])/dt
        velocities[b,3+i]=omega[i]
    poses[b]=pose
    for j in range(4):
        q[4*b+j]=u[base+j]; qd[4*b+j]=(u[base+j]-old[base+j])/dt
