"""Exact five-dimensional spectral projection of unmollified IPC Hessians.

The distance Hessian spans w (Kronecker) I3 plus at most two closest-feature
coordinate directions (Kronecker) the contact normal. Mollified EE stencils
retain the original full 12-dimensional projection.
"""
import warp as wp
from .ipc_gpu_math import mat12,contact_terms,barrier,psd12
from .ipc_primitives_warp import _delta,_weights
from .ipc_gpu_contact_hessian import has_free_node
wp.set_module_options({'enable_backward':False})
mat5=wp.types.matrix((5,5),wp.float64)
mat12x5=wp.types.matrix((12,5),wp.float64)
vec5=wp.types.vector(5,wp.float64)


@wp.func
def psd5(input_h: mat5):
    """Cyclic Jacobi eigensystem; positive reconstruction is PSD even at the iteration cap."""
    a = (input_h + wp.transpose(input_h)) * wp.float64(0.5)
    vectors = wp.identity(n=5, dtype=wp.float64)
    for sweep in range(16):
        scale, off = wp.float64(0.0), wp.float64(0.0)
        for i in range(5):
            scale = wp.max(scale, wp.abs(a[i, i]))
            for j in range(i + 1, 5):
                off = wp.max(off, wp.abs(a[i, j]))
        if off <= wp.float64(1e-13) * wp.max(scale, wp.float64(1e-30)):
            break
        for p in range(4):
            for q in range(p + 1, 5):
                apq = a[p, q]
                if wp.abs(apq) > wp.float64(1e-15) * wp.max(scale, wp.float64(1e-30)):
                    tau = (a[q, q] - a[p, p]) / (wp.float64(2.0) * apq)
                    sign = wp.float64(1.0)
                    if tau < wp.float64(0.0):
                        sign = wp.float64(-1.0)
                    t = sign / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
                    c = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
                    s = t * c
                    a[p, p] = a[p, p] - t * apq
                    a[q, q] = a[q, q] + t * apq
                    a[p, q] = wp.float64(0.0)
                    a[q, p] = wp.float64(0.0)
                    for k in range(5):
                        if k != p and k != q:
                            kp, kq = a[k, p], a[k, q]
                            a[k, p] = c * kp - s * kq
                            a[p, k] = a[k, p]
                            a[k, q] = s * kp + c * kq
                            a[q, k] = a[k, q]
                        vp, vq = vectors[k, p], vectors[k, q]
                        vectors[k, p] = c * vp - s * vq
                        vectors[k, q] = s * vp + c * vq
    result = mat5(0.0)
    for i in range(5):
        for j in range(5):
            value = wp.float64(0.0)
            for k in range(5):
                value += (
                    wp.max(a[k, k], wp.float64(0.0)) * vectors[i, k] * vectors[j, k]
                )
            result[i, j] = value
    return result


@wp.func
def basis_and_coordinates(w:wp.vec4d,r:wp.vec3d,kind:int):
    u,v=wp.vec4d(0),wp.vec4d(0);rank=int(0)
    if kind==0:
        if w[1]!=wp.float64(0) and w[2]!=wp.float64(0) and w[3]!=wp.float64(0):
            u=wp.vec4d(0,1,-1,0);v=wp.vec4d(0,1,0,-1);rank=2
        else:
            lo,hi=int(-1),int(-1)
            for i in range(1,4):
                if w[i]!=wp.float64(0):
                    if lo<0:lo=i
                    else:hi=i
            if hi>=0:u[lo]=wp.float64(1);u[hi]=wp.float64(-1);rank=1
    else:
        if w[0]!=wp.float64(0) and w[1]!=wp.float64(0):u=wp.vec4d(-1,1,0,0);rank=1
        if w[2]!=wp.float64(0) and w[3]!=wp.float64(0):
            if rank==1:v=wp.vec4d(0,0,1,-1);rank=2
            else:u=wp.vec4d(0,0,1,-1);rank=1
    z=wp.normalize(w);normal=wp.normalize(r);original_u=u;original_v=v
    if rank>0:
        u=wp.normalize(u-z*wp.dot(z,u))
    if rank>1:
        v=wp.normalize(v-z*wp.dot(z,v)-u*wp.dot(u,v))
    basis=mat12x5(0)
    for i in range(12):
        basis[i,i%3]=z[i//3]
        basis[i,3]=u[i//3]*normal[i%3]
        basis[i,4]=v[i//3]*normal[i%3]
    return basis,original_u,original_v,rank


@wp.func
def contact_hessian(a:wp.vec3d,b:wp.vec3d,c:wp.vec3d,d:wp.vec3d,w:wp.vec4d,r:wp.vec3d,
                    kind:int,dhat:wp.float64,dmin:wp.float64,stiff:wp.float64,area:wp.float64):
    basis,u,v,rank=basis_and_coordinates(w,r,kind)
    bu,bv=_delta(u,a,b,c,d),_delta(v,a,b,c,d)
    iu,iv,iuv=wp.float64(0),wp.float64(0),wp.float64(0)
    if rank==1:iu=wp.float64(.5)/wp.dot(bu,bu)
    elif rank==2:
        cross=wp.cross(bu,bv);det=wp.dot(cross,cross)
        iu=wp.float64(.5)*wp.dot(bv,bv)/det
        iv=wp.float64(.5)*wp.dot(bu,bu)/det
        iuv=-wp.float64(.5)*wp.dot(bu,bv)/det
    cu,cv=vec5(0),vec5(0)
    for i in range(5):
        for j in range(12):
            cu[i]+=basis[j,i]*wp.float64(2)*(bu[j%3]*w[j//3]+u[j//3]*r[j%3])
            cv[i]+=basis[j,i]*wp.float64(2)*(bv[j%3]*w[j//3]+v[j//3]*r[j%3])
    threshold=(wp.float64(2)*dmin+dhat)*dhat
    energy,first,second=barrier(wp.dot(r,r)-dmin*dmin,threshold)
    scale=stiff*area*dhat/(threshold*threshold);first*=scale;second*=scale
    gw=vec5(0);w2=wp.dot(w,w)
    for i in range(3):gw[i]=wp.float64(2)*wp.sqrt(w2)*r[i]
    small=mat5(0)
    for i in range(5):
        for j in range(5):
            value=-iu*cu[i]*cu[j]-iv*cv[i]*cv[j]-iuv*(cu[i]*cv[j]+cv[i]*cu[j])
            if i==j and i<3:value+=wp.float64(2)*w2
            small[i,j]=first*value+second*gw[i]*gw[j]
    return basis*psd5(small)*wp.transpose(basis)


@wp.kernel
def normals(x:wp.array(dtype=wp.vec3d),ids:wp.array(dtype=wp.vec4i),kinds:wp.array(dtype=int),
            areas:wp.array(dtype=wp.float64),eps:wp.array(dtype=wp.float64),count:wp.array(dtype=int),
            dhat:wp.float64,dmin:wp.float64,stiff:wp.float64,hessians:wp.array(dtype=mat12),mass:wp.array(dtype=float)):
    i=wp.tid()
    if i<count[0]:
        ind=ids[i];h=mat12(0)
        if has_free_node(ind,mass):
            a,b,c,d=x[ind[0]],x[ind[1]],x[ind[2]],x[ind[3]];kind=kinds[i]
            w=_weights(a,b,c,d,kind);r=_delta(w,a,b,c,d);q=wp.dot(r,r)
            if wp.sqrt(q)<dmin+dhat:
                cross=wp.cross(b-a,d-c)
                if kind!=0 and wp.dot(cross,cross)<eps[i]:
                    energy,gradient,h,w=contact_terms(a,b,c,d,kind,dhat,dmin,stiff,areas[i],wp.float64(0),eps[i],True)
                    h=psd12(h)
                else:h=contact_hessian(a,b,c,d,w,r,kind,dhat,dmin,stiff,areas[i])
        hessians[i]=h
