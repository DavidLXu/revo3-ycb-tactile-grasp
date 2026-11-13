"""Unprojected normal-contact and affine-shape tangents for checked PCG.

These are derivatives of the existing energy, not a new contact law. The graph
may try them only with curvature/descent checks and a projected same-iterate
retry. Frozen friction and strain-damping approximations remain unchanged.
The rank-five basis is the same as ipc_contact_spectral; omitting its PSD clamp
preserves negative curvature instead of artificially stiffening each stencil.
"""
import warp as wp
from .ipc_gpu_math import mat9,mat12,barrier
from .ipc_primitives_warp import _delta,_weights
from .ipc_contact_spectral import basis_and_coordinates,mat5,vec5
from .ipc_gpu_contact_hessian import has_free_node
from .abd_gpu import affine
wp.set_module_options({'enable_backward':False})

@wp.func
def contact_hessian_raw(a:wp.vec3d,b:wp.vec3d,c:wp.vec3d,d:wp.vec3d,w:wp.vec4d,r:wp.vec3d,
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
    return basis*small*wp.transpose(basis)


@wp.kernel
def raw_reduced(x:wp.array(dtype=wp.vec3d),ids:wp.array(dtype=wp.vec4i),kinds:wp.array(dtype=int),
        areas:wp.array(dtype=wp.float64),eps:wp.array(dtype=wp.float64),count:wp.array(dtype=int),
        dhat:wp.float64,dmin:wp.float64,stiff:wp.float64,hessians:wp.array(dtype=mat12),
        mass:wp.array(dtype=float),pending:wp.array(dtype=int),pending_count:wp.array(dtype=int)):
    i=wp.tid()
    if i<count[0]:
        ind=ids[i];h=mat12(0)
        if areas[i]!=wp.float64(0) and has_free_node(ind,mass):
            a,b,c,d=x[ind[0]],x[ind[1]],x[ind[2]],x[ind[3]];kind=kinds[i]
            w=_weights(a,b,c,d,kind);r=_delta(w,a,b,c,d);q=wp.dot(r,r)
            if wp.sqrt(q)<dmin+dhat:
                cross=wp.cross(b-a,d-c)
                if kind!=0 and wp.dot(cross,cross)<eps[i]:
                    slot=wp.atomic_add(pending_count,0,1);pending[slot]=i
                else:h=contact_hessian_raw(a,b,c,d,w,r,kind,dhat,dmin,stiff,areas[i])
        hessians[i]=h


@wp.kernel
def raw_body_hessian(u: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64),
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
    result=mat12(0.)
    for i in range(12):
        for j in range(12):
            value=wp.float64(0.)
            if i%3==j%3:value=mass_matrix[b][i//3,j//3]/(dt*dt)
            if i>=3 and j>=3:value+=hf[i-3,j-3]
            result[i,j]=value
    h[b]=result; diagonal[b]=result

