"""GPU closest-feature surface integral for closed convex rigid solids.

Dynamic BVH search, Voronoi clipping, adaptive area quadrature and Coulomb
friction stay on the device. Immutable topology is prepared on the CPU. This
is an experimental continuum-area formulation, NOT a full GCP implementation.
The normal tangent is PSD Gauss-Newton; global IPC CCD remains mandatory.
"""
import numpy as np
import warp as wp
from .ipc_surface_reference import ConvexFeatures, _BARY, _WEIGHTS
from .ipc_primitives_warp import _weights, _delta
from .ipc_gpu_math import friction_terms

wp.set_module_options({'enable_backward': False})
vec6i=wp.types.vector(length=6,dtype=wp.int32)
vec6d=wp.types.vector(length=6,dtype=wp.float64)
vec38=wp.types.vector(length=38,dtype=wp.float64)
mat63=wp.types.matrix(shape=(6,3),dtype=wp.float64)
poly64=wp.types.matrix(shape=(64,3),dtype=wp.float64)

@wp.func
def density(d:wp.float64,dh:wp.float64,dm:wp.float64,k:wp.float64):
    s=(d-dm)/dh
    if s<=wp.float64(0):return wp.vec3d(wp.float64(wp.inf),-wp.float64(wp.inf),wp.float64(wp.inf))
    if s>wp.float64(0) and s<wp.float64(1):
        # Factor the inverse-power barrier and its derivatives. This is the
        # same potential, with a single reciprocal reused for E, E' and E''.
        # Keep powers of (1-s) near the support boundary to avoid cancellation.
        inv=wp.float64(1)/s;inv2=inv*inv;inv3=inv2*inv;inv4=inv2*inv2
        if s<wp.float64(.5):
            return wp.vec3d(k*dh*(wp.float64(0.66666666666666666667)*inv2-wp.float64(4)*(wp.float64(1)-s)),
                k*(wp.float64(4)-wp.float64(1.33333333333333333333)*inv3),wp.float64(4)*k/dh*inv4)
        t=wp.float64(1)-s
        return wp.vec3d(wp.float64(1.33333333333333333333)*k*dh*t*t*t*inv2,
            -wp.float64(1.33333333333333333333)*k*t*t*(s+wp.float64(2))*inv3,
            wp.float64(8)*k/dh*t*inv4)
    return wp.vec3d(0.)

@wp.func
def feature(p:wp.vec3d,a:wp.vec3d,b:wp.vec3d,c:wp.vec3d,kind:int):
    e=b-a;f=c-a;r=p-a;u=wp.float64(0);v=wp.float64(0)
    if kind==1:u=wp.dot(r,e)/wp.dot(e,e)
    elif kind==2:
        n=wp.cross(e,f);den=wp.dot(n,n);u=wp.dot(wp.cross(r,f),n)/den;v=wp.dot(wp.cross(e,r),n)/den
    return r-u*e-v*f,wp.vec3d(wp.float64(1)-u-v,u,v)

@wp.func
def feature_map(sa:wp.vec3d,sb:wp.vec3d,sc:wp.vec3d,a:wp.vec3d,b:wp.vec3d,c:wp.vec3d,kind:int):
    # Projection onto a fixed face/edge/corner is affine over its Voronoi cell.
    # Reuse its vertex values across both child rules and accepted samples.
    da,aa=feature(sa,a,b,c,kind);db,ab=feature(sb,a,b,c,kind);dc,ac=feature(sc,a,b,c,kind)
    return wp.matrix_from_rows(da,db,dc),wp.matrix_from_rows(aa,ab,ac)

@wp.func
def mapped_feature(beta:wp.vec3d,delta:wp.mat33d,alpha:wp.mat33d):
    # Use the same origin form as point(); do not amplify barycentric sum
    # roundoff through the absolute world translation.
    return (delta[0]+beta[1]*(delta[1]-delta[0])+beta[2]*(delta[2]-delta[0]),
            alpha[0]+beta[1]*(alpha[1]-alpha[0])+beta[2]*(alpha[2]-alpha[0]))

@wp.func
def point(beta:wp.vec3d,a:wp.vec3d,b:wp.vec3d,c:wp.vec3d):
    return a+beta[1]*(b-a)+beta[2]*(c-a)

@wp.func
def plane_clip(poly:poly64,n:int,values:wp.vec3d):
    output=poly64(0.);count=int(0)
    if n>0:
        a=poly[n-1];fa=wp.dot(a,values)
        for i in range(n):
            b=poly[i];fb=wp.dot(b,values)
            if (fa>=wp.float64(0))!=(fb>=wp.float64(0)):
                if count<64:output[count]=a+(fa/(fa-fb))*(b-a)
                count+=1
            if fb>=wp.float64(0):
                if count<64:output[count]=b
                count+=1
            a=b;fa=fb
    return output,count

@wp.kernel
def feature_bounds(x:wp.array(dtype=wp.vec3d),ids:wp.array(dtype=wp.vec3i),extent:wp.float64,
                   lo:wp.array(dtype=wp.vec3),hi:wp.array(dtype=wp.vec3)):
    i=wp.tid();ind=ids[i];a=x[ind[0]];b=x[ind[1]];c=x[ind[2]]
    scale=wp.max(wp.length(a),wp.max(wp.length(b),wp.length(c)))
    pad=extent+wp.float64(1.9073486328125e-6)*(scale+extent)
    lo[i]=wp.vec3(wp.min(a,wp.min(b,c))-wp.vec3d(pad));hi[i]=wp.vec3(wp.max(a,wp.max(b,c))+wp.vec3d(pad))

@wp.kernel
def find_pairs(x:wp.array(dtype=wp.vec3d),faces:wp.array(dtype=wp.vec3i),bvh:wp.uint64,
               pairs:wp.array(dtype=wp.vec2i),count:wp.array(dtype=int)):
    i=wp.tid();f=faces[i];a=x[f[0]];b=x[f[1]];c=x[f[2]]
    pad=wp.float64(1.9073486328125e-6)*wp.max(wp.length(a),wp.max(wp.length(b),wp.length(c)))
    query=wp.bvh_query_aabb(bvh,wp.vec3(wp.min(a,wp.min(b,c))-wp.vec3d(pad)),wp.vec3(wp.max(a,wp.max(b,c))+wp.vec3d(pad)))
    j=int(0)
    while wp.bvh_query_next(query,j):
        slot=wp.atomic_add(count,0,1)
        if slot<pairs.shape[0]:pairs[slot]=wp.vec2i(i,j)

@wp.func
def relative_motion(beta:wp.vec3d,alpha:wp.vec3d,motion:mat63):
    result=wp.vec3d(0.)
    for i in range(3):result+=beta[i]*motion[i]-alpha[i]*motion[i+3]
    return result

@wp.func
def friction_energy_gradient(displacement:wp.vec3d,normal:wp.vec3d,scale:wp.float64,eps:wp.float64):
    # Quadrature needs E/G only; the accepted samples get their Hessians later.
    tangent=displacement-normal*wp.dot(normal,displacement);r=wp.length(tangent)
    e=wp.float64(0);first_over_r=wp.float64(0)
    if r<eps:
        e=r*r/eps-r*r*r/(wp.float64(3)*eps*eps)
        first_over_r=wp.float64(2)/eps-r/(eps*eps)
    else:
        e=r-eps/wp.float64(3);first_over_r=wp.float64(1)/r
    return scale*e,scale*first_over_r*tangent

@wp.func
def rule(tri:wp.mat33d,area:wp.float64,delta_map:wp.mat33d,projection:wp.mat33d,dh:wp.float64,dm:wp.float64,k:wp.float64,
         bary:wp.array(dtype=wp.vec3d),qw:wp.array(dtype=wp.float64),motion:mat63,friction_mode:bool,mu:wp.float64,epsilon:wp.float64):
    result=vec38(0.)
    for j in range(bary.shape[0]):
        q=bary[j];beta=q[0]*tri[0]+q[1]*tri[1]+q[2]*tri[2]
        delta,alpha=mapped_feature(beta,delta_map,projection);d=wp.length(delta)
        v=density(d,dh,dm,k);w=area*qw[j];result[0]+=w*v[0]
        unit=delta/d;traction=(w*v[1])*unit
        for i in range(6):
            t=wp.float64(0)
            if i<3:t=beta[i]
            else:t=-alpha[i-3]
            g=t*traction
            for axis in range(3):result[1+3*i+axis]+=g[axis]
        if friction_mode:
            dis=relative_motion(beta,alpha,motion)
            fe,fg,fh=friction_terms(dis,unit,-mu*w*v[1],epsilon)
            result[19]+=fe
            for i in range(6):
                t=wp.float64(0)
                if i<3:t=beta[i]
                else:t=-alpha[i-3]
                for axis in range(3):result[20+3*i+axis]+=t*fg[axis]
    return result

@wp.func
def rule_moments(tri:wp.mat33d,area:wp.float64,delta_map:wp.mat33d,projection:wp.mat33d,dh:wp.float64,dm:wp.float64,k:wp.float64,
                 bary:wp.array(dtype=wp.vec3d),qw:wp.array(dtype=wp.float64),motion:mat63,friction_mode:bool,mu:wp.float64,epsilon:wp.float64):
    # The rigid weights are affine in beta[1:3]. Integrate the traction and
    # three soft barycentric moments, then reconstruct the rigid moments ONCE.
    # Keep the zeroth moment independent: this identity does not assume a
    # floating-point beta[0]+beta[1]+beta[2] == 1, or divide by a resultant
    # which can vanish through cancellation. The six nodal forces are retained.
    result=vec38(0.);total=wp.vec3d(0.);friction_total=wp.vec3d(0.)
    motion0=wp.vec3d(0.);motion1=wp.vec3d(0.);motion2=wp.vec3d(0.)
    if friction_mode:
        # alpha(beta) uses the origin form, so retain its independent constant
        # term. Do not assume beta[0]+beta[1]+beta[2] is exactly one in floats.
        motion1=motion[1];motion2=motion[2]
        for i in range(3):
            motion0+=projection[0,i]*motion[i+3]
            motion1-=(projection[1,i]-projection[0,i])*motion[i+3]
            motion2-=(projection[2,i]-projection[0,i])*motion[i+3]
    for j in range(bary.shape[0]):
        q=bary[j];beta=q[0]*tri[0]+q[1]*tri[1]+q[2]*tri[2]
        delta,alpha=mapped_feature(beta,delta_map,projection);d=wp.length(delta)
        v=density(d,dh,dm,k);w=area*qw[j];result[0]+=w*v[0]
        unit=delta/d;traction=(w*v[1])*unit;total+=traction
        for i in range(3):
            g=beta[i]*traction
            for axis in range(3):result[1+3*i+axis]+=g[axis]
        if friction_mode:
            dis=beta[0]*motion[0]+beta[1]*motion1+beta[2]*motion2-motion0
            fe,fg=friction_energy_gradient(dis,unit,-mu*w*v[1],epsilon)
            result[19]+=fe;friction_total+=fg
            for i in range(3):
                for axis in range(3):result[20+3*i+axis]+=beta[i]*fg[axis]
    for i in range(3):
        a=projection[0,i];b=projection[1,i]-a;c=projection[2,i]-a
        for axis in range(3):
            result[10+3*i+axis]=-(a*total[axis]+b*result[4+axis]+c*result[7+axis])
            if friction_mode:
                result[29+3*i+axis]=-(a*friction_total[axis]+b*result[23+axis]+c*result[26+axis])
    return result

@wp.func
def emit(tri:wp.mat33d,area:wp.float64,delta_map:wp.mat33d,projection:wp.mat33d,kind:int,extent:wp.float64,
         si:wp.vec3i,ti:wp.vec3i,bary:wp.array(dtype=wp.vec3d),qw:wp.array(dtype=wp.float64),
         ids:wp.array(dtype=vec6i),weights:wp.array(dtype=vec6d),normal:wp.array(dtype=wp.vec3d),
         areas:wp.array(dtype=wp.float64),distance:wp.array(dtype=wp.float64),kinds:wp.array(dtype=int),count:wp.array(dtype=int),cell:int,head:wp.array(dtype=int),bins:wp.array2d(dtype=int),next_sample:wp.array(dtype=int)):
    for j in range(bary.shape[0]):
        q=bary[j];beta=q[0]*tri[0]+q[1]*tri[1]+q[2]*tri[2]
        delta,alpha=mapped_feature(beta,delta_map,projection);d=wp.length(delta)
        if d<extent:
            slot=wp.atomic_add(count,0,1)
            if slot<ids.shape[0]:
                wp.atomic_exch(head,cell,slot)
                next_sample[slot]=wp.atomic_exch(bins,cell,slot%32,slot)
                ids[slot]=vec6i(si[0],si[1],si[2],ti[0],ti[1],ti[2])
                weights[slot]=vec6d(beta[0],beta[1],beta[2],-alpha[0],-alpha[1],-alpha[2])
                normal[slot]=delta/d;areas[slot]=area*qw[j];distance[slot]=d;kinds[slot]=kind

@wp.kernel
def clip_geometry(x:wp.array(dtype=wp.vec3d),sf:wp.array(dtype=wp.vec3i),ref_area:wp.array(dtype=wp.float64),
                  fi:wp.array(dtype=wp.vec3i),fk:wp.array(dtype=int),offsets:wp.array(dtype=int),planes:wp.array2d(dtype=int),
                  pairs:wp.array(dtype=wp.vec2i),pair_count:wp.array(dtype=int),dh:wp.float64,dm:wp.float64,
                  invalid:wp.array(dtype=int),cell_ids:wp.array(dtype=vec6i),
                  work_tri:wp.array(dtype=wp.mat33d),work_area:wp.array(dtype=wp.float64),
                  work_cell:wp.array(dtype=int),work_count:wp.array(dtype=int),cell_area:wp.array(dtype=wp.float64)):
    job=wp.tid()
    if job>=pair_count[0]:return
    pair=pairs[job];si=sf[pair[0]];j=pair[1];ti=fi[j];kind=fk[j]
    cell_ids[job]=vec6i(si[0],si[1],si[2],ti[0],ti[1],ti[2])
    sa=x[si[0]];sb=x[si[1]];sc=x[si[2]];a=x[ti[0]];b=x[ti[1]];c=x[ti[2]]
    poly=poly64(0.);poly[0]=wp.vec3d(1.,0.,0.);poly[1]=wp.vec3d(0.,1.,0.);poly[2]=wp.vec3d(0.,0.,1.);n=int(3)
    for p in range(offsets[j],offsets[j+1]):
        pk=planes[p,0];pa=x[planes[p,1]];pb=x[planes[p,2]];pc=x[planes[p,3]];e=pb-pa
        nn=e
        if pk==4:nn=-e
        elif pk==1:nn=wp.cross(e,pc-pa)
        elif pk==2:nn=-(pc-pa-e*wp.dot(e,pc-pa)/wp.dot(e,e))
        elif pk==3:nn=wp.cross(wp.cross(e,pc-pa),e)
        length=wp.length(nn)
        if length==wp.float64(0) or not wp.isfinite(length):wp.atomic_max(invalid,0,1);return
        nn=nn/length;values=wp.vec3d(wp.dot(sa-pa,nn),wp.dot(sb-pa,nn),wp.dot(sc-pa,nn))
        rounding=wp.float64(1.4210854715202004e-14)*(wp.max(wp.length(sa),wp.max(wp.length(sb),wp.length(sc)))+wp.length(pa)+dh+dm)
        if wp.max(wp.abs(values[0]),wp.max(wp.abs(values[1]),wp.abs(values[2])))<=rounding:
            if planes[p,4]!=0:return
            values=wp.vec3d(0.)
        poly,n=plane_clip(poly,n,values)
        if n>64:wp.atomic_max(invalid,0,2);return
        if n<3:return
    extent=dh+dm
    if kind==2:
        # Face distance is affine throughout its front-side Voronoi cell.
        # Clip its zero-density support exactly; refining triangles across a
        # slanted activation plane wastes work on an identically zero field.
        nn=wp.normalize(wp.cross(b-a,c-a))
        values=wp.vec3d(extent-wp.dot(sa-a,nn),extent-wp.dot(sb-a,nn),extent-wp.dot(sc-a,nn))
        poly,n=plane_clip(poly,n,values)
        if n>64:wp.atomic_max(invalid,0,2);return
        if n<3:return
    lo=wp.min(a,wp.min(b,c))-wp.vec3d(extent);hi=wp.max(a,wp.max(b,c))+wp.vec3d(extent)
    for axis in range(3):
        poly,n=plane_clip(poly,n,wp.vec3d(sa[axis]-lo[axis],sb[axis]-lo[axis],sc[axis]-lo[axis]))
        if n>64:wp.atomic_max(invalid,0,2);return
        poly,n=plane_clip(poly,n,wp.vec3d(hi[axis]-sa[axis],hi[axis]-sb[axis],hi[axis]-sc[axis]))
        if n>64:wp.atomic_max(invalid,0,2);return
        if n<3:return
    total_area=wp.float64(0)
    for fan in range(1,n-1):
        tri=wp.matrix_from_rows(poly[0],poly[fan],poly[fan+1]);u=tri[1]-tri[0];v=tri[2]-tri[0]
        area=ref_area[pair[0]]*wp.abs(u[1]*v[2]-u[2]*v[1])
        if area>wp.float64(0):
            slot=wp.atomic_add(work_count,0,1)
            if slot>=work_cell.shape[0]:wp.atomic_max(invalid,0,6);return
            work_tri[slot]=tri;work_area[slot]=area;work_cell[slot]=job
            total_area+=area
    cell_area[job]=total_area

@wp.kernel
def check_root_capacity(count:wp.array(dtype=int),capacity:int,invalid:wp.array(dtype=int)):
    if count[0]>capacity:wp.atomic_max(invalid,0,6)

@wp.kernel
def cache_feature_maps(x:wp.array(dtype=wp.vec3d),ids:wp.array(dtype=vec6i),
                       pairs:wp.array(dtype=wp.vec2i),kinds:wp.array(dtype=int),
                       count:wp.array(dtype=int),invalid:wp.array(dtype=int),area:wp.array(dtype=wp.float64),
                       delta:wp.array(dtype=wp.mat33d),projection:wp.array(dtype=wp.mat33d)):
    cell=wp.tid()
    if invalid[0]!=0 or cell>=count[0]:return
    if area[cell]<=wp.float64(0):return
    ind=ids[cell]
    d,p=feature_map(x[ind[0]],x[ind[1]],x[ind[2]],x[ind[3]],x[ind[4]],x[ind[5]],kinds[pairs[cell][1]])
    delta[cell]=d;projection[cell]=p

@wp.kernel
def seed_roots(tri:wp.array(dtype=wp.mat33d),area:wp.array(dtype=wp.float64),cell:wp.array(dtype=int),
               count:wp.array(dtype=int),stride:int,value:wp.array(dtype=vec38),out_tri:wp.array(dtype=wp.mat33d),
               out_area:wp.array(dtype=wp.float64),out_cell:wp.array(dtype=int),out_value:wp.array(dtype=vec38)):
    for i in range(wp.tid(),wp.min(count[0],wp.min(cell.shape[0],out_cell.shape[0])),stride):
        out_tri[i]=tri[i];out_area[i]=area[i];out_cell[i]=cell[i];out_value[i]=value[i]

def _make_rough_roots(specialization,cached=False):
    @wp.kernel(module="unique", enable_backward=False)
    def rough_roots(x:wp.array(dtype=wp.vec3d),tri:wp.array(dtype=wp.mat33d),area:wp.array(dtype=wp.float64),
                    cell:wp.array(dtype=int),count:wp.array(dtype=int),stride:int,invalid:wp.array(dtype=int),
                    cell_ids:wp.array(dtype=vec6i),pairs:wp.array(dtype=wp.vec2i),kinds:wp.array(dtype=int),
                    cell_delta:wp.array(dtype=wp.mat33d),cell_projection:wp.array(dtype=wp.mat33d),
                    dh:wp.float64,dm:wp.float64,k:wp.float64,bary:wp.array(dtype=wp.vec3d),qw:wp.array(dtype=wp.float64),
                    current:wp.array(dtype=wp.vec3d),old:wp.array(dtype=wp.vec3d),friction:bool,mu:wp.float64,eps:wp.float64,
                    rough:wp.array2d(dtype=wp.float64),root_value:wp.array(dtype=vec38)):
        if wp.static(specialization>=0):
            friction=wp.static(specialization==1)
        if invalid[0]!=0:return
        for i in range(wp.tid(),wp.min(count[0],cell.shape[0]),stride):
            job=cell[i];ind=cell_ids[job];motion=mat63(0.)
            if friction:
                for j in range(6):motion[j]=current[ind[j]]-old[ind[j]]
            delta_map=wp.mat33d(0.);projection=wp.mat33d(0.)
            if wp.static(cached):
                delta_map=cell_delta[job];projection=cell_projection[job]
            else:
                delta_map,projection=feature_map(x[ind[0]],x[ind[1]],x[ind[2]],x[ind[3]],x[ind[4]],x[ind[5]],kinds[pairs[job][1]])
            value=vec38(0.)
            if wp.static(cached):
                value=rule_moments(tri[i],area[i],delta_map,projection,dh,dm,k,bary,qw,motion,friction,mu,eps)
            else:
                value=rule(tri[i],area[i],delta_map,projection,dh,dm,k,bary,qw,motion,friction,mu,eps)
            root_value[i]=value
            for component in range(19):wp.atomic_add(rough,job,component,value[component])
            if friction:
                for component in range(19,38):wp.atomic_add(rough,job,component,value[component])
    return rough_roots

# Keep the dynamic kernel as a scheduling/mode/cache oracle. It recomputes
# projections directly from geometry, without reading the cached maps.
rough_roots=_make_rough_roots(-1)
rough_roots_normal=_make_rough_roots(0,True)
rough_roots_friction=_make_rough_roots(1,True)

def _make_refine_wave(specialization,cached=False):
    @wp.kernel(module="unique", enable_backward=False)
    def refine_wave(x:wp.array(dtype=wp.vec3d),feature_kinds:wp.array(dtype=int),pairs:wp.array(dtype=wp.vec2i),
                    cell_ids:wp.array(dtype=vec6i),cell_delta:wp.array(dtype=wp.mat33d),cell_projection:wp.array(dtype=wp.mat33d),
                    work_tri:wp.array(dtype=wp.mat33d),work_area:wp.array(dtype=wp.float64),
                    work_cell:wp.array(dtype=int),work_count:wp.array(dtype=int),work_value:wp.array(dtype=vec38),
                    next_tri:wp.array(dtype=wp.mat33d),next_area:wp.array(dtype=wp.float64),next_cell:wp.array(dtype=int),next_count:wp.array(dtype=int),next_value:wp.array(dtype=vec38),
                    dh:wp.float64,dm:wp.float64,k:wp.float64,rtol:wp.float64,depth:int,max_depth:int,
                    bary:wp.array(dtype=wp.vec3d),qw:wp.array(dtype=wp.float64),
                    ids:wp.array(dtype=vec6i),weights:wp.array(dtype=vec6d),normal:wp.array(dtype=wp.vec3d),
                    areas:wp.array(dtype=wp.float64),distance:wp.array(dtype=wp.float64),kinds:wp.array(dtype=int),count:wp.array(dtype=int),
                    invalid:wp.array(dtype=int),deepest:wp.array(dtype=int),head:wp.array(dtype=int),bins:wp.array2d(dtype=int),next_sample:wp.array(dtype=int),
                    current:wp.array(dtype=wp.vec3d),old:wp.array(dtype=wp.vec3d),friction_mode:bool,mu:wp.float64,epsilon:wp.float64,
                    cell_area:wp.array(dtype=wp.float64),rough:wp.array2d(dtype=wp.float64),
                    integral:wp.array2d(dtype=wp.float64),estimated_error:wp.array2d(dtype=wp.float64),cell_scale:wp.array(dtype=wp.float64)):
        if wp.static(specialization>=0):
            friction_mode=wp.static(specialization==1)
        index=wp.tid()
        if index>=work_count[0] or invalid[0]!=0:return
        if count[0]>ids.shape[0]:wp.atomic_max(invalid,0,5);return
        job=work_cell[index];ind=cell_ids[job];kind=feature_kinds[pairs[job][1]]
        si=wp.vec3i(ind[0],ind[1],ind[2]);ti=wp.vec3i(ind[3],ind[4],ind[5])
        tri=work_tri[index];area=work_area[index];extent=dh+dm
        motion=mat63(0.)
        if friction_mode:
            for i in range(6):motion[i]=current[ind[i]]-old[ind[i]]
        delta_map=wp.mat33d(0.);projection=wp.mat33d(0.)
        if wp.static(cached):
            delta_map=cell_delta[job];projection=cell_projection[job]
        else:
            delta_map,projection=feature_map(x[si[0]],x[si[1]],x[si[2]],x[ti[0]],x[ti[1]],x[ti[2]],kind)
        d0,al0=mapped_feature(tri[0],delta_map,projection);d1,al1=mapped_feature(tri[1],delta_map,projection);d2,al2=mapped_feature(tri[2],delta_map,projection)
        minimum=wp.min(wp.length(d0),wp.min(wp.length(d1),wp.length(d2)))
        maximum=wp.max(wp.length(d0),wp.max(wp.length(d1),wp.length(d2)))
        if kind!=2:
            w=_weights(wp.vec3d(0.),d0,d1,d2,0);minimum=wp.length(_delta(w,wp.vec3d(0.),d0,d1,d2))
        if minimum<=dm or not wp.isfinite(minimum):wp.atomic_max(invalid,0,3);return
        if minimum>=extent:return
        score0=wp.length_sq(d0-d1)/(dh*dh);score1=wp.length_sq(d1-d2)/(dh*dh);score2=wp.length_sq(d2-d0)/(dh*dh)
        if friction_mode:
            m0=relative_motion(tri[0],al0,motion);m1=relative_motion(tri[1],al1,motion);m2=relative_motion(tri[2],al2,motion)
            score0+=wp.length_sq(m0-m1)/(epsilon*epsilon);score1+=wp.length_sq(m1-m2)/(epsilon*epsilon);score2+=wp.length_sq(m2-m0)/(epsilon*epsilon)
        ia=int(0);ib=int(1);length=score0
        if score1>length:ia=1;ib=2;length=score1
        if score2>length:ia=2;ib=0
        mid=wp.float64(.5)*(tri[ia]+tri[ib]);left=tri;right=tri;left[ia]=mid;right[ib]=mid
        # A child's coarse rule is exactly the fine rule already evaluated by
        # its parent, at the same geometry/motion. Reuse it within this evaluation.
        # Never reuse these values across normal states or changed friction motion.
        coarse=work_value[index]
        left_value=vec38(0.);right_value=vec38(0.)
        if wp.static(cached):
            left_value=rule_moments(left,area*wp.float64(.5),delta_map,projection,dh,dm,k,bary,qw,motion,friction_mode,mu,epsilon)
            right_value=rule_moments(right,area*wp.float64(.5),delta_map,projection,dh,dm,k,bary,qw,motion,friction_mode,mu,epsilon)
        else:
            left_value=rule(left,area*wp.float64(.5),delta_map,projection,dh,dm,k,bary,qw,motion,friction_mode,mu,epsilon)
            right_value=rule(right,area*wp.float64(.5),delta_map,projection,dh,dm,k,bary,qw,motion,friction_mode,mu,epsilon)
        fine=left_value+right_value
        if not wp.isfinite(coarse[0]) or not wp.isfinite(fine[0]) or not wp.isfinite(fine[19]):wp.atomic_max(invalid,0,3);return
        # Resolve concentrated pressure relative to its local integral, while
        # allowing vanishing support tails a cell-area share. Using only the
        # coarse cell average can undershoot a narrow peak by orders of magnitude
        # and drive accurate leaves into the depth cap. The independent FINAL
        # summed error check still enforces the original cell tolerance.
        atol=area*k*wp.float64(1e-9)*cell_scale[job];fraction=area/cell_area[job]
        budget=wp.float64(.25)*rtol*cell_scale[job]
        ee=wp.abs(coarse[0]-fine[0]);error=ee/(budget*wp.max(wp.abs(fine[0]),fraction*wp.abs(rough[job,0]))+atol*dh)
        for node in range(6):
            err=wp.vec3d(0.);scale=wp.vec3d(0.);local=wp.vec3d(0.)
            for axis in range(3):err[axis]=coarse[1+3*node+axis]-fine[1+3*node+axis];scale[axis]=rough[job,1+3*node+axis];local[axis]=fine[1+3*node+axis]
            error=wp.max(error,wp.length(err)/(budget*wp.max(wp.length(local),fraction*wp.length(scale))+atol))
        if friction_mode:
            error=wp.max(error,wp.abs(coarse[19]-fine[19])/(budget*wp.max(wp.abs(fine[19]),fraction*wp.abs(rough[job,19]))+atol*epsilon))
            for node in range(6):
                err=wp.vec3d(0.);scale=wp.vec3d(0.);local=wp.vec3d(0.)
                for axis in range(3):err[axis]=coarse[20+3*node+axis]-fine[20+3*node+axis];scale[axis]=rough[job,20+3*node+axis];local[axis]=fine[20+3*node+axis]
                error=wp.max(error,wp.length(err)/(budget*wp.max(wp.length(local),fraction*wp.length(scale))+atol))
        bound=density(minimum,dh,dm,k)
        negligible=area*wp.abs(bound[1])*wp.max(wp.float64(1),mu)<=atol
        resolved=maximum-minimum<=wp.float64(2)*(minimum-dm)
        bounded=depth>=max_depth
        if negligible or bounded or (resolved and fine[0]>wp.float64(0) and error<=wp.float64(1)):
            for component in range(19):wp.atomic_add(integral,job,component,fine[component])
            if friction_mode:
                for component in range(19,38):wp.atomic_add(integral,job,component,fine[component])
            for mode in range(2):
                if mode==0 or friction_mode:
                    offset=19*mode;energy_error=wp.abs(coarse[offset]-fine[offset])
                    upper=area*wp.abs(bound[1])
                    if mode==1:upper*=mu
                    oscillation=wp.float64(0);coefficient=wp.float64(0)
                    if bounded:
                        # A depth cap is not permission to ignore error. Bound the
                        # entire unresolved integral using monotone pressure and
                        # affine barycentric/motion extrema on this leaf. The
                        # final cell check must still pass the ORIGINAL tolerance.
                        energy_upper=area*bound[0]
                        diameter=wp.max(wp.length(d0-d1),wp.max(wp.length(d1-d2),wp.length(d2-d0)))
                        pressure=wp.abs(bound[1]);curvature=bound[2]
                        # Positive quadrature and the true integral are averages
                        # of the same field. Their difference is bounded by its
                        # oscillation, even when that field itself is large.
                        # Along every segment of this projected triangle d>=min.
                        energy_variation=area*pressure*diameter
                        oscillation=area*(curvature+pressure/minimum)*diameter
                        coefficient=area*pressure
                        if mode==1:
                            m0=relative_motion(tri[0],al0,motion);m1=relative_motion(tri[1],al1,motion);m2=relative_motion(tri[2],al2,motion)
                            motion_bound=wp.max(wp.length(m0),wp.max(wp.length(m1),wp.length(m2)))
                            ratio=wp.min(wp.float64(1),motion_bound/epsilon)
                            potential_bound=motion_bound-epsilon/wp.float64(3)
                            if motion_bound<epsilon:potential_bound=motion_bound*ratio*(wp.float64(1)-ratio/wp.float64(3))
                            energy_upper=upper*potential_bound
                            # The regularized Coulomb force is lambda*(2r/e-r²/e²)
                            # in stick, lambda in slip. At ZERO motion its bound
                            # must be zero, rather than the full Coulomb limit.
                            upper*=ratio*(wp.float64(2)-ratio)
                            slope=ratio*(wp.float64(2)-ratio)
                            motion_diameter=wp.max(wp.length(m0-m1),wp.max(wp.length(m1-m2),wp.length(m2-m0)))
                            tangent_diameter=motion_diameter+wp.float64(2)*motion_bound*diameter/minimum
                            energy_variation=area*mu*(curvature*diameter*potential_bound+pressure*slope*tangent_diameter)
                            oscillation=area*mu*(curvature*diameter*slope+pressure*wp.float64(2)/epsilon*tangent_diameter)
                            coefficient=area*mu*pressure*slope
                        energy_error=wp.min(energy_upper+wp.abs(fine[offset]),energy_variation)
                    wp.atomic_add(estimated_error,job,7*mode,energy_error)
                    for node in range(6):
                        err=wp.vec3d(0.);fv=wp.vec3d(0.)
                        for axis in range(3):
                            err[axis]=coarse[offset+1+3*node+axis]-fine[offset+1+3*node+axis]
                            fv[axis]=fine[offset+1+3*node+axis]
                        force_error=wp.length(err)
                        if bounded:
                            ww=wp.vec3d(0.)
                            if node<3:ww=wp.vec3d(tri[0,node],tri[1,node],tri[2,node])
                            else:ww=wp.vec3d(al0[node-3],al1[node-3],al2[node-3])
                            weight=wp.max(wp.abs(ww[0]),wp.max(wp.abs(ww[1]),wp.abs(ww[2])))
                            span=wp.max(ww[0],wp.max(ww[1],ww[2]))-wp.min(ww[0],wp.min(ww[1],ww[2]))
                            # Both positive rules integrate affine weights
                            # exactly. Subtract w*f(anchor) before bounding the
                            # error: constant pressure must not be counted as
                            # uncertainty just because barycentric weights vary.
                            variation=wp.min(weight*oscillation+span*coefficient,(weight+span)*oscillation)
                            rounding=wp.float64(1.4210854715202004e-14)*upper*weight
                            force_error=wp.min(upper*weight+wp.length(fv),variation+rounding)
                        wp.atomic_add(estimated_error,job,7*mode+1+node,force_error)
            emit(left,area*wp.float64(.5),delta_map,projection,kind,extent,si,ti,bary,qw,ids,weights,normal,areas,distance,kinds,count,job,head,bins,next_sample)
            emit(right,area*wp.float64(.5),delta_map,projection,kind,extent,si,ti,bary,qw,ids,weights,normal,areas,distance,kinds,count,job,head,bins,next_sample)
            wp.atomic_max(deepest,0,depth)
        else:
            slot=wp.atomic_add(next_count,0,2)
            if slot+1>=next_cell.shape[0]:wp.atomic_max(invalid,0,6);return
            next_tri[slot]=left;next_tri[slot+1]=right
            next_area[slot]=area*wp.float64(.5);next_area[slot+1]=area*wp.float64(.5)
            next_cell[slot]=job;next_cell[slot+1]=job
            next_value[slot]=left_value;next_value[slot+1]=right_value
    return refine_wave

# Keep the dynamic kernel as an independent scheduling/mode oracle.
refine_wave=_make_refine_wave(-1)
refine_wave_normal=_make_refine_wave(0,True)
refine_wave_friction=_make_refine_wave(1,True)

@wp.kernel
def check_integral(count:wp.array(dtype=int),area:wp.array(dtype=wp.float64),value:wp.array2d(dtype=wp.float64),
                   error:wp.array2d(dtype=wp.float64),rtol:wp.float64,k:wp.float64,dh:wp.float64,
                   epsilon:wp.float64,friction:bool,invalid:wp.array(dtype=int)):
    cell=wp.tid()
    if cell>=count[0] or invalid[0]!=0:return
    atol=area[cell]*k*wp.float64(1e-9)
    for mode in range(2):
        if mode==0 or friction:
            length=dh
            if mode==1:length=epsilon
            offset=19*mode
            if error[cell,7*mode]>rtol*wp.abs(value[cell,offset])+atol*length:wp.atomic_max(invalid,0,4)
            for node in range(6):
                scale=wp.vec3d(0.)
                for axis in range(3):scale[axis]=value[cell,offset+1+3*node+axis]
                if error[cell,7*mode+1+node]>rtol*wp.length(scale)+atol:wp.atomic_max(invalid,0,4)

@wp.kernel
def check_counts(pairs:wp.array(dtype=int),samples:wp.array(dtype=int),pc:int,sc:int,invalid:wp.array(dtype=int)):
    if pairs[0]>pc or samples[0]>sc:wp.atomic_max(invalid,0,5)

@wp.kernel
def refine_cell_budgets(count:wp.array(dtype=int),area:wp.array(dtype=wp.float64),value:wp.array2d(dtype=wp.float64),
                       error:wp.array2d(dtype=wp.float64),rtol:wp.float64,k:wp.float64,dh:wp.float64,
                       epsilon:wp.float64,friction:bool,invalid:wp.array(dtype=int),scale:wp.array(dtype=wp.float64),retry:wp.array(dtype=int)):
    cell=wp.tid()
    if cell>=count[0] or invalid[0]!=4:return
    atol=area[cell]*k*wp.float64(1e-9);ratio=wp.float64(0)
    for mode in range(2):
        if mode==0 or friction:
            length=dh
            if mode==1:length=epsilon
            offset=19*mode
            target=rtol*wp.abs(value[cell,offset])+atol*length
            ratio=wp.max(ratio,error[cell,7*mode]/wp.max(target,wp.float64(1e-300)))
            for node in range(6):
                v=wp.vec3d(0.)
                for axis in range(3):v[axis]=value[cell,offset+1+3*node+axis]
                target=rtol*wp.length(v)+atol
                ratio=wp.max(ratio,error[cell,7*mode+1+node]/wp.max(target,wp.float64(1e-300)))
    if ratio>wp.float64(1):
        scale[cell]*=wp.float64(.25)/ratio
        wp.atomic_max(retry,0,1)

@wp.kernel
def count_integral_repair(repairs:wp.array(dtype=int),total:wp.array(dtype=int)):
    repairs[0]+=1;total[0]+=1

@wp.kernel
def friction_needs_refresh(x:wp.array(dtype=wp.vec3d),old:wp.array(dtype=wp.vec3d),
                           geometry:wp.array(dtype=wp.vec3d),pristine:wp.array(dtype=int),
                           refresh:wp.array(dtype=int)):
    i=wp.tid()
    if pristine[0]==0:
        wp.atomic_max(refresh,0,1);return
    for axis in range(3):
        # Exact equality, including the geometry at which the load was frozen.
        # NaN must never qualify; no displacement tolerance is used here.
        if not wp.isfinite(x[i][axis]) or x[i][axis]!=old[i][axis] or old[i][axis]!=geometry[i][axis]:
            wp.atomic_max(refresh,0,1)

@wp.kernel
def normal_value(ids:wp.array(dtype=vec6i),w:wp.array(dtype=vec6d),n:wp.array(dtype=wp.vec3d),
                 area:wp.array(dtype=wp.float64),distance:wp.array(dtype=wp.float64),count:wp.array(dtype=int),
                 invalid:wp.array(dtype=int),dh:wp.float64,dm:wp.float64,k:wp.float64,
                 gradient:wp.array(dtype=wp.vec3d),force:wp.array(dtype=wp.vec3d),stats:wp.array(dtype=wp.float64),h:wp.array(dtype=wp.mat33d)):
    i=wp.tid()
    if i>=count[0] or invalid[0]!=0:return
    val=density(distance[i],dh,dm,k);scale=area[i];wp.atomic_add(stats,0,scale*val[0])
    h[i]=scale*val[2]*wp.outer(n[i],n[i])
    for j in range(6):
        g=scale*val[1]*w[i][j]*n[i];wp.atomic_add(gradient,ids[i][j],g);wp.atomic_add(force,ids[i][j],-g)

@wp.kernel
def freeze_load(distance:wp.array(dtype=wp.float64),area:wp.array(dtype=wp.float64),count:wp.array(dtype=int),
                dh:wp.float64,dm:wp.float64,k:wp.float64,mu:wp.float64,load:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if i<count[0]:load[i]=-mu*area[i]*density(distance[i],dh,dm,k)[1]

@wp.kernel
def friction_value(x:wp.array(dtype=wp.vec3d),old:wp.array(dtype=wp.vec3d),
                   ids:wp.array(dtype=vec6i),w:wp.array(dtype=vec6d),n:wp.array(dtype=wp.vec3d),load:wp.array(dtype=wp.float64),
                   count:wp.array(dtype=int),eps:wp.float64,gradient:wp.array(dtype=wp.vec3d),
                   force:wp.array(dtype=wp.vec3d),stats:wp.array(dtype=wp.float64),h:wp.array(dtype=wp.mat33d)):
    i=wp.tid()
    if i>=count[0]:return
    d=wp.vec3d(0.)
    for j in range(6):d+=w[i][j]*(x[ids[i][j]]-old[ids[i][j]])
    e,g,H=friction_terms(d,n[i],load[i],eps);wp.atomic_add(stats,0,e);h[i]=H
    for j in range(6):
        gg=w[i][j]*g;wp.atomic_add(gradient,ids[i][j],gg);wp.atomic_add(force,ids[i][j],-gg)

@wp.kernel
def scatter_integral(ids:wp.array(dtype=vec6i),count:wp.array(dtype=int),invalid:wp.array(dtype=int),
                     integral:wp.array2d(dtype=wp.float64),offset:int,
                     gradient:wp.array(dtype=wp.vec3d),force:wp.array(dtype=wp.vec3d),stats:wp.array(dtype=wp.float64)):
    cell=wp.tid()
    if cell>=count[0] or invalid[0]!=0:return
    # These are the same accepted fine rules that emitted the samples, already
    # summed per clipping cell. Reuse their E/G instead of recomputing them and
    # atomically scattering each sample. Never publish an unresolved integral.
    wp.atomic_add(stats,0,integral[cell,offset])
    for j in range(6):
        g=wp.vec3d(0.)
        for axis in range(3):g[axis]=integral[cell,offset+1+3*j+axis]
        wp.atomic_add(gradient,ids[cell][j],g);wp.atomic_add(force,ids[cell][j],-g)

@wp.kernel
def normal_tangent(n:wp.array(dtype=wp.vec3d),area:wp.array(dtype=wp.float64),
                   distance:wp.array(dtype=wp.float64),count:wp.array(dtype=int),invalid:wp.array(dtype=int),
                   dh:wp.float64,dm:wp.float64,k:wp.float64,h:wp.array(dtype=wp.mat33d)):
    i=wp.tid()
    if i>=count[0] or invalid[0]!=0:return
    h[i]=area[i]*density(distance[i],dh,dm,k)[2]*wp.outer(n[i],n[i])

@wp.kernel
def friction_tangent(x:wp.array(dtype=wp.vec3d),old:wp.array(dtype=wp.vec3d),
                     ids:wp.array(dtype=vec6i),w:wp.array(dtype=vec6d),n:wp.array(dtype=wp.vec3d),
                     area:wp.array(dtype=wp.float64),distance:wp.array(dtype=wp.float64),count:wp.array(dtype=int),
                     invalid:wp.array(dtype=int),dh:wp.float64,dm:wp.float64,k:wp.float64,mu:wp.float64,eps:wp.float64,
                     load:wp.array(dtype=wp.float64),h:wp.array(dtype=wp.mat33d)):
    i=wp.tid()
    if i>=count[0] or invalid[0]!=0:return
    load[i]=-mu*area[i]*density(distance[i],dh,dm,k)[1]
    d=wp.vec3d(0.)
    for j in range(6):d+=w[i][j]*(x[ids[i][j]]-old[ids[i][j]])
    e,g,H=friction_terms(d,n[i],load[i],eps);h[i]=H

@wp.kernel
def product(ids:wp.array(dtype=vec6i),w:wp.array(dtype=vec6d),h:wp.array(dtype=wp.mat33d),count:wp.array(dtype=int),
            x:wp.array(dtype=wp.vec3d),out:wp.array(dtype=wp.vec3d)):
    i=wp.tid()
    if i>=count[0]:return
    d=wp.vec3d(0.)
    for j in range(6):d+=w[i][j]*x[ids[i][j]]
    v=h[i]*d
    for j in range(6):wp.atomic_add(out,ids[i][j],w[i][j]*v)

@wp.kernel
def diagonal(ids:wp.array(dtype=vec6i),w:wp.array(dtype=vec6d),h:wp.array(dtype=wp.mat33d),count:wp.array(dtype=int),out:wp.array(dtype=wp.mat33d)):
    i=wp.tid()
    if i>=count[0]:return
    for j in range(6):
        # Point/edge padding repeats IDs with zero weights. Sum equal IDs
        # before squaring, also supporting future merged embeddings.
        first=True;weight=wp.float64(0)
        for l in range(6):
            if ids[i][l]==ids[i][j]:
                weight+=w[i][l]
                if l<j:first=False
        if first:wp.atomic_add(out,ids[i][j],weight*weight*h[i])

class SurfaceFrame:
    def __init__(self,capacity,device,pair_capacity=16384):
        self.capacity=capacity;self.device=device
        self.cell_capacity=pair_capacity
        self.cell_ids=wp.zeros(pair_capacity,dtype=vec6i,device=device)
        # Geometry-epoch data only: current normal and frozen friction frames
        # own independent maps. Forces/rules/tangents are still reevaluated.
        self.cell_delta=wp.zeros(pair_capacity,dtype=wp.mat33d,device=device)
        self.cell_projection=wp.zeros_like(self.cell_delta)
        self.head=wp.full(pair_capacity,-1,dtype=int,device=device)
        self.bins=wp.full((pair_capacity,32),-1,dtype=int,device=device)
        self.next_sample=wp.zeros(capacity,dtype=int,device=device)
        self.cell_area=wp.zeros(pair_capacity,dtype=wp.float64,device=device)
        self.cell_rough=wp.zeros((pair_capacity,38),dtype=wp.float64,device=device)
        self.cell_integral=wp.zeros_like(self.cell_rough)
        self.cell_error=wp.zeros((pair_capacity,14),dtype=wp.float64,device=device)
        self.cell_scale=wp.ones(pair_capacity,dtype=wp.float64,device=device)
        self.integral_retry=wp.zeros(1,dtype=int,device=device)
        self.integral_repairs=wp.zeros(1,dtype=int,device=device)
        self.total_integral_repairs=wp.zeros(1,dtype=int,device=device)
        self.cell_h=wp.zeros((pair_capacity,18,18),dtype=wp.float64,device=device)
        for name,dtype in [('ids',vec6i),('weights',vec6d),('normals',wp.vec3d),('area',wp.float64),('distance',wp.float64),('kinds',int),('loads',wp.float64),('h',wp.mat33d)]:
            setattr(self,name,wp.zeros(capacity,dtype=dtype,device=device))
        self.count=wp.zeros(1,dtype=int,device=device)
    def product(self,x,out):wp.launch(product,self.capacity,[self.ids,self.weights,self.h,self.count,x,out],device=self.device)
    def diagonal(self,out):wp.launch(diagonal,self.capacity,[self.ids,self.weights,self.h,self.count,out],device=self.device)

@wp.kernel
def select_refinement_grid(count:wp.array(dtype=int),small:wp.array(dtype=int),
                           medium:wp.array(dtype=int),large:wp.array(dtype=int)):
    n=count[0]
    small[0]=int(n>0 and n<=1024)
    medium[0]=int(n>1024 and n<=8192)
    large[0]=int(n>8192)


class SurfaceContactGPU(SurfaceFrame):
    def __init__(self,rest,faces,groups,*,device='cuda:0',dhat=5e-5,min_gap=1e-6,stiffness=1000.,mu=0.,rtol=1e-5,max_depth=32,pair_capacity=16384,sample_capacity=1048576):
        if not wp.get_device(device).is_cuda:raise ValueError('surface runtime requires CUDA; no CPU physics fallback')
        if any(not np.isfinite(v) or v<=0 for v in (dhat,min_gap,stiffness)) or not 0<rtol<1 or not 0<=max_depth<=40 or mu<0 or not np.isfinite(mu):raise ValueError('invalid surface contact controls')
        if pair_capacity<1 or sample_capacity<1:raise ValueError('positive capacities required')
        rest=np.asarray(rest,float);faces=np.asarray(faces);groups=np.asarray(groups)
        features=ConvexFeatures.build(rest,faces,groups);slave=faces[np.all(groups[faces]<0,axis=1)]
        if not len(features.ids) or not len(slave):raise ValueError('soft surface and convex rigid solid required')
        super().__init__(sample_capacity,device,pair_capacity)
        self.dhat=float(dhat);self.dmin=float(min_gap);self.stiffness=float(stiffness);self.mu=float(mu);self.rtol=float(rtol);self.max_depth=int(max_depth)
        self.pair_capacity=int(pair_capacity);self.frame=SurfaceFrame(sample_capacity,device,pair_capacity)
        self.reuse_zero_friction=True
        # Experimental scheduling controls: disabled until CUDA trajectory
        # and timing comparisons pass. They do not change quadrature targets.
        self.bucket_refinement=False
        self.defer_tangent=False
        self.refinement_group_size=1
        self.refinement_gates=[wp.zeros(1,dtype=int,device=device) for _ in range(3)]
        self.friction_pristine=wp.zeros(1,dtype=int,device=device)
        self.friction_refresh=wp.zeros(1,dtype=int,device=device)
        self.trial_rejections=wp.zeros(1,dtype=int,device=device)
        def array(v,dtype):return wp.array(v,dtype=dtype,device=device)
        self.faces=array(slave,wp.vec3i);self.feature_ids=array(features.ids,wp.vec3i);self.feature_kinds=array(features.kinds,int)
        self.offsets=array(features.offsets,int);self.planes=array(features.planes,int)
        tri=rest[slave];self.reference_area=array(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2,wp.float64)
        self.bary=array(_BARY,wp.vec3d);self.quad_weights=array(_WEIGHTS,wp.float64)
        self.lower=wp.zeros(len(features.ids),dtype=wp.vec3,device=device);self.upper=wp.zeros_like(self.lower)
        self.bvh=wp.Bvh(self.lower,self.upper)
        for target in (self,self.frame):
            target.pairs=wp.zeros(pair_capacity,dtype=wp.vec2i,device=device)
            target.pair_count=wp.zeros(1,dtype=int,device=device)
            target.invalid=wp.zeros(1,dtype=int,device=device);target.deepest=wp.zeros_like(target.invalid)
        self.frame.geometry=array(rest,wp.vec3d)
        self.work_capacity=max(4*pair_capacity,sample_capacity//7)
        self.work_tri=[wp.zeros(self.work_capacity,dtype=wp.mat33d,device=device) for _ in range(2)]
        self.work_area=[wp.zeros(self.work_capacity,dtype=wp.float64,device=device) for _ in range(2)]
        self.work_cell=[wp.zeros(self.work_capacity,dtype=int,device=device) for _ in range(2)]
        self.work_count=[wp.zeros(1,dtype=int,device=device) for _ in range(2)]
        self.work_value=[wp.zeros(self.work_capacity,dtype=vec38,device=device) for _ in range(2)]
        self.root_value=wp.zeros(self.work_capacity,dtype=vec38,device=device)
        for target in (self,self.frame):
            target.root_tri=wp.zeros(self.work_capacity,dtype=wp.mat33d,device=device)
            target.root_area=wp.zeros(self.work_capacity,dtype=wp.float64,device=device)
            target.root_cell=wp.zeros(self.work_capacity,dtype=int,device=device)
            target.root_count=wp.zeros(1,dtype=int,device=device)
            target.geometry_invalid=wp.zeros(1,dtype=int,device=device)
    def _clip(self,geometry,f):
        f.pair_count.zero_();f.root_count.zero_();f.count.zero_();f.geometry_invalid.zero_();f.cell_area.zero_()
        wp.launch(feature_bounds,len(self.feature_ids),[geometry,self.feature_ids,self.dhat+self.dmin,self.lower,self.upper],device=self.device);self.bvh.refit()
        wp.launch(find_pairs,len(self.faces),[geometry,self.faces,self.bvh.id,f.pairs,f.pair_count],device=self.device)
        wp.launch(clip_geometry,self.pair_capacity,[geometry,self.faces,self.reference_area,self.feature_ids,self.feature_kinds,self.offsets,self.planes,
            f.pairs,f.pair_count,self.dhat,self.dmin,f.geometry_invalid,f.cell_ids,
            f.root_tri,f.root_area,f.root_cell,f.root_count,f.cell_area],device=self.device,block_dim=32)
        wp.launch(check_counts,1,[f.pair_count,f.count,self.pair_capacity,f.capacity,f.geometry_invalid],device=self.device)
        wp.launch(check_root_capacity,1,[f.root_count,self.work_capacity,f.geometry_invalid],device=self.device)
        wp.launch(cache_feature_maps,self.pair_capacity,[geometry,f.cell_ids,f.pairs,self.feature_kinds,
            f.pair_count,f.geometry_invalid,f.cell_area,f.cell_delta,f.cell_projection],device=self.device)
    def _integrate(self,geometry,target,current,old,friction_mode,epsilon):
        f=target
        f.count.zero_();wp.copy(f.invalid,f.geometry_invalid);f.deepest.zero_();f.head.fill_(-1);f.bins.fill_(-1)
        f.cell_integral.zero_();f.cell_error.zero_()
        width=min(self.work_capacity,4096)
        wp.copy(self.work_count[0],f.root_count)
        wp.launch(seed_roots,width,[f.root_tri,f.root_area,f.root_cell,f.root_count,width,self.root_value,
            self.work_tri[0],self.work_area[0],self.work_cell[0],self.work_value[0]],device=self.device)
        # A fixed sequence is captured once. Each leaf runs on its own GPU
        # thread; no host counters or per-cell serial refinement loop.
        def level(depth):
            i=depth%2;j=1-i;self.work_count[j].zero_()
            def refine(width,i=i,j=j,depth=depth):
                wp.launch(refine_wave_friction if friction_mode else refine_wave_normal,width,[geometry,self.feature_kinds,f.pairs,f.cell_ids,
                    f.cell_delta,f.cell_projection,
                    self.work_tri[i],self.work_area[i],self.work_cell[i],self.work_count[i],self.work_value[i],
                    self.work_tri[j],self.work_area[j],self.work_cell[j],self.work_count[j],self.work_value[j],
                    self.dhat,self.dmin,self.stiffness,self.rtol,depth,self.max_depth,self.bary,self.quad_weights,
                    f.ids,f.weights,f.normals,f.area,f.distance,f.kinds,f.count,f.invalid,f.deepest,f.head,f.bins,f.next_sample,
                    current,old,friction_mode,self.mu,epsilon,f.cell_area,f.cell_rough,f.cell_integral,f.cell_error,f.cell_scale],device=self.device,block_dim=64)
            if self.bucket_refinement:
                # Select launch size on the GPU. Every active leaf still gets
                # its own thread, with the original capacity/error guards.
                wp.launch(select_refinement_grid,1,[self.work_count[i],*self.refinement_gates],device=self.device)
                for gate,width in zip(self.refinement_gates,(1024,8192,self.work_capacity)):
                    wp.capture_if(gate,lambda width=width:refine(min(width,self.work_capacity)))
            else:
                wp.capture_if(self.work_count[i],lambda:refine(self.work_capacity))

        group=self.refinement_group_size
        if isinstance(group,bool) or group not in (1,4,8):
            raise ValueError('refinement_group_size must be 1, 4 or 8')
        if group==1:
            for depth in range(self.max_depth+1):level(depth)
        else:
            # Once a wave is empty, every later wave is empty. Skip the whole
            # inactive group on-device instead of executing its zero/IF nodes.
            # Group sizes are even, so every group reads count[0]. If it is
            # empty, later groups stay empty regardless of stale count[1].
            # The next evaluation resets count[0] from its fresh root count.
            for start in range(0,self.max_depth+1,group):
                def levels(start=start):
                    for depth in range(start,min(start+group,self.max_depth+1)):level(depth)
                wp.capture_if(self.work_count[0],levels)

        wp.launch(check_counts,1,[f.pair_count,f.count,self.pair_capacity,f.capacity,f.invalid],device=self.device)
        wp.launch(check_integral,self.pair_capacity,[f.pair_count,f.cell_area,f.cell_integral,f.cell_error,self.rtol,self.stiffness,self.dhat,epsilon,friction_mode,f.invalid],device=self.device)
    def _evaluate_roots(self,geometry,f,current,old,friction_mode,epsilon):
        if f is self.frame:self.friction_pristine.zero_()
        f.cell_scale.fill_(1.);f.integral_repairs.zero_();f.cell_rough.zero_()
        width=min(self.work_capacity,4096)
        wp.launch(rough_roots_friction if friction_mode else rough_roots_normal,width,[geometry,f.root_tri,f.root_area,f.root_cell,f.root_count,width,f.geometry_invalid,
            f.cell_ids,f.pairs,self.feature_kinds,f.cell_delta,f.cell_projection,self.dhat,self.dmin,self.stiffness,self.bary,self.quad_weights,
            current,old,friction_mode,self.mu,epsilon,f.cell_rough,self.root_value],device=self.device)
        self._integrate(geometry,f,current,old,friction_mode,epsilon)
        def repair():
            wp.launch(count_integral_repair,1,[f.integral_repairs,f.total_integral_repairs],device=self.device)
            self._integrate(geometry,f,current,old,friction_mode,epsilon)
        # A small resultant can hide cancellation of large local tractions.
        # If the final cell test fails, REFINE its integration budget rather
        # than making an artificial physical obstacle at that error contour.
        # Pairs stay fixed, so cell budgets retain their exact identity.
        # Runtime ABD captures these branches as device conditional nodes.
        for _ in range(4):
            f.integral_retry.zero_()
            wp.launch(refine_cell_budgets,self.pair_capacity,[f.pair_count,f.cell_area,f.cell_integral,f.cell_error,
                self.rtol,self.stiffness,self.dhat,epsilon,friction_mode,f.invalid,f.cell_scale,f.integral_retry],device=self.device)
            wp.capture_if(f.integral_retry,repair)
    def build(self,x):
        if x.dtype!=wp.vec3d:raise ValueError('surface contact requires float64 positions')
        self._clip(x,self)
        self._evaluate_roots(x,self,x,x,False,1.)
    def freeze(self,x):
        # Freeze the material contact FIELD, not a sampling density. Every
        # value can refine its quadrature for the current tangential motion.
        wp.copy(self.frame.geometry,x)
        for name in ('ids','weights','normals','area','distance','count','cell_ids','head','bins','next_sample',
                     'pairs','pair_count','cell_area','cell_delta','cell_projection','root_tri','root_area','root_cell','root_count','geometry_invalid',
                     'invalid','deepest','cell_integral','cell_error','cell_rough','cell_scale'):
            wp.copy(getattr(self.frame,name),getattr(self,name))
        self.frame.integral_repairs.zero_();self.friction_pristine.fill_(1)
        wp.launch(freeze_load,self.capacity,[self.frame.distance,self.frame.area,self.frame.count,self.dhat,self.dmin,self.stiffness,self.mu,self.frame.loads],device=self.device)
    def value(self,x,old,eps,gradient,force,stats,*,compute_tangent=True):
        wp.launch(scatter_integral,self.pair_capacity,[self.cell_ids,self.pair_count,self.invalid,self.cell_integral,0,
            gradient,force,stats],device=self.device)
        if compute_tangent:self._normal_tangent()
        if self.mu:
            f=self.frame
            def integrate():self._evaluate_roots(f.geometry,f,x,old,True,eps)
            if self.reuse_zero_friction:
                self.friction_refresh.zero_()
                wp.launch(friction_needs_refresh,len(x),[x,old,f.geometry,self.friction_pristine,self.friction_refresh],device=self.device)
                wp.capture_if(self.friction_refresh,integrate)
            else:integrate()
            wp.launch(invalid_value,1,[f.invalid,stats],device=self.device)
            wp.launch(scatter_integral,self.pair_capacity,[f.cell_ids,f.pair_count,f.invalid,f.cell_integral,19,
                gradient,force,stats],device=self.device)
            if compute_tangent:self._friction_tangent(x,old,eps)
    def tangent(self,x,old,eps):
        """Refresh sample tangents after value at exactly these positions.

        Standalone value() remains eager by default. ABD may defer this work
        until Hessian assembly, avoiding unused line-search trial tangents.
        """
        self._normal_tangent()
        if self.mu:self._friction_tangent(x,old,eps)
    def _normal_tangent(self):
        wp.launch(normal_tangent,self.capacity,[self.normals,self.area,self.distance,self.count,self.invalid,
            self.dhat,self.dmin,self.stiffness,self.h],device=self.device)
    def _friction_tangent(self,x,old,eps):
        f=self.frame
        wp.launch(friction_tangent,f.capacity,[x,old,f.ids,f.weights,f.normals,f.area,f.distance,f.count,f.invalid,
            self.dhat,self.dmin,self.stiffness,self.mu,eps,f.loads,f.h],device=self.device)
    def add_product(self,x,out):
        self.product(x,out)
        if self.mu:self.frame.product(x,out)
    def linearize(self):
        for f in ((self,self.frame) if self.mu else (self,)):
            width=min(f.cell_capacity,128)
            wp.launch(clear_active_hessian,(width,18,18),[f.pair_count,width,f.cell_h],device=self.device)
            wp.launch(group_hessian,(width,32,21),[f.bins,f.next_sample,f.weights,f.h,f.pair_count,width,f.cell_h],device=self.device)
    def add_diagonal(self,out):
        for f in ((self,self.frame) if self.mu else (self,)):
            wp.launch(cell_diagonal,(min(f.cell_capacity,128),6),[f.cell_ids,f.head,f.cell_h,f.pair_count,min(f.cell_capacity,128),out],device=self.device)

@wp.kernel
def mask_legacy(ids:wp.array(dtype=wp.vec4i),count:wp.array(dtype=int),groups:wp.array(dtype=int),area:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if i>=count[0]:return
    soft=False;rigid=False
    for j in range(4):
        if groups[ids[i][j]]<0:soft=True
        else:rigid=True
    if soft and rigid:area[i]=wp.float64(0)

@wp.func
def mapped_vector(node:int,x:wp.array(dtype=wp.vec3d),owner:wp.array(dtype=int),affine:wp.array(dtype=wp.vec4d),
                  mass:wp.array(dtype=float),axes:wp.array(dtype=wp.vec3d),n:int,reduced:bool):
    body=owner[node];v=wp.vec3d(0.)
    if body>=0:
        for r in range(4):
            value=x[n+4*body+r]
            if reduced and wp.length_sq(axes[body])>wp.float64(0):
                value=wp.vec3d(0.)
                if r==0:value=axes[body]*wp.dot(axes[body],x[n+4*body])
            v+=affine[node][r]*value
    elif mass[node]>0.:v=x[node]
    return v

@wp.kernel
def mapped_product(ids:wp.array(dtype=vec6i),w:wp.array(dtype=vec6d),h:wp.array(dtype=wp.mat33d),count:wp.array(dtype=int),
                   x:wp.array(dtype=wp.vec3d),owner:wp.array(dtype=int),affine:wp.array(dtype=wp.vec4d),
                   mass:wp.array(dtype=float),axes:wp.array(dtype=wp.vec3d),n:int,reduced:bool,scale:wp.float64,out:wp.array(dtype=wp.vec3d)):
    i=wp.tid()
    if i>=count[0]:return
    d=wp.vec3d(0.)
    for j in range(6):d+=w[i][j]*mapped_vector(ids[i][j],x,owner,affine,mass,axes,n,reduced)
    v=scale*(h[i]*d)
    for j in range(6):
        node=ids[i][j];body=owner[node];g=w[i][j]*v
        if body>=0:
            for r in range(4):
                value=affine[node][r]*g
                if reduced and wp.length_sq(axes[body])>wp.float64(0):
                    value=wp.vec3d(0.)
                    if r==0:value=axes[body]*wp.dot(axes[body],affine[node][r]*g)
                wp.atomic_add(out,n+4*body+r,value)
        elif mass[node]>0.:wp.atomic_add(out,node,g)

@wp.kernel
def body_diagonal(ids:wp.array(dtype=vec6i),w:wp.array(dtype=vec6d),h:wp.array(dtype=wp.mat33d),count:wp.array(dtype=int),
                  owner:wp.array(dtype=int),affine:wp.array(dtype=wp.vec4d),out:wp.array3d(dtype=wp.float64)):
    i=wp.tid()
    if i>=count[0]:return
    for j in range(6):
        body=owner[ids[i][j]];first=True
        for l in range(j):
            if owner[ids[i][l]]==body:first=False
        if body>=0 and first:
            coeff=wp.vec4d(0.)
            for l in range(6):
                if owner[ids[i][l]]==body:coeff+=w[i][l]*affine[ids[i][l]]
            for r in range(12):
                for s in range(12):wp.atomic_add(out,body,r,s,coeff[r//3]*coeff[s//3]*h[i][r%3,s%3])

def mask_old(surface,mesh):
    if mesh.convergent_contact or mesh.aggregation is not None:
        raise ValueError('surface integral replaces cross contacts; cannot combine with other aggregation laws')
    b=mesh.batch;wp.launch(mask_legacy,b.capacity,[b.ids,b.count,mesh.groups,b.areas],device=surface.device)

def add_joint_product(surface,c,x,out,scale):
    for f in ((surface,surface.frame) if surface.mu else (surface,)):
        wp.launch(cell_product,(min(f.cell_capacity,1024),6),[f.cell_ids,f.head,f.cell_h,f.pair_count,min(f.cell_capacity,1024),x,c.owner,c.weights,c.s.model.particle_mass,c.guide_axes,c.n,c.reduced_guides,scale,out],device=surface.device)

def add_body_diagonal(surface,c):
    for f in ((surface,surface.frame) if surface.mu else (surface,)):
        wp.launch(cell_body_diagonal,(min(f.cell_capacity,128),12,12),[f.cell_ids,f.head,f.cell_h,f.pair_count,min(f.cell_capacity,128),c.owner,c.weights,c.body_diagonal.view(wp.float64).reshape((c.nb,12,12))],device=surface.device)

@wp.kernel
def add_joint(value:wp.array(dtype=wp.vec3d),scale:wp.float64,out:wp.array(dtype=wp.vec3d)):
    i=wp.tid();out[i]+=scale*value[i]

def prepare_joint_product(surface,c,x):
    surface.joint_product.zero_()
    add_joint_product(surface,c,x,surface.joint_product,1.)

def finish_joint_product(surface,out,scale):
    wp.launch(add_joint,len(out),[surface.joint_product,scale,out],device=surface.device)

@wp.kernel
def clear_active_hessian(count:wp.array(dtype=int),stride:int,out:wp.array3d(dtype=wp.float64)):
    first,r,s=wp.tid()
    for cell in range(first,wp.min(count[0],out.shape[0]),stride):out[cell,r,s]=wp.float64(0)

@wp.kernel
def group_hessian_full(bins:wp.array2d(dtype=int),next_sample:wp.array(dtype=int),weights:wp.array(dtype=vec6d),
                  h:wp.array(dtype=wp.mat33d),count:wp.array(dtype=int),stride:int,out:wp.array3d(dtype=wp.float64)):
    first,lane,entry=wp.tid()
    for cell in range(first,wp.min(count[0],bins.shape[0]),stride):
        a=entry//6;b=entry%6;sample=bins[cell,lane]
        if sample<0:continue
        block=wp.mat33d(0.)
        while sample>=0:
            block+=weights[sample][a]*weights[sample][b]*h[sample];sample=next_sample[sample]
        for r in range(3):
            for s in range(3):wp.atomic_add(out,cell,3*a+r,3*b+s,block[r,s])

@wp.kernel
def group_hessian(bins:wp.array2d(dtype=int),next_sample:wp.array(dtype=int),weights:wp.array(dtype=vec6d),
                  h:wp.array(dtype=wp.mat33d),count:wp.array(dtype=int),stride:int,out:wp.array3d(dtype=wp.float64)):
    first,lane,entry=wp.tid()
    # H_ab = sum(w_a*w_b*H_sample) = H_ba: only 21 distinct node
    # pairs. Copy the SAME 3x3 block, not its transpose; this identity even
    # holds for an asymmetric sample H and does not silently symmetrize it.
    a=int(0);b=entry
    while b>=6-a:
        b-=6-a;a+=1
    b+=a
    for cell in range(first,wp.min(count[0],bins.shape[0]),stride):
        sample=bins[cell,lane]
        if sample<0:continue
        block=wp.mat33d(0.)
        while sample>=0:
            block+=weights[sample][a]*weights[sample][b]*h[sample];sample=next_sample[sample]
        for r in range(3):
            for s in range(3):
                wp.atomic_add(out,cell,3*a+r,3*b+s,block[r,s])
                if a!=b:wp.atomic_add(out,cell,3*b+r,3*a+s,block[r,s])

@wp.kernel
def cell_product(ids:wp.array(dtype=vec6i),head:wp.array(dtype=int),h:wp.array3d(dtype=wp.float64),count:wp.array(dtype=int),stride:int,
                 x:wp.array(dtype=wp.vec3d),owner:wp.array(dtype=int),affine:wp.array(dtype=wp.vec4d),
                 mass:wp.array(dtype=float),axes:wp.array(dtype=wp.vec3d),n:int,reduced:bool,scale:wp.float64,out:wp.array(dtype=wp.vec3d)):
    first,j=wp.tid()
    for cell in range(first,wp.min(count[0],head.shape[0]),stride):
        if head[cell]<0:continue
        g=wp.vec3d(0.)
        for l in range(6):
            v=mapped_vector(ids[cell][l],x,owner,affine,mass,axes,n,reduced)
            for r in range(3):
                for s in range(3):g[r]+=h[cell,3*j+r,3*l+s]*v[s]
        node=ids[cell][j];body=owner[node];g=scale*g
        if body>=0:
            for r in range(4):
                value=affine[node][r]*g
                if reduced and wp.length_sq(axes[body])>wp.float64(0):
                    value=wp.vec3d(0.)
                    if r==0:value=axes[body]*wp.dot(axes[body],affine[node][r]*g)
                wp.atomic_add(out,n+4*body+r,value)
        elif mass[node]>0.:wp.atomic_add(out,node,g)

@wp.kernel
def cell_diagonal(ids:wp.array(dtype=vec6i),head:wp.array(dtype=int),h:wp.array3d(dtype=wp.float64),count:wp.array(dtype=int),stride:int,out:wp.array(dtype=wp.mat33d)):
    first,j=wp.tid()
    for cell in range(first,wp.min(count[0],head.shape[0]),stride):
        if head[cell]<0:continue
        block=wp.mat33d(0.)
        for l in range(6):
            if ids[cell][l]==ids[cell][j]:
                for r in range(3):
                    for s in range(3):block[r,s]+=h[cell,3*j+r,3*l+s]
        wp.atomic_add(out,ids[cell][j],block)

@wp.kernel
def cell_body_diagonal(ids:wp.array(dtype=vec6i),head:wp.array(dtype=int),h:wp.array3d(dtype=wp.float64),count:wp.array(dtype=int),stride:int,
                       owner:wp.array(dtype=int),affine:wp.array(dtype=wp.vec4d),out:wp.array3d(dtype=wp.float64)):
    first,r,s=wp.tid()
    for cell in range(first,wp.min(count[0],head.shape[0]),stride):
        if head[cell]<0:continue
        for j in range(6):
            body=owner[ids[cell][j]]
            if body>=0:
                value=wp.float64(0)
                for l in range(6):
                    if owner[ids[cell][l]]==body:
                        value+=affine[ids[cell][j]][r//3]*affine[ids[cell][l]][s//3]*h[cell,3*j+r%3,3*l+s%3]
                wp.atomic_add(out,body,r,s,value)


@wp.kernel
def invalid_value(invalid:wp.array(dtype=int),stats:wp.array(dtype=wp.float64)):
    if invalid[0]!=0:stats[2]=wp.float64(1);stats[0]=wp.float64(wp.inf)


@wp.kernel
def contact_status(invalid:wp.array(dtype=int),status:wp.array(dtype=int)):
    if status[0]==0 and invalid[0]!=0:status[0]=50+invalid[0]
