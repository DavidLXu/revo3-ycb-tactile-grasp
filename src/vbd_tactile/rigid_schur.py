"""GPU condensed tangent for the partitioned rigid interface.

S = M_r + J.T H_rr J - (H_fr J).T H_ff^-1 (H_fr J).
Uses the projected FEM/contact tangent and the frozen friction frame. This is
a quasi-Newton search matrix, not the exact derivative of relagged friction or
a monolithic ABD solver. Every accepted step retains the original residuals.
"""
import math
from time import perf_counter
import warp as wp
from . import rigid_dynamics_gpu as k
from .ipc_gpu_nested_cg import NestedCG
from .ipc_gpu_math import mat12, contact_terms, squared_distance, psd12

wp.set_module_options({'enable_backward':False})


@wp.func
def left_jacobian(v: wp.vec3d):
    t=wp.length(v)
    a,b=wp.float64(.5),wp.float64(1./6.)
    if t>wp.float64(1.e-5):
        a=(wp.float64(1.)-wp.cos(t))/(t*t)
        b=(t-wp.sin(t))/(t*t*t)
    else:
        a-=t*t/wp.float64(24.); b-=t*t/wp.float64(120.)
    x,y,z=v[0],v[1],v[2]
    skew=wp.mat33d(0.,-z,y,z,0.,-x,-y,x,0.)
    return wp.identity(n=3,dtype=wp.float64)+a*skew+b*skew*skew


@wp.kernel
def vertex_jacobians(x: wp.array(dtype=wp.vec3d), owners: wp.array(dtype=int),
                     pose: wp.array(dtype=wp.mat44d), com: wp.array(dtype=wp.vec3d),
                     radius: wp.array(dtype=wp.float64), q: wp.array(dtype=wp.float64),
                     out: wp.array2d(dtype=wp.vec3d)):
    i,col=wp.tid(); owner=owners[i]
    value=wp.vec3d(0.)
    if owner>=0 and col//6==owner:
        axis=wp.vec3d(0.); axis[col%3]=wp.float64(1.)
        value=axis
        if col%6>=3:
            base=6*owner
            theta=wp.vec3d(q[base+3],q[base+4],q[base+5])/radius[owner]
            arm=x[i]-k.translation(pose[owner])-k.rotation(pose[owner])*com[owner]
            value=wp.cross(left_jacobian(theta)*axis,arm)/radius[owner]
    out[i,col]=value


@wp.kernel
def contact_columns(ids: wp.array(dtype=wp.vec4i), count: wp.array(dtype=int),
                    h: wp.array3d(dtype=wp.float64), j: wp.array2d(dtype=wp.vec3d),
                    columns: wp.array3d(dtype=wp.float64)):
    contact,row,col=wp.tid()
    if contact<count[0]:
        ind=ids[contact]; node=ind[row//3]
        if node>=0:
            value=wp.float64(0.)
            for b in range(4):
                if ind[b]>=0:
                    v=j[ind[b],col]
                    for axis in range(3):
                        value+=h[contact,row,3*b+axis]*v[axis]
            wp.atomic_add(columns,col,node,row%3,value)


@wp.kernel
def fixed_contact_hessian(x: wp.array(dtype=wp.vec3d), ids: wp.array(dtype=wp.vec4i),
                          kinds: wp.array(dtype=int), areas: wp.array(dtype=wp.float64),
                          eps: wp.array(dtype=wp.float64), count: wp.array(dtype=int),
                          dhat: wp.float64, dmin: wp.float64, stiffness: wp.float64,
                          mass: wp.array(dtype=float), owners: wp.array(dtype=int),
                          h: wp.array(dtype=mat12)):
    i=wp.tid()
    if i<count[0]:
        ind=ids[i]
        dynamic=False
        free=False
        for vertex in range(4):
            if ind[vertex]>=0:
                dynamic=dynamic or owners[ind[vertex]]>=0
                free=free or mass[ind[vertex]]>0.
        # FEM already computed all stencils with free soft nodes. Only recover
        # rigid/rigid or rigid/fixed terms pruned from that assembly.
        if dynamic and not free:
            a,b,c,d=x[ind[0]],x[ind[1]],x[ind[2]],x[ind[3]]
            q,gq,hq,w=squared_distance(a,b,c,d,kinds[i],False)
            block=mat12(0.)
            if wp.sqrt(q)<dmin+dhat:
                energy,g,block,w=contact_terms(a,b,c,d,kinds[i],dhat,dmin,stiffness,areas[i],wp.float64(0.),eps[i],True)
                block=psd12(block)
            h[i]=block


@wp.kernel
def project(j: wp.array2d(dtype=wp.vec3d), columns: wp.array2d(dtype=wp.vec3d),
            out: wp.array2d(dtype=wp.float64)):
    i,row,col=wp.tid()
    wp.atomic_add(out,row,col,wp.dot(j[i,row],columns[col,i]))


@wp.kernel
def rhs_column(columns: wp.array2d(dtype=wp.vec3d), mass: wp.array(dtype=float),
               col: int, rhs: wp.array(dtype=wp.vec3d)):
    i=wp.tid(); value=wp.vec3d(0.)
    if mass[i]>0.: value=columns[col,i]
    rhs[i]=value


@wp.kernel
def condense(columns: wp.array2d(dtype=wp.vec3d), solved: wp.array(dtype=wp.vec3d),
             mass: wp.array(dtype=float), col: int, out: wp.array2d(dtype=wp.float64)):
    i,row=wp.tid()
    if mass[i]>0.: wp.atomic_add(out,row,col,-wp.dot(columns[row,i],solved[i]))


@wp.kernel
def linear_error(rhs: wp.array(dtype=wp.vec3d), applied: wp.array(dtype=wp.vec3d),
                 col: int, stats: wp.array2d(dtype=wp.float64)):
    i=wp.tid(); error=rhs[i]-applied[i]
    wp.atomic_add(stats,col,0,wp.dot(error,error))
    wp.atomic_add(stats,col,1,wp.dot(rhs[i],rhs[i]))


@wp.kernel
def cg_work(iterations: wp.array(dtype=int), col: int, stats: wp.array2d(dtype=wp.float64)):
    stats[col,2]=wp.float64(iterations[0])


@wp.kernel
def finish(stiffness: wp.array2d(dtype=wp.float64), mass: wp.array2d(dtype=wp.float64),
           errors: wp.array2d(dtype=wp.float64), fem_stats: wp.array(dtype=wp.float64),
           tolerance: wp.float64, out: wp.array2d(dtype=wp.float64),
           status: wp.array(dtype=wp.float64)):
    i,j=wp.tid()
    value=(stiffness[i,j]+stiffness[j,i])*wp.float64(.5)+mass[i,j]
    out[i,j]=value
    limit=wp.max(wp.float64(1.e-10),tolerance*wp.sqrt(errors[i,1]))
    if (not wp.isfinite(value) or not wp.isfinite(errors[i,0]) or not wp.isfinite(errors[i,1])
            or errors[i,0]>limit*limit*wp.float64(1.1) or fem_stats[2]!=wp.float64(0.)):
        wp.atomic_max(status,0,wp.float64(wp.inf))


class RigidSchur:
    def __init__(self,coupled):
        c=self.c=coupled; s=c.s; n=s.model.particle_count
        self.tolerance=1.e-3
        self.j=wp.zeros((n,c.n),dtype=wp.vec3d,device=c.device)
        self.columns=wp.zeros((c.n,n),dtype=wp.vec3d,device=c.device)
        self.stiffness=wp.zeros((c.n,c.n),dtype=wp.float64,device=c.device)
        self.mass=wp.zeros_like(self.stiffness)
        self.matrix=wp.zeros_like(self.stiffness)
        self.inverse=wp.zeros_like(self.stiffness)
        self.status=wp.zeros(1,dtype=wp.float64,device=c.device)
        self.errors=wp.zeros((c.n,3),dtype=wp.float64,device=c.device)
        self.rhs=wp.zeros(n,dtype=wp.vec3d,device=c.device)
        self.solved=wp.zeros_like(self.rhs); self.applied=wp.zeros_like(self.rhs)
        self.graph=None; self.key=None
        self.calls=0; self.pcg_iterations=0; self.rejected=0; self.captures=0; self.wall_ms=0.

    def record(self,dt):
        c=self.c; s=c.s; e=s.energy; g=s._device_graph; n=s.model.particle_count
        launch=c.launch
        self.status.zero_(); self.errors.zero_(); self.stiffness.zero_(); self.columns.zero_(); self.mass.zero_()
        e.stats.zero_()
        g.hessian(s.positions,s.old,dt,g.pose_old,g.pose_new)
        # H_rr includes rigid/rigid and rigid/fixed contacts too. Such stencils
        # have no soft DOFs and may have been pruned by the FEM assembly. Build
        # only their interface contributions with the full normal kernel.
        b=e.mesh.batch
        wp.launch(fixed_contact_hessian,b.capacity,[s.positions,b.ids,b.kinds,b.areas,b.eps,b.count,
            e.mesh.dhat,e.mesh.dmin,e.mesh.stiffness,s.model.particle_mass,c.owners,b.h],device=c.device,block_dim=32)
        launch(vertex_jacobians,(n,c.n),[s.positions,c.owners,c.pose,c.com,c.radius,c.q,self.j])
        for batch in (e.mesh.batch,e.mesh_frame):
            launch(contact_columns,(batch.capacity,12,c.n),[batch.ids,batch.count,
                batch.h.view(wp.float64).reshape((batch.capacity,12,12)),self.j,
                self.columns.view(wp.float64).reshape((c.n,n,3))])
        launch(project,(n,c.n,c.n),[self.j,self.columns,self.stiffness])
        # Added fixed-only terms have no H_ff rows. Reuse the FEM matrix and
        # preconditioner without another assembly or eigendecomposition.
        for col in range(c.n):
            launch(rhs_column,n,[self.columns,s.model.particle_mass,col,self.rhs])
            self.solved.zero_()
            result=self.cg()
            e._matvec(self.solved,self.solved,self.applied,1.,0.)
            launch(linear_error,n,[self.rhs,self.applied,col,self.errors])
            launch(cg_work,1,[result[0],col,self.errors])
            launch(condense,(n,c.n),[self.columns,self.solved,s.model.particle_mass,col,self.stiffness])
        launch(k.inertial_matrix,c.nb,[c.pose,c.mass,c.inertia,c.radius,dt,self.mass])
        launch(finish,(c.n,c.n),[self.stiffness,self.mass,self.errors,e.stats,self.tolerance,self.matrix,self.status])
        launch(k.invert,1,[self.matrix,self.inverse,self.status])
        launch(k.choose_inverse,(c.n,c.n),[self.inverse,self.status,c.inverse])

    def build(self,dt):
        started=perf_counter()
        c=self.c; e=c.s.energy; g=c.s._device_graph
        if e.contact_hessian=='constrained':
            raise ValueError('rigid Schur requires contact fixed/free cross blocks')
        bindings,retained=g._cache_key(c._soft_trial,c._soft_trial,c.s._active_control,dt)
        key=(dt,self.tolerance,e.linear_max_iterations,bindings)
        if key!=self.key:
            self.cg=NestedCG(e.operator,self.rhs,self.solved,M=e.preconditioner,
                tol=self.tolerance,atol=1.e-10,maxiter=e.linear_max_iterations,
                check_every=0,use_cuda_graph=True)
            self.cg.zero_initial_guess=True
            self.rhs.zero_(); self.solved.zero_(); self.cg()
            with wp.ScopedCapture(device=c.device) as capture:self.record(dt)
            self.graph=capture.graph; self.key=key; self.references=retained
            self.captures+=1
        wp.capture_launch(self.graph)
        self.calls+=1
        self.pcg_iterations+=int(self.errors.numpy()[:,2].sum())
        valid=math.isfinite(float(self.status.numpy()[0]))
        self.rejected+=int(not valid)
        self.wall_ms+=(perf_counter()-started)*1000
        return valid
