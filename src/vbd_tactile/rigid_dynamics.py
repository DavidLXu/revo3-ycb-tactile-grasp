"""Implicit, partitioned two-way rigid/FEM coupling using the GPU IPC backend.

The interface unknowns are 6 DOFs per dynamic body. Every residual evaluation
solves the SAME old FEM state to a candidate rigid endpoint. GPU Broyden and,
when needed, a numerical interface Jacobian drive rigid momentum residuals to
tolerance. This is not a monolithic ABD matrix or explicit force feedback.
Host code schedules trials and transfers small pose/residual arrays; all FEM,
contact, wrench reduction, rigid integration and dense algebra run on CUDA.
"""
from dataclasses import dataclass
from time import perf_counter
import copy
import numpy as np
import warp as wp
from .mounted_ipc import rigid_poses, midpoint
from .rigid_mesh_ipc import SolverRigidMeshIPCGPU, combine
from .ipc_vbd import IPCStepRejected
from . import rigid_dynamics_gpu as k


class _SharedTimeStepRequired(IPCStepRejected):
    """Refine both systems instead of solving an inadmissible rigid path."""


@dataclass(frozen=True)
class RigidMassProperties:
    mass_kg: float
    center_m: np.ndarray
    inertia_kg_m2: np.ndarray

    def __post_init__(self):
        m=float(self.mass_kg);c=np.array(self.center_m,dtype=float,copy=True);j=np.array(self.inertia_kg_m2,dtype=float,copy=True)
        if not np.isfinite(m) or m<=0 or c.shape!=(3,) or j.shape!=(3,3) or not np.isfinite(c).all() or not np.isfinite(j).all():
            raise ValueError('finite positive mass, COM and 3x3 inertia required')
        if np.max(np.abs(j-j.T))>1e-12*np.max(np.abs(j)) or np.linalg.eigvalsh(j).min()<=0:
            raise ValueError('inertia must be symmetric positive definite about the COM')
        j=(j+j.T)*.5
        eigen=np.linalg.eigvalsh(j)
        if eigen[-1]>eigen[:2].sum()*(1+1e-10):raise ValueError('inertia violates the triangle inequality')
        c.setflags(write=False);j.setflags(write=False)
        object.__setattr__(self,'mass_kg',m);object.__setattr__(self,'center_m',c);object.__setattr__(self,'inertia_kg_m2',j)

    @classmethod
    def from_mesh(cls,mesh,*,mass_kg):
        # Exact signed tetrahedral integration for a homogeneous closed solid.
        origin=mesh.vertices.mean(0); tri=mesh.vertices[mesh.triangles]-origin
        a,b,c=tri[:,0],tri[:,1],tri[:,2]
        volume=np.einsum('ij,ij->i',a,np.cross(b,c))/6
        total=volume.sum()
        if not np.isfinite(total) or total<=0:raise ValueError('positive closed mesh volume required')
        sums=a+b+c;center=np.einsum('i,ij->j',volume,sums)/(4*total)
        second=np.einsum('i,ij,ik->jk',volume,sums,sums)
        for vertex in (a,b,c):second+=np.einsum('i,ij,ik->jk',volume,vertex,vertex)
        covariance=second/(20*total)-np.outer(center,center)
        inertia=float(mass_kg)*(np.trace(covariance)*np.eye(3)-covariance)
        return cls(mass_kg,origin+center,inertia)


@dataclass
class RigidState:
    poses: object
    velocities: object  # COM linear velocity, followed by world angular velocity.


class _Checkpoint:
    def __init__(self,s):
        self.arrays={name:wp.clone(getattr(s,name)) for name in ('contact_forces','self_contact_forces',
            '_trial_contact_forces','_trial_self_contact_forces','_certified_positions')}
        self.values={name:copy.deepcopy(getattr(s,name)) for name in ('mount_poses','_mount_requested',
            '_has_certified_positions','last_diagnostics','last_unified_trace','last_step_work') if hasattr(s,name)}

    def restore(self,s):
        for name,value in self.arrays.items():wp.copy(getattr(s,name),value)
        for name,value in self.values.items():setattr(s,name,copy.deepcopy(value))
        s._mount_device.assign(s.mount_poses)
        s.energy._evaluation=None;s.energy._hessian_ready=False


class SolverCoupledRigidIPCGPU:
    """Converged six-DOF rigid momentum coupled to an existing mesh IPC solver.

    No joint dynamics. Non-dynamic meshes and soft mount poses may remain
    prescribed. Force inputs and gravity are constant over an outer step.
    """
    def __init__(self,solver,properties,*,dynamic_ids=None,gravity=(0,0,-9.81),
                 force_tolerance_N=1e-6,max_iterations=32,max_subdivision_depth=4,warm_start=False,
                 reuse_interface=False,inexact_fem=False,retry_stalled_linear=False,interface_bfgs=False,
                 interface_schur=False,physical_path_trial_budget=4,warm_friction=False):
        if not isinstance(solver,SolverRigidMeshIPCGPU):raise TypeError('requires a rigid-mesh GPU IPC solver')
        if solver.friction_iterations<2:raise ValueError('enable iterated friction before two-way coupling')
        self.s=solver;self.device=solver.model.device;self.properties=tuple(properties)
        for name,value in (('warm_start',warm_start),('reuse_interface',reuse_interface),('inexact_fem',inexact_fem),
                           ('retry_stalled_linear',retry_stalled_linear),('interface_bfgs',interface_bfgs),
                           ('interface_schur',interface_schur),('warm_friction',warm_friction)):
            if not isinstance(value,bool):raise ValueError(f'{name} must be a bool')
        if interface_schur and solver.energy.contact_hessian=='constrained':
            raise ValueError('rigid Schur requires contact fixed/free cross blocks')
        if warm_friction and not warm_start:
            raise ValueError('warm_friction requires a guarded warm_start')
        properties=self.properties
        ids=np.arange(len(properties)) if dynamic_ids is None else np.asarray(dynamic_ids)
        if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(ids)!=len(properties) or not len(ids) or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=len(solver.meshes):raise ValueError('distinct dynamic mesh IDs must match mass properties')
        if not all(isinstance(p,RigidMassProperties) for p in properties):raise TypeError('validated mass properties required')
        self.ids=ids.copy();self.nb=len(ids);self.n=6*self.nb
        if self.nb>8:raise ValueError('this partitioned interface currently supports at most eight dynamic bodies')
        self.gravity=np.asarray(gravity,float)
        if self.gravity.shape!=(3,) or not np.isfinite(self.gravity).all():raise ValueError('gravity must be a finite three-vector')
        self.tolerance=float(force_tolerance_N)
        if not np.isfinite(self.tolerance) or self.tolerance<=0:raise ValueError('positive rigid force tolerance required')
        for value in (max_iterations,max_subdivision_depth):
            if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError('integer iteration/subdivision budgets required')
        if max_iterations<1:raise ValueError('positive iteration budget required')
        self.max_iterations=max_iterations;self.max_depth=max_subdivision_depth
        self.mass=wp.array([p.mass_kg for p in properties],dtype=wp.float64,device=self.device)
        self.com=wp.array([p.center_m for p in properties],dtype=wp.vec3d,device=self.device)
        self.inertia=wp.array([p.inertia_kg_m2 for p in properties],dtype=wp.mat33d,device=self.device)
        radii=[np.linalg.norm(solver.meshes[i].vertices-p.center_m,axis=1).max() for i,p in zip(ids,properties)]
        self.radius=wp.array(radii,dtype=wp.float64,device=self.device)
        self.origin_radius=wp.array(solver._mount_radii[solver.soft_mount_count+ids],dtype=wp.float64,device=self.device)
        owners=np.full(solver.model.particle_count,-1,int)
        for b,i in enumerate(ids):owners[solver.rigid_ranges[i]]=b
        self.owners=wp.array(owners,dtype=int,device=self.device)
        for name in ('q','trial_q','base_q','residual','base_residual','direction','work'):
            setattr(self,name,wp.zeros(self.n,dtype=wp.float64,device=self.device))
        self.inverse=wp.zeros((self.n,self.n),dtype=wp.float64,device=self.device)
        self.jacobian=wp.zeros_like(self.inverse)
        self.pose=wp.empty(self.nb,dtype=wp.mat44d,device=self.device)
        self.wrench=wp.zeros((self.nb,6),dtype=wp.float64,device=self.device)
        self.external=wp.zeros_like(self.wrench)
        self.stats=wp.zeros(1,dtype=wp.float64,device=self.device)
        self.reuse_interface=reuse_interface
        self.inexact_fem=inexact_fem
        self.retry_stalled_linear=retry_stalled_linear
        self.interface_bfgs=interface_bfgs
        if isinstance(physical_path_trial_budget,bool) or not isinstance(physical_path_trial_budget,int) or physical_path_trial_budget<1:
            raise ValueError('physical_path_trial_budget must be a positive integer')
        self.physical_path_trial_budget=physical_path_trial_budget
        self._cached_stiffness=wp.zeros_like(self.inverse)
        self._mass_matrix=wp.zeros_like(self.inverse)
        self._cached_inverse=wp.zeros_like(self.inverse)
        self._interface_valid=wp.zeros(1,dtype=int,device=self.device)
        self._input_certificate=wp.empty_like(solver._certified_positions)
        self.last_step_work=None
        self.warm_graph=None
        if warm_start:
            from .rigid_warm_start import CoupledWarmStartGraph
            self.warm_graph=CoupledWarmStartGraph(solver,warm_friction=warm_friction)
            solver._device_graph=self.warm_graph
        self.schur=None
        if interface_schur:
            from .rigid_schur import RigidSchur
            self.schur=RigidSchur(self)

    def launch(self,kernel,dim,args):wp.launch(kernel,dim,args,device=self.device)

    def state(self,*,poses=None,velocities=None):
        poses=self.s.mount_poses[self.s.soft_mount_count+self.ids] if poses is None else rigid_poses(poses)
        velocity=np.zeros((self.nb,6)) if velocities is None else np.asarray(velocities,float)
        if np.asarray(poses).shape!=(self.nb,4,4) or velocity.shape!=(self.nb,6) or not np.isfinite(velocity).all():raise ValueError('rigid state dimensions or velocity invalid')
        return RigidState(wp.array(poses,dtype=wp.mat44d,device=self.device),wp.array(velocity,dtype=wp.float64,device=self.device))

    def _solve_fem(self,old,trial,control,dt,target):
        s=self.s;original_tolerance=s.energy.linear_tolerance
        try:
            for attempt in range(2):
                previous_work=s.last_step_work
                self._evaluations+=1
                try:return s.step(old,trial,control,None,dt,mount_target=target)
                except IPCStepRejected as error:
                    work=s.last_step_work if s.last_step_work is not previous_work else None
                    rows=(work or {}).get('attempts',())
                    stalled=bool(rows) and rows[-1]['code']==6
                    if not (self.retry_stalled_linear and attempt==0 and error.retryable
                            and stalled and original_tolerance>1e-4):raise
                    # One tighter GPU linear solve before changing physical dt.
                    # Nonlinear force/CCD tolerances are unchanged. s.step has
                    # already rolled back its failed trial transactionally.
                    s.energy.linear_tolerance=max(1e-4,original_tolerance*.1)
                    self._linear_retries+=1
                finally:
                    if s.last_step_work is not previous_work:
                        for row in (s.last_step_work or {}).get('attempts',[]):
                            self._fem_work['attempts']+=1
                            self._fem_work['captures']+=int(row.get('capture_ms',0)>0)
                            for name in ('newton_iterations','pcg_iterations','hessians','backtracks',
                                         'capture_ms','wall_ms','warm_seed_accepted'):
                                self._fem_work[name]+=row.get(name,0) or 0
        finally:s.energy.linear_tolerance=original_tolerance

    def _evaluate(self,q,soft_old,rigid_old,dt,mount0,target,control):
        self.stats.zero_()
        self.launch(k.rotation_guard,self.nb,[q,self.radius,self.origin_radius,self.s.self_motion.min_gap,self.s.curved_path,self.stats])
        if not np.isfinite(self.stats.numpy()[0]):
            self._rotation_failure()
        self.launch(k.poses,self.nb,[q,rigid_old.poses,self.com,self.radius,self.pose])
        poses=self.pose.numpy()  # Only 16 doubles per body; never FEM positions.
        if not np.isfinite(poses).all():raise IPCStepRejected('nonfinite rigid trial pose')
        next_mount=target.copy();next_mount[self.s.soft_mount_count+self.ids]=poses
        self.s.mount_poses=mount0.copy();self.s._mount_device.assign(mount0)
        wp.copy(self.s._certified_positions,self._input_certificate)
        self.s._has_certified_positions=self._input_certified
        trial=self._soft_trial
        previous_work=self.s.last_step_work
        try:
            d=self._solve_fem(soft_old,trial,control,dt,next_mount)
        except IPCStepRejected as error:
            rows=((self.s.last_step_work or {}).get('attempts',[])
                  if self.s.last_step_work is not previous_work else [])
            if error.retryable and rows and rows[-1].get('code')==8:
                self._path_rejections+=1
                if self._path_rejections>=self.physical_path_trial_budget:
                    raise _SharedTimeStepRequired('physical path repeatedly uncertified; refine shared timestep') from error
            raise
        # The successful solve validated its unchanged INPUT geometry too.
        # Keep this separate from the solver's newly certified trial output.
        wp.copy(self._input_certificate,soft_old.particle_q)
        self._input_certified=True
        if d.internal_substeps!=1:raise _SharedTimeStepRequired('rigid and FEM require a shared smaller physical substep')
        if self.warm_graph is not None:
            wp.copy(self.warm_graph.warm_positions,trial.particle_q)
            wp.copy(self.warm_graph.warm_mounts,self.s._mount_device)
            self.warm_graph.warm_active.fill_(1)
        self.wrench.zero_()
        self.launch(k.wrench,self.s.model.particle_count,[trial.particle_q,self.s.energy.self_force,self.owners,self.pose,self.com,self.wrench])
        error=self._momentum_residual(q,rigid_old,dt)
        if not np.isfinite(error):raise IPCStepRejected('nonfinite rigid momentum residual')
        return trial,d,error

    def _rotation_failure(self):
        raise _SharedTimeStepRequired('rigid rotation requires a shared smaller physical substep')

    def _momentum_residual(self,q,rigid_old,dt):
        """Response operator hook; the default remains backward Euler."""
        self.stats.zero_()
        self.launch(k.balance,self.nb,[q,rigid_old.poses,self.pose,rigid_old.velocities,self.mass,self.inertia,
            self.radius,self.wrench,self.external,wp.vec3d(*self.gravity),dt,self.residual,self.stats])
        return float(self.stats.numpy()[0])

    def _accepted_rigid_state(self,dt):
        state=RigidState(wp.clone(self.pose),wp.zeros((self.nb,6),dtype=wp.float64,device=self.device))
        self.launch(k.velocities,(self.nb,6),[self.q,self.radius,dt,state.velocities])
        return state

    def _make_soft_trial(self):
        return self.s.state()

    def _numerical_inverse(self,soft_old,rigid_old,dt,mount0,target,control):
        if self.schur is not None:
            # Restore the base configuration after rejected line trials, then
            # build a condensed tangent instead of six full FEM differences.
            self._evaluate(self.base_q,soft_old,rigid_old,dt,mount0,target,control)
            if self.schur.build(dt):
                wp.copy(self.residual,self.base_residual)
                return
        # Finite-difference ONLY the small rigid interface. FEM Hessians remain
        # analytic GPU kernels. Record this expensive fallback explicitly.
        self._jacobians+=1
        h=1.e-7
        for col in range(self.n):
            value=h
            self.launch(k.perturb,self.n,[self.base_q,col,value,self.trial_q])
            try:self._evaluate(self.trial_q,soft_old,rigid_old,dt,mount0,target,control)
            except _SharedTimeStepRequired:raise
            except IPCStepRejected as rejected:
                if not rejected.retryable:raise
                value=-h
                self.launch(k.perturb,self.n,[self.base_q,col,value,self.trial_q])
                self._evaluate(self.trial_q,soft_old,rigid_old,dt,mount0,target,control)
            self.launch(k.derivative,self.n,[self.residual,self.base_residual,col,value,self.jacobian])
        self.stats.zero_();self.launch(k.invert,1,[self.jacobian,self.inverse,self.stats])
        if not np.isfinite(self.stats.numpy()[0]):raise IPCStepRejected('singular rigid interface Jacobian')
        wp.copy(self.residual,self.base_residual)

    def _attempt(self,soft_old,rigid_old,dt,target,control):
        s=self.s;checkpoint=_Checkpoint(s);mount0=s.mount_poses.copy()
        history=dict(dt_s=dt,residuals_N=[],rejected_fem_trials={})
        self._path_rejections=0
        self._interface_history.append(history)
        budgets=(s.max_depth,s.seed_subdivision_depth,s.force_tolerance)
        s.max_depth=0;s.seed_subdivision_depth=None
        # Resolve the inner system more tightly than the interface residual.
        strict_tolerance=min(s.force_tolerance,self.tolerance*.01)
        s.force_tolerance=min(s.force_tolerance,self.tolerance) if self.inexact_fem else strict_tolerance
        try:
            if self.warm_graph is not None:self.warm_graph.warm_active.zero_()
            wp.copy(self._input_certificate,s._certified_positions)
            self._input_certified=s._has_certified_positions
            # Reuse one output through all interface trials. Fresh arrays per
            # trial would force repeated CUDA graph capture. Allocate per
            # attempt so recursive half steps never overwrite ancestor input.
            self._soft_trial=self._make_soft_trial()
            self.q.zero_();self.inverse.zero_()
            self.launch(k.initial_inverse,self.nb,[rigid_old.poses,self.mass,self.inertia,self.radius,dt,self.inverse])
            if self.reuse_interface:
                self._mass_matrix.zero_()
                self.launch(k.inertial_matrix,self.nb,[rigid_old.poses,self.mass,self.inertia,self.radius,dt,self._mass_matrix])
                self.launch(k.interface_matrix,(self.n,self.n),[self._cached_stiffness,self._mass_matrix,self._interface_valid,self.jacobian])
                self.stats.zero_();self.launch(k.invert,1,[self.jacobian,self._cached_inverse,self.stats])
                self.launch(k.choose_inverse,(self.n,self.n),[self._cached_inverse,self.stats,self.inverse])
            soft,d,error=self._evaluate(self.q,soft_old,rigid_old,dt,mount0,target,control)
            for iteration in range(self.max_iterations+1):
                if s.force_tolerance>strict_tolerance and (error<=10*self.tolerance or iteration==self.max_iterations):
                    # Early interface guesses need not solve the inner system
                    # to final accuracy. Publication ALWAYS re-solves the same
                    # old state at the original strict tolerance and recomputes
                    # the rigid momentum residual; no coarse force is accepted.
                    s.force_tolerance=strict_tolerance
                    history['strict_refinement_iteration']=iteration
                    soft,d,error=self._evaluate(self.q,soft_old,rigid_old,dt,mount0,target,control)
                history['residuals_N'].append(error)
                if error<=self.tolerance:
                    if self.reuse_interface:
                        wp.copy(self.jacobian,self.inverse)
                        self.stats.zero_();self.launch(k.invert,1,[self.jacobian,self._cached_inverse,self.stats])
                        self._interface_valid.fill_(1)
                        self.launch(k.cache_stiffness,(self.n,self.n),
                            [self._cached_inverse,self._mass_matrix,self.stats,self._interface_valid,self._cached_stiffness])
                    rigid_new=self._accepted_rigid_state(dt)
                    wp.copy(soft.particle_f,soft_old.particle_f)
                    self._accepted.append(dict(dt_s=dt,rigid_residual_N=error,interface_iterations=iteration,
                        fem_residual_N=d.max_residual_N))
                    return soft,rigid_new,d
                if iteration==self.max_iterations:break
                if self.schur is not None and iteration==0:
                    self.schur.build(dt)
                wp.copy(self.base_q,self.q);wp.copy(self.base_residual,self.residual)
                accepted=False
                for retry in range(2):
                    if retry:self._numerical_inverse(soft_old,rigid_old,dt,mount0,target,control)
                    self.launch(k.direction,self.n,[self.inverse,self.base_residual,self.direction])
                    for line in range(12):
                        self.launch(k.trial,self.n,[self.base_q,self.direction,2.**(-line),self.trial_q])
                        try:new_soft,new_d,new_error=self._evaluate(self.trial_q,soft_old,rigid_old,dt,mount0,target,control)
                        except _SharedTimeStepRequired:raise
                        except IPCStepRejected as rejected:
                            if not rejected.retryable:raise
                            key=str(rejected)
                            history['rejected_fem_trials'][key]=history['rejected_fem_trials'].get(key,0)+1
                            continue
                        if new_error<error*(1-1.e-4*2.**(-line)) or new_error<=self.tolerance:
                            wp.copy(self.q,self.trial_q)
                            self.launch(k.secant,self.n,[self.q,self.base_q,self.residual,self.base_residual,self.inverse,self.work])
                            if self.interface_bfgs:
                                self.launch(k.update_inverse_bfgs,(self.n,self.n),[self.q,self.base_q,self.residual,self.base_residual,self.work,self.inverse])
                            else:
                                self.launch(k.update_inverse,(self.n,self.n),[self.residual,self.base_residual,self.work,self.inverse])
                            soft,d,error=new_soft,new_d,new_error;accepted=True;break
                    if accepted:break
                if not accepted:raise IPCStepRejected('rigid interface line search failed')
            raise IPCStepRejected('rigid momentum residual exceeds interface iteration budget')
        except BaseException:
            checkpoint.restore(s)
            raise
        finally:
            history['physical_path_rejections']=self._path_rejections
            s.max_depth,s.seed_subdivision_depth,s.force_tolerance=budgets

    def _advance(self,soft,rigid,dt,target,control,depth):
        start=self.s.mount_poses.copy();mark=len(self._accepted)
        try:return self._attempt(soft,rigid,dt,target,control)
        except IPCStepRejected as error:
            self._failures.append(dict(depth=depth,dt_s=dt,reason=str(error)))
            if not error.retryable or depth>=self.max_depth:raise
            self._accepted=self._accepted[:mark]
            mid=midpoint(start,target)
            a,b,first=self._advance(soft,rigid,dt*.5,mid,control,depth+1)
            c,d,second=self._advance(a,b,dt*.5,target,control,depth+1)
            return c,d,combine(first,second)

    def step(self,soft_in,soft_out,rigid_in,rigid_out,dt,*,mount_target=None,external_wrenches=None,control=None):
        begin=perf_counter();self._evaluations=0;self._jacobians=0;self._linear_retries=0
        self._accepted=[];self._failures=[];self._interface_history=[]
        if self.schur is not None:
            self.schur.calls=0;self.schur.pcg_iterations=0;self.schur.rejected=0
            self.schur.captures=0;self.schur.wall_ms=0.
        self._fem_work={name:0 for name in ('attempts','captures','newton_iterations','pcg_iterations','hessians',
                                          'backtracks','capture_ms','wall_ms','warm_seed_accepted')}
        accepted=False;error=None;checkpoint=_Checkpoint(self.s)
        try:
            if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite timestep required')
            friction=self.s.friction_iterations
            if isinstance(friction,(bool,np.bool_)) or not isinstance(friction,(int,np.integer)) or (friction<2 and not getattr(self,'allow_lagged_friction',False)):
                raise ValueError('two-way coupling requires at least two friction consistency iterations')
            self.s._validate_states(soft_in,soft_out)
            for state in (rigid_in,rigid_out):
                if state.poses.shape!=(self.nb,) or state.poses.dtype!=wp.mat44d or state.poses.device!=self.device or state.velocities.shape!=(self.nb,6) or state.velocities.dtype!=wp.float64 or state.velocities.device!=self.device:
                    raise ValueError('rigid state arrays must match this solver')
            initial=rigid_poses(rigid_in.poses.numpy())
            if not np.isfinite(rigid_in.velocities.numpy()).all():raise ValueError('rigid velocity must be finite')
            if not np.allclose(initial,self.s.mount_poses[self.s.soft_mount_count+self.ids],atol=1e-12,rtol=0):raise ValueError('rigid input pose must match accepted mesh state')
            target=self.s.mount_poses.copy() if mount_target is None else rigid_poses(mount_target)
            if target.shape!=self.s.mount_poses.shape:raise ValueError('mount target count cannot change')
            if not np.array_equal(target[self.s.soft_mount_count+self.ids],self.s.mount_poses[self.s.soft_mount_count+self.ids]):raise ValueError('dynamic bodies cannot also receive prescribed targets')
            external=np.zeros((self.nb,6)) if external_wrenches is None else np.asarray(external_wrenches,float)
            if external.shape!=(self.nb,6) or not np.isfinite(external).all():raise ValueError('finite world force and COM torque required per dynamic body')
            self.external.assign(external)
            out,body,d=self._advance(soft_in,rigid_in,float(dt),target,control,0)
            for name in ('particle_q','particle_qd','body_q','body_qd'):wp.copy(getattr(soft_out,name),getattr(out,name))
            wp.copy(rigid_out.poses,body.poses);wp.copy(rigid_out.velocities,body.velocities)
            self.s.last_diagnostics=d
            accepted=True;return d
        except BaseException as exc:
            self._interface_valid.zero_()
            error=str(exc) or type(exc).__name__;checkpoint.restore(self.s);raise
        finally:
            self.last_step_work=dict(accepted=accepted,error=error,wall_ms=(perf_counter()-begin)*1000,
                fem_evaluations=self._evaluations,numerical_interface_jacobians=self._jacobians,
                linear_accuracy_retries=self._linear_retries,
                schur_work=(dict(builds=self.schur.calls,pcg_iterations=self.schur.pcg_iterations,
                    rejected=self.schur.rejected,captures=self.schur.captures,wall_ms=self.schur.wall_ms)
                    if self.schur is not None else None),
                published_substeps=self._accepted if accepted else [],failed_attempts=self._failures,
                interface_attempts=self._interface_history,
                fem_work=self._fem_work,
                acceleration=dict(warm_start=self.warm_graph is not None,reuse_interface=self.reuse_interface,
                                  inexact_fem=self.inexact_fem,retry_stalled_linear=self.retry_stalled_linear,
                                  interface_bfgs=self.interface_bfgs,interface_schur=self.schur is not None,
                                  warm_friction=self.warm_graph is not None and self.warm_graph.warm_friction),
                method='partitioned implicit 6-DOF rigid momentum, GPU algebra, host orchestration')
