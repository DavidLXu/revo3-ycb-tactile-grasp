"""Opt-in monolithic GPU ABD/FEM/IPC dynamics on Newton particle states.

This backend owns free affine objects. FEM backing and non-dynamic object poses
are stationary in this first adapter. External COM forces are supported; torque
commands are rejected instead of silently treating a dead affine load as torque.
The canonical body state is q/qd, not a projected six-DOF rigid transform.
"""
from dataclasses import dataclass
from numbers import Real
from time import perf_counter
import numpy as np
import warp as wp
from .rigid_dynamics import RigidMassProperties, _Checkpoint
from .rigid_mesh_ipc import SolverRigidMeshIPCGPU, combine
from .ipc_gpu_math import mat12
from .ipc_vbd import IPCStepRejected


@dataclass
class AffineState:
    q: object
    qd: object
    poses: object  # Affine mesh transform; no polar projection is applied.
    velocities: object  # Derived COM velocity and local velocity-field spin.


class SolverABDFEMIPCGPU:
    def __init__(self,solver,properties,*,dynamic_ids=None,gravity=(0,0,-9.81),
                 stiffness_pa=1.e8,max_shape_error=1.e-3,max_subdivision_depth=6,shape_hessian='jacobi',
                 operator_mode='matrix_free',stable_energy=False,contact_projection='full',inexact_friction=False,
                 rowwise_blocks=False,reuse_contact_work=False,body_inverse='scalar',translation_axes=None,
                 reduced_guides=False,fast_guided_blocks=False,friction_forcing=0.,skip_empty_contact_matvec=False,
                 skip_fixed_hessians=False,cg_group_size=1,preconditioner_mode='block',
                 reuse_refinement_buffers=False,adaptive_substep_hint=False,max_backtracks_per_attempt=0,
                 ccd_safe_step=False,split_contact_hessian=False,cooperative_contact_projection=False,merge_contact_hessians=False,
                 friction_policy="strict",friction_passes=1,force_tolerance_N=None,
                 schwarz_symmetry_cache=False,schwarz_cooperative_restriction=False,intra_step_hint=False,ccd_candidate_skin=0.,
                 failure_aware_growth=False,compact_linear_dofs=False,cooperative_matvec=False,
                 fused_pcg=False,trial_ccd_prepass=False,fused_preconditioner=False,trial_ccd_cancel=False,
                 newton_convergence='force',velocity_tolerance_m_s=.05,transform_rate_tolerance_s=.1,
                 velocity_residual_guard=True,velocity_stop_at_current=False,velocity_stop_factor=.1,
                 final_path_policy='physical_segment',temporal_tolerance_m=0.,temporal_predictive=True,
                 material_projection='always', coupled_projection=False,
                 newton_limit_policy='reject',rotation_line_search=False):
        if not isinstance(solver,SolverRigidMeshIPCGPU):raise TypeError('ABD requires the mesh IPC adapter')
        if solver.positions.dtype!=wp.vec3d or not solver.model.device.is_cuda:
            raise ValueError('ABD requires float64 CUDA states')
        self.friction_policy=friction_policy; self.friction_passes=friction_passes
        self._validate_friction_policy()
        if solver.friction_iterations<2:raise ValueError('ABD requires a friction report budget of at least two')
        if solver.model.joint_count:raise ValueError('ABD articulated joints are not implemented')
        if solver.curved_path:raise ValueError('ABD uses affine CCD paths; disable prescribed curved_path mode')
        if solver.energy.contact_hessian!='free' or solver.energy.multilevel is not None:
            raise ValueError('ABD owns its coupled Hessian and block preconditioner; use free/jacobi configuration')
        if not np.isfinite(stiffness_pa) or stiffness_pa<=0:raise ValueError('positive finite ABD stiffness required')
        if not np.isfinite(max_shape_error) or not 0<max_shape_error<1:raise ValueError('ABD shape error bound must be in (0,1)')
        if isinstance(max_subdivision_depth,bool) or not isinstance(max_subdivision_depth,int) or not 0<=max_subdivision_depth<=12:
            raise ValueError('subdivision depth must be an integer in [0,12]')
        self.s=solver; self.device=solver.model.device; self.properties=tuple(properties)
        # None preserves the historical clamp, including stricter base tolerances.
        # An explicit value sets the final joint residual AND friction-change
        # target. It changes solve accuracy, never geometry/CCD acceptance.
        self.force_tolerance_N=force_tolerance_N
        self.newton_convergence=newton_convergence
        self.newton_limit_policy=newton_limit_policy
        self.velocity_tolerance_m_s=velocity_tolerance_m_s
        self.transform_rate_tolerance_s=transform_rate_tolerance_s
        self.velocity_residual_guard=velocity_residual_guard
        self.velocity_stop_at_current=velocity_stop_at_current
        self.velocity_stop_factor=velocity_stop_factor
        from .abd_path_policy import validate_final_path_policy
        validate_final_path_policy(final_path_policy)
        self.final_path_policy=final_path_policy
        self.temporal_tolerance_m=temporal_tolerance_m
        self.temporal_predictive=temporal_predictive
        self._validate_convergence()
        self.effective_force_tolerance_N  # Validate before allocating body data.
        if not self.properties or not all(isinstance(p,RigidMassProperties) for p in self.properties):
            raise ValueError('rigid mass properties required')
        ids=np.arange(len(self.properties)) if dynamic_ids is None else np.asarray(dynamic_ids)
        if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(ids)!=len(self.properties) or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=len(solver.meshes):
            raise ValueError('distinct dynamic mesh IDs must match mass properties')
        self.ids=ids.astype(int); self.nb=len(ids); self.n=solver.model.particle_count; self.nd=self.n+4*self.nb
        axes=np.zeros((self.nb,3)) if translation_axes is None else np.asarray(translation_axes,dtype=float)
        if axes.shape!=(self.nb,3) or not np.isfinite(axes).all():
            raise ValueError('translation axes must be finite world vectors per dynamic body')
        # Scale first so a valid very small/large direction does not silently
        # become a free body or overflow during normalization.
        scales=np.max(np.abs(axes),axis=1);moving=scales>0.
        axes=axes.copy();axes[moving]/=scales[moving,None]
        lengths=np.linalg.norm(axes,axis=1)
        if not np.isfinite(lengths).all():raise ValueError('translation axis magnitude overflow')
        axes=axes.copy();moving=lengths>0.;axes[moving]/=lengths[moving,None]
        self._translation_axes=axes;self.has_guides=bool(moving.any())
        self.all_guided=bool(moving.all())
        self.reduced_guides=bool(reduced_guides)
        self.fast_guided_blocks=bool(fast_guided_blocks)
        if self.fast_guided_blocks and not (self.all_guided and self.reduced_guides):
            raise ValueError('fast guided blocks require all bodies guided and reduced_guides')
        self.guide_axes=wp.array(axes,dtype=wp.vec3d,device=self.device)
        for name in ('guide_input','guide_product','guide_pre_input','guide_pre_product'):
            setattr(self,name,wp.zeros(self.nd,dtype=wp.vec3d,device=self.device))
        self.gravity=np.asarray(gravity,float)
        if self.gravity.shape!=(3,) or not np.isfinite(self.gravity).all():raise ValueError('finite gravity vector required')
        self.stiffness_pa=float(stiffness_pa); self._built_stiffness=self.stiffness_pa
        if shape_hessian not in ('jacobi','spectral','spectral_rows'):raise ValueError('ABD shape Hessian must be jacobi, spectral or spectral_rows')
        self.shape_hessian=shape_hessian
        if operator_mode not in ('matrix_free','assembled','fused'):raise ValueError('ABD operator must be matrix_free, assembled or fused')
        self.operator_mode=operator_mode
        self.skip_empty_contact_matvec=bool(skip_empty_contact_matvec)
        self.skip_fixed_hessians=bool(skip_fixed_hessians)
        if isinstance(cg_group_size,bool) or not isinstance(cg_group_size,int) or cg_group_size not in (1,2,4,8):
            raise ValueError('CG group size must be 1, 2, 4 or 8')
        self.cg_group_size=cg_group_size
        self.material_projection=material_projection
        self.coupled_projection=bool(coupled_projection)
        self.rotation_line_search=bool(rotation_line_search)
        if preconditioner_mode not in ('block','schwarz'):raise ValueError('ABD preconditioner must be block or schwarz')
        self.preconditioner_mode=preconditioner_mode
        self.schwarz_symmetry_cache=bool(schwarz_symmetry_cache)
        self.schwarz_cooperative_restriction=bool(schwarz_cooperative_restriction)
        self.intra_step_hint=bool(intra_step_hint)
        self.failure_aware_growth=bool(failure_aware_growth)
        self.compact_linear_dofs=bool(compact_linear_dofs)
        self.cooperative_matvec=bool(cooperative_matvec)
        self.fused_pcg=bool(fused_pcg)
        self.fused_preconditioner=bool(fused_preconditioner)
        self.trial_ccd_prepass=bool(trial_ccd_prepass)
        self.trial_ccd_cancel=bool(trial_ccd_cancel)
        self.ccd_candidate_skin=ccd_candidate_skin
        self.reuse_refinement_buffers=bool(reuse_refinement_buffers)
        self.adaptive_substep_hint=bool(adaptive_substep_hint)
        if isinstance(max_backtracks_per_attempt,bool) or not isinstance(max_backtracks_per_attempt,int) or max_backtracks_per_attempt<0:
            raise ValueError('backtrack budget must be a nonnegative integer')
        self.max_backtracks_per_attempt=max_backtracks_per_attempt
        self.ccd_safe_step=bool(ccd_safe_step)
        self._validate_remaining_options()
        self.split_contact_hessian=bool(split_contact_hessian)
        self.merge_contact_hessians=bool(merge_contact_hessians)
        self.cooperative_contact_projection=bool(cooperative_contact_projection)
        self._attempt_depth=0
        self._midpoint_pool={};self._hint_dt=None;self._initial_depth=0
        self._intra_hint_dt=None;self._intra_skips=0
        self._failed_dt_cap=None;self._failure_growth_skips=0;self._successes_after_failure=0
        if self.preconditioner_mode=='schwarz' and self.operator_mode!='assembled':
            raise ValueError('joint Schwarz requires assembled operator')
        if self.skip_empty_contact_matvec and self.operator_mode!='assembled':
            raise ValueError('empty contact matvec optimization requires assembled operator')
        self.stable_energy=bool(stable_energy)
        if contact_projection not in ('full','low_rank'):raise ValueError('ABD contact projection must be full or low_rank')
        self.contact_projection=contact_projection
        self.inexact_friction=bool(inexact_friction)
        if not np.isfinite(friction_forcing) or not 0<=friction_forcing<1:
            raise ValueError('friction forcing must be finite and in [0,1)')
        self.friction_forcing=float(friction_forcing)
        if self.friction_forcing and not self.inexact_friction:
            raise ValueError('relative friction forcing requires inexact_friction')
        self.rowwise_blocks=bool(rowwise_blocks)
        self.reuse_contact_work=bool(reuse_contact_work)
        if body_inverse not in ('scalar','tiled'):raise ValueError('ABD body inverse must be scalar or tiled')
        self.body_inverse_mode=body_inverse
        self.max_shape_error=float(max_shape_error)
        self.max_depth=max_subdivision_depth; self.last_step_work=None; self._pool={}
        owner=np.full(self.n,-1,np.int32); weights=np.zeros((self.n,4)); radii=[]; matrices=[]; kv=[]
        for b,(i,p) in enumerate(zip(ids,self.properties)):
            mesh=solver.meshes[i]; local=mesh.vertices-p.center_m
            radius=float(np.linalg.norm(local,axis=1).max())
            covariance=.5*np.trace(p.inertia_kg_m2)*np.eye(3)-p.inertia_kg_m2
            if radius<=0 or np.linalg.eigvalsh(covariance).min()<=0:
                raise ValueError('ABD needs a nondegenerate volumetric mass distribution')
            tri=local[mesh.triangles]
            volume=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
            if not np.isfinite(volume) or volume<=0:raise ValueError('positive closed-body volume required')
            matrix=np.zeros((4,4)); matrix[0,0]=p.mass_kg; matrix[1:,1:]=covariance/radius**2
            matrices.append(matrix); radii.append(radius); kv.append(stiffness_pa*volume)
            bounds=solver.rigid_ranges[i]; owner[bounds]=b; weights[bounds,0]=1.; weights[bounds,1:]=local/radius
        self._radii=np.array(radii)
        def array(value,dtype):return wp.array(value,dtype=dtype,device=self.device)
        self.owner=array(owner,int); self.weights=array(weights,wp.vec4d)
        self.radius=array(radii,wp.float64); self.mass_matrix=array(matrices,wp.mat44d)
        self.mass=array([p.mass_kg for p in self.properties],wp.float64)
        self.com=array([p.center_m for p in self.properties],wp.vec3d); self.kv=array(kv,wp.float64)
        masses=solver.model.particle_mass.numpy()
        self.mobile_mass=array(np.where((masses>0)|(owner>=0),1.,0.).astype(np.float32),float)
        self.dof_mass=array(np.r_[masses,np.ones(4*self.nb)].astype(np.float32),float)
        for name in ('u','uold','base','gradient','line_gradient','rhs','direction','applied','product','preconditioned'):
            setattr(self,name,wp.zeros(self.nd,dtype=wp.vec3d,device=self.device))
        for name in ('full_direction','full_product','line_physical_gradient'):
            setattr(self,name,wp.zeros(self.n,dtype=wp.vec3d,device=self.device))
        for name in ('old_q','old_qd','predictor','next_q','next_qd'):
            setattr(self,name,wp.zeros(4*self.nb,dtype=wp.vec3d,device=self.device))
        for name in ('body_h','body_diagonal','body_inverse'):
            setattr(self,name,wp.zeros(self.nb,dtype=mat12,device=self.device))
        self.external=wp.zeros(self.nb,dtype=wp.vec3d,device=self.device)
        self.next_poses=wp.zeros(self.nb,dtype=wp.mat44d,device=self.device)
        self.next_velocities=wp.zeros((self.nb,6),dtype=wp.float64,device=self.device)
        self.shape_stats=wp.zeros(2,dtype=wp.float64,device=self.device)
        from .abd_graph import ABDDeviceGraph
        self.graph=ABDDeviceGraph(self)
        if self.temporal_tolerance_m:self.graph.cache_capacity=64

    def state(self,*,velocities=None):
        poses=self.s.mount_poses[self.s.soft_mount_count+self.ids].copy()
        v=np.zeros((self.nb,6)) if velocities is None else np.asarray(velocities,float)
        if v.shape!=(self.nb,6) or not np.isfinite(v).all():raise ValueError('finite body velocity array required')
        for b,axis in enumerate(self._translation_axes):
            if np.dot(axis,axis)>0. and (np.linalg.norm(v[b,:3]-axis*np.dot(axis,v[b,:3]))>1e-12 or np.linalg.norm(v[b,3:])>1e-12):
                raise ValueError('initial velocity violates translation guide')
        q=np.zeros((4*self.nb,3)); qd=np.zeros_like(q)
        for b,(p,r) in enumerate(zip(self.properties,self._radii)):
            a=poses[b,:3,:3]; w=v[b,3:]
            skew=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
            q[4*b]=poses[b,:3,3]+a@p.center_m; q[4*b+1:4*b+4]=r*a.T
            qd[4*b]=v[b,:3]; qd[4*b+1:4*b+4]=r*(skew@a).T
        return AffineState(wp.array(q,dtype=wp.vec3d,device=self.device),wp.array(qd,dtype=wp.vec3d,device=self.device),
            wp.array(poses,dtype=wp.mat44d,device=self.device),wp.array(v,dtype=wp.float64,device=self.device))

    @property
    def effective_force_tolerance_N(self):
        value=self.force_tolerance_N
        if value is None:value=min(self.s.force_tolerance,1.e-8)
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<=0:
            raise ValueError('ABD force tolerance must be a positive finite scalar or None')
        return float(value)

    @property
    def effective_backtrack_budget(self):
        # Early rejection only helps if a smaller physical interval remains.
        # At the finest allowed step, preserve the original full solve budget.
        return self.max_backtracks_per_attempt if self._attempt_depth<self.max_depth else 0

    def _validate_convergence(self):
        policy=getattr(self,'newton_limit_policy','reject')
        if policy not in ('reject','accept_feasible'):
            raise ValueError('Newton limit policy must be reject or accept_feasible')
        if policy=='accept_feasible' and (self.newton_convergence!='velocity' or
                self.friction_policy!='lagged' or self.friction_passes!=1):
            raise ValueError('accept_feasible requires velocity stopping and one lagged friction pass')
        temporal=getattr(self,'temporal_tolerance_m',0.)
        if not isinstance(getattr(self,'temporal_predictive',True),(bool,np.bool_)):
            raise ValueError('temporal predictive scheduling must be boolean')
        if isinstance(temporal,(bool,np.bool_)) or not isinstance(temporal,Real) or not np.isfinite(temporal) or temporal<0:
            raise ValueError('temporal tolerance must be a finite nonnegative scalar in metres')
        if not isinstance(self.velocity_residual_guard,(bool,np.bool_)):
            raise ValueError('velocity_residual_guard must be boolean')
        if not isinstance(getattr(self,'velocity_stop_at_current',False),(bool,np.bool_)):
            raise ValueError('velocity_stop_at_current must be boolean')
        factor=getattr(self,'velocity_stop_factor',.1)
        if isinstance(factor,(bool,np.bool_)) or not isinstance(factor,Real) or not np.isfinite(factor) or not 0<factor<=1:
            raise ValueError('velocity_stop_factor must be a finite scalar in (0, 1]')
        if self.newton_convergence not in ('force','velocity'):
            raise ValueError('Newton convergence must be force or velocity')
        for value in (self.velocity_tolerance_m_s,self.transform_rate_tolerance_s):
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<=0:
                raise ValueError('Newton velocity and transform-rate tolerances must be positive finite scalars')
        if self.newton_convergence=='velocity':
            if self.friction_policy!='lagged' or self.friction_passes!=1:
                raise ValueError('velocity convergence requires one frame-lagged friction solve')
            if self.force_tolerance_N is not None:
                raise ValueError('velocity convergence does not use a force tolerance; omit force_tolerance_N')

    def _validate_friction_policy(self):
        if self.friction_policy not in ('strict','lagged','newton'):
            raise ValueError('friction policy must be strict, lagged or newton')
        if isinstance(self.friction_passes,bool) or not isinstance(self.friction_passes,int) or not 1<=self.friction_passes<=32:
            raise ValueError('friction passes must be an integer in [1,32]')
        if self.friction_policy!='lagged' and self.friction_passes!=1:
            raise ValueError('friction passes only applies to lagged policy')

    def _validate_remaining_options(self):
        if self.fused_preconditioner and not self.compact_linear_dofs:
            raise ValueError('fused preconditioner requires compact linear DOFs')
        if self.fused_pcg and not (self.compact_linear_dofs and self.cooperative_matvec and self.skip_empty_contact_matvec):
            raise ValueError('fused PCG requires compact cooperative matvec and empty-tail routing')
        if self.material_projection not in ('always','on_demand'):
            raise ValueError('material projection must be always or on_demand')
        if self.cooperative_matvec and self.operator_mode!='assembled':
            raise ValueError('cooperative matvec requires assembled operator')
        if self.compact_linear_dofs and (self.operator_mode!='assembled' or self.preconditioner_mode!='schwarz'):
            raise ValueError('compact linear DOFs require assembled operator and Schwarz')
        if self.compact_linear_dofs and self.has_guides and not self.reduced_guides:
            raise ValueError('compact linear DOFs require reduced guides for guided bodies')
        if self.schwarz_symmetry_cache and self.preconditioner_mode!='schwarz':
            raise ValueError('Schwarz symmetry cache requires schwarz preconditioner')
        if self.schwarz_cooperative_restriction and self.preconditioner_mode!='schwarz':
            raise ValueError('Schwarz cooperative restriction requires schwarz preconditioner')
        skin=self.ccd_candidate_skin
        if isinstance(skin,(bool,np.bool_)) or not isinstance(skin,Real) or not np.isfinite(skin) or skin<0:
            raise ValueError('CCD candidate skin must be a nonnegative finite scalar')
        if skin and not self.ccd_safe_step:
            raise ValueError('CCD candidate skin requires CCD safe step')

    def _attempt(self,soft,body,dt,control,depth):
        self._attempt_depth=depth
        s=self.s
        if depth not in self._pool:self._pool[depth]=(s.state(),self.state())
        output,next_body=self._pool[depth]
        wp.copy(self.old_q,body.q); wp.copy(self.old_qd,body.qd)
        s._start_mount_device.assign(s.mount_poses)
        tolerance=s.force_tolerance; s.force_tolerance=self.effective_force_tolerance_N
        try:d=self.graph.attempt(soft,output,control,dt,np.zeros(3),np.zeros(3))
        finally:s.force_tolerance=tolerance
        for field,source in (('q',self.next_q),('qd',self.next_qd),('poses',self.next_poses),('velocities',self.next_velocities)):
            wp.copy(getattr(next_body,field),source)
        wp.copy(output.particle_f,soft.particle_f)
        wp.copy(s.contact_forces,s._trial_contact_forces); wp.copy(s.self_contact_forces,s._trial_self_contact_forces)
        wp.copy(s._certified_positions,output.particle_q); s._has_certified_positions=True
        s.mount_poses[s.soft_mount_count+self.ids]=next_body.poses.numpy()
        s._mount_device.assign(s.mount_poses)
        shape=self.shape_stats.numpy()
        residual=s._attempt_work[-1]
        self._accepted.append(dict(dt_s=dt,joint_residual_N=d.max_residual_N,
            converged=d.converged,termination_reason=residual['termination_reason'],
            accepted_without_convergence=residual['accepted_without_convergence'],
            fem_residual_N=residual['fem_residual_N'],affine_residual_N=residual['affine_residual_N'],
            com_residual_N=residual['com_residual_N'],shape_residual_N=residual['shape_residual_N'],
            max_orthogonality_error=float(shape[0]),min_affine_determinant=float(shape[1]),
            temporal_defect_m=s._attempt_work[-1].get('temporal_defect_m'),
            final_path_policy=self.final_path_policy,
            physical_segment_certified=s._attempt_work[-1].get('physical_segment_certified')))
        if self.temporal_tolerance_m and self.temporal_predictive:
            defect=s._attempt_work[-1]['temporal_defect_m']
            self._temporal_hint_dt=dt*(2 if defect<=.25*self.temporal_tolerance_m else 1)
        self._successes_after_failure+=1
        if self.intra_step_hint:
            work=s._attempt_work[-1]
            easy=(work['newton_iterations']<=max(1,s.iterations//2) and
                  work['friction_iterations']<=max(1,s.friction_iterations//2))
            if self.temporal_tolerance_m:
                easy=easy and work['temporal_defect_m']<=.25*self.temporal_tolerance_m
            self._intra_hint_dt=dt*(2 if easy else 1)
        return output,next_body,d

    def _advance(self,soft,body,dt,control,depth):
        mark=len(self._accepted)
        # Only a previously ACCEPTED interval in this outer step supplies the
        # hint. Do not skip the finest permitted solve or change its checks.
        skip=(self.intra_step_hint and self._intra_hint_dt is not None and
              dt>self._intra_hint_dt*(1+1e-12) and depth<self.max_depth)
        if skip and depth>=self._initial_depth:self._intra_skips+=1
        # A converged small interval does not certify that a recently rejected
        # larger interval is now safe. Retain the smallest failed size for this
        # OUTER step only. This changes scheduling, never acceptance or budgets.
        failure_skip=(self.failure_aware_growth and self._failed_dt_cap is not None
                      and dt>self._failed_dt_cap*(1+1e-12) and depth<self.max_depth)
        if failure_skip and not skip and depth>=self._initial_depth:self._failure_growth_skips+=1
        skip=skip or failure_skip
        temporal_skip=(self.temporal_tolerance_m and self.temporal_predictive and
            self._temporal_hint_dt is not None and dt>self._temporal_hint_dt*(1+1e-12) and depth<self.max_depth)
        if temporal_skip and not skip:self._temporal_skips+=1
        skip=skip or temporal_skip
        if depth>=self._initial_depth and not skip:
            attempt_mark=len(self.s._attempt_work)
            try:return self._attempt(soft,body,dt,control,depth)
            except IPCStepRejected as error:
                self._failures.append(dict(depth=depth,dt_s=dt,reason=str(error)))
                if not error.retryable or depth>=self.max_depth:raise
                # Host-side rejection can precede any graph report. Never read
                # an empty list or classify it using a previous trial's code.
                temporal=(len(self.s._attempt_work)>attempt_mark and self.s._attempt_work[-1]['code']==15)
                if temporal and self.temporal_predictive:
                    defect=self.s._attempt_work[-1]['temporal_defect_m']
                    # Quadratic local scaling is only a scheduling proposal.
                    # Every attempted interval still checks its actual defect.
                    factor=max(.125,min(.5,.8*np.sqrt(self.temporal_tolerance_m/defect)))
                    self._temporal_hint_dt=dt*factor
                if self.failure_aware_growth and not temporal:
                    self._failed_dt_cap=min(dt*.5,self._failed_dt_cap or dt*.5)
                    self._successes_after_failure=0
                self._accepted=self._accepted[:mark]
        a,b,first=self._advance(soft,body,dt*.5,control,depth+1)
        # One distinct midpoint per recursion level, never an ancestor/input
        # buffer. Reuse is exact and also avoids invalidating graph input keys.
        if self.reuse_refinement_buffers:
            if depth not in self._midpoint_pool:self._midpoint_pool[depth]=(self.s.state(),self.state())
            middle,middle_body=self._midpoint_pool[depth]
        else:middle,middle_body=self.s.state(),self.state()
        for name in ('particle_q','particle_qd','particle_f','body_q','body_qd'):
            wp.copy(getattr(middle,name),getattr(a,name))
        for name in ('q','qd','poses','velocities'):wp.copy(getattr(middle_body,name),getattr(b,name))
        out,body_out,second=self._advance(middle,middle_body,dt*.5,control,depth+1)
        return out,body_out,combine(first,second)

    def _validate_affine_states(self,soft_in,soft_out,rigid_in,rigid_out):
        buffers=[]
        for state in (soft_in,soft_out):
            for field in ('particle_q','particle_qd','particle_f','body_q','body_qd','body_f'):
                a=getattr(state,field,None)
                if isinstance(a,wp.array) and a.size:
                    buffers.append(('soft.'+field,a.ptr,a.ptr+a.size*a.strides[0]))
        for body in (rigid_in,rigid_out):
            for field,shape,dtype in (('q',(4*self.nb,),wp.vec3d),('qd',(4*self.nb,),wp.vec3d),
                    ('poses',(self.nb,),wp.mat44d),('velocities',(self.nb,6),wp.float64)):
                a=getattr(body,field,None)
                if not isinstance(a,wp.array) or a.shape!=shape or a.dtype!=dtype or a.device!=self.device or not a.is_contiguous:
                    raise ValueError('ABD state arrays must match solver layout, device and contiguous storage')
                begin,end=a.ptr,a.ptr+a.shape[0]*a.strides[0]
                key='abd.'+field
                for other,lo,hi in buffers:
                    if begin<hi and lo<end and (key,begin,end)!=(other,lo,hi):
                        raise ValueError('ABD state buffers overlap across fields or domains')
                buffers.append((key,begin,end))

    def step(self,soft_in,soft_out,rigid_in,rigid_out,dt,*,mount_target=None,external_wrenches=None,control=None):
        start=perf_counter(); s=self.s; checkpoint=_Checkpoint(s)
        self._accepted=[]; self._failures=[]; s._attempt_work=[]; s._trace=[]
        self._initial_depth=0
        self._temporal_hint_dt=None;self._temporal_skips=0
        self._intra_hint_dt=None;self._intra_skips=0
        self._failed_dt_cap=None;self._failure_growth_skips=0
        self._successes_after_failure=0
        accepted=False; error=None; effective_tolerance=None
        try:
            if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite timestep required')
            if self.stiffness_pa!=self._built_stiffness:raise ValueError('rebuild ABD solver to change stiffness')
            if self.shape_hessian not in ('jacobi','spectral','spectral_rows'):raise ValueError('ABD shape Hessian must be jacobi, spectral or spectral_rows')
            if self.body_inverse_mode not in ('scalar','tiled'):raise ValueError('ABD body inverse must be scalar or tiled')
            if self.operator_mode not in ('matrix_free','assembled','fused'):raise ValueError('ABD operator must be matrix_free, assembled or fused')
            if self.skip_empty_contact_matvec and self.operator_mode!='assembled':
                raise ValueError('empty contact matvec optimization requires assembled operator')
            if isinstance(self.cg_group_size,bool) or not isinstance(self.cg_group_size,int) or self.cg_group_size not in (1,2,4,8):
                raise ValueError('CG group size must be 1, 2, 4 or 8')
            if self.preconditioner_mode not in ('block','schwarz'):raise ValueError('ABD preconditioner must be block or schwarz')
            if self.preconditioner_mode=='schwarz' and self.operator_mode!='assembled':
                raise ValueError('joint Schwarz requires assembled operator')
            self._validate_remaining_options()
            if isinstance(self.max_backtracks_per_attempt,bool) or not isinstance(self.max_backtracks_per_attempt,int) or self.max_backtracks_per_attempt<0:
                raise ValueError('backtrack budget must be a nonnegative integer')
            if self.fast_guided_blocks and not (self.all_guided and self.reduced_guides):
                raise ValueError('fast guided blocks require all bodies guided and reduced_guides')
            if self.contact_projection not in ('full','low_rank'):raise ValueError('ABD contact projection must be full or low_rank')
            if not np.isfinite(self.friction_forcing) or not 0<=self.friction_forcing<1:
                raise ValueError('friction forcing must be finite and in [0,1)')
            if self.friction_forcing and not self.inexact_friction:
                raise ValueError('relative friction forcing requires inexact_friction')
            if not np.isfinite(self.max_shape_error) or not 0<self.max_shape_error<1:
                raise ValueError('ABD shape error bound must be in (0,1)')
            if isinstance(self.max_depth,bool) or not isinstance(self.max_depth,int) or not 0<=self.max_depth<=12:
                raise ValueError('subdivision depth must be an integer in [0,12]')
            if np.shape(self.gravity)!=(3,) or not np.isfinite(self.gravity).all():raise ValueError('finite gravity vector required')
            s._validate_runtime_parameters(); s._validate_states(soft_in,soft_out)
            effective_tolerance=self.effective_force_tolerance_N
            if s.curved_path:raise ValueError('ABD uses affine CCD paths; disable prescribed curved_path mode')
            self._validate_friction_policy()
            self._validate_convergence()
            from .abd_path_policy import validate_final_path_policy
            validate_final_path_policy(self.final_path_policy)
            if s.friction_iterations<2:raise ValueError('ABD requires a friction report budget of at least two')
            if s.energy.contact_hessian!='free' or s.energy.multilevel is not None:
                raise ValueError('ABD requires its coupled free/jacobi configuration')
            self._validate_affine_states(soft_in,soft_out,rigid_in,rigid_out)
            if mount_target is not None and not np.array_equal(np.asarray(mount_target),s.mount_poses):
                raise ValueError('ABD currently requires stationary backings and non-dynamic mounts; free objects cannot receive pose targets')
            wrench=np.zeros((self.nb,6)) if external_wrenches is None else np.asarray(external_wrenches,float)
            if wrench.shape!=(self.nb,6) or not np.isfinite(wrench).all():raise ValueError('finite world force per ABD required')
            if np.any(wrench[:,3:]!=0.):raise ValueError('ABD external torque commands are not implemented; apply COM forces')
            self.external.assign(wrench[:,:3].copy())
            if self.adaptive_substep_hint and self._hint_dt is not None:
                # Only scheduling uses the already-read previous work report.
                # Every actual substep still solves/checks the original GPU
                # physical system. This deliberately may change the time grid.
                self._initial_depth=min(self.max_depth,max(0,int(np.ceil(np.log2(dt/self._hint_dt)))))
            out,body,d=self._advance(soft_in,rigid_in,float(dt),control,0)
            for name in ('particle_q','particle_qd','body_q','body_qd'):wp.copy(getattr(soft_out,name),getattr(out,name))
            for name in ('q','qd','poses','velocities'):wp.copy(getattr(rigid_out,name),getattr(body,name))
            s.last_diagnostics=d; s.last_unified_trace=tuple(s._trace)
            if self.adaptive_substep_hint:
                # The terminal accepted interval is closest to the next
                # physical state. Reusing the smallest interval anywhere in
                # the previous step can lock a later easy state into too many
                # subdivisions after a brief contact transition.
                terminal=self._accepted[-1]['dt_s']
                last=s._attempt_work[-1]
                # Failure memory expires at the outer boundary. A successful
                # tail may cautiously grow again; otherwise the smaller final
                # interval would cause several subsequent steps to over-refine.
                # Reaching the nonlinear cap is evidence that the larger
                # interval was expensive, not just a transient geometry miss.
                # Two successful halves must not immediately schedule that
                # failed size again next frame. A failure-free outer step can
                # resume the existing growth rule; no acceptance check changes.
                exhausted=any(w.get('code')==7 for w in s._attempt_work)
                recovered=(self.failure_aware_growth and not exhausted and self._successes_after_failure>=2
                           and last['friction_iterations']<=max(1,s.friction_iterations//2))
                easy=last['converged'] and (not self._failures or recovered) and last['newton_iterations']<=max(1,s.iterations//2)
                self._hint_dt=min(float(dt),terminal*(2 if easy else 1))
            accepted=True; return d
        except BaseException as exc:
            error=str(exc) or type(exc).__name__; checkpoint.restore(s); raise
        finally:
            work=s._attempt_work
            self.last_step_work=dict(accepted=accepted,error=error,wall_ms=1000*(perf_counter()-start),
                converged=accepted and all(v['converged'] for v in self._accepted),
                newton_limit_policy=self.newton_limit_policy,newton_iteration_limit=s.iterations,
                accepted_without_convergence=accepted and any(v['accepted_without_convergence'] for v in self._accepted),
                budget_accepted_intervals=sum(v['accepted_without_convergence'] for v in self._accepted) if accepted else 0,
                joint_solves=len(work),fem_evaluations=0,published_substeps=self._accepted if accepted else [],
                failed_attempts=self._failures,attempts=list(work),
                newton_iterations=sum(w['newton_iterations'] for w in work),
                newton_iteration_budget_scope='per_internal_solve',
                max_internal_newton_iterations=max((w['newton_iterations'] for w in work),default=0),
                rejected_newton_iterations=sum(w['newton_iterations'] for w in work if w.get('code')!=0),
                pcg_iterations=sum(w['pcg_iterations'] for w in work),
                capture_ms=sum(w['capture_ms'] for w in work),
                operator_mode=self.operator_mode,
                skip_empty_contact_matvec=self.skip_empty_contact_matvec,
                skip_fixed_hessians=self.skip_fixed_hessians,cg_group_size=self.cg_group_size,
                material_projection=self.material_projection,coupled_projection=self.coupled_projection,
                rotation_line_search=self.rotation_line_search,
                preconditioner_mode=self.preconditioner_mode,
                reuse_refinement_buffers=self.reuse_refinement_buffers,
                adaptive_substep_hint=self.adaptive_substep_hint,initial_subdivision_depth=self._initial_depth,
                intra_step_hint=self.intra_step_hint,intra_step_scheduled_skips=self._intra_skips,
                failure_aware_growth=self.failure_aware_growth,failure_growth_skips=self._failure_growth_skips,
                compact_linear_dofs=self.compact_linear_dofs,
                cooperative_matvec=self.cooperative_matvec,
                fused_pcg=self.fused_pcg,trial_ccd_prepass=self.trial_ccd_prepass,
                trial_ccd_cancel=self.trial_ccd_cancel,
                fused_preconditioner=self.fused_preconditioner,
                schwarz_symmetry_cache=self.schwarz_symmetry_cache,ccd_candidate_skin=self.ccd_candidate_skin,
                schwarz_cooperative_restriction=self.schwarz_cooperative_restriction,
                max_backtracks_per_attempt=self.max_backtracks_per_attempt,
                ccd_safe_step=self.ccd_safe_step,split_contact_hessian=self.split_contact_hessian,merge_contact_hessians=self.merge_contact_hessians,
                cooperative_contact_projection=self.cooperative_contact_projection,
                stable_energy=self.stable_energy,
                contact_projection=self.contact_projection,
                friction_policy=self.friction_policy,friction_passes=self.friction_passes,
                newton_convergence=self.newton_convergence,
                final_path_policy=self.final_path_policy,
                accepted_uncertified_physical_segments=sum(
                    w.get('code')==0 and w.get('physical_segment_certified') is False for w in work),
                velocity_tolerance_m_s=self.velocity_tolerance_m_s if self.newton_convergence=='velocity' else None,
                velocity_stop_at_current=self.velocity_stop_at_current,
                velocity_stop_factor=self.velocity_stop_factor,
                transform_rate_tolerance_s=self.transform_rate_tolerance_s if self.newton_convergence=='velocity' else None,
                force_tolerance_N=effective_tolerance if self.newton_convergence=='force' else None,
                force_tolerance_mode=('not_used' if self.newton_convergence=='velocity' else
                    'legacy' if self.force_tolerance_N is None else 'explicit'),
                temporal_tolerance_m=self.temporal_tolerance_m,
                temporal_max_depth=self.max_depth,graph_cache_capacity=self.graph.cache_capacity,
                temporal_predictive=self.temporal_predictive,temporal_scheduled_skips=self._temporal_skips,
                temporal_rejections=sum(w['code']==15 for w in work),
                friction_consistency_required=self.friction_policy!='lagged',
                inexact_friction=self.inexact_friction and (self.friction_policy=='strict' or
                    (self.friction_policy=='lagged' and self.friction_passes>1)),
                friction_forcing=self.friction_forcing,fast_guided_blocks=self.fast_guided_blocks,
                rowwise_blocks=self.rowwise_blocks,
                reuse_contact_work=self.reuse_contact_work,body_inverse=self.body_inverse_mode,
                method='joint GPU ABD/FEM/IPC; affine backward Euler; stationary mounts; COM forces')
