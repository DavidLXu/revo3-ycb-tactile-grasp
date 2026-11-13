"""One CUDA graph solves FEM and affine unknowns together, including friction."""
import warp as wp
from dataclasses import replace
from warp.optim.linear import LinearOperator
from .ipc_gpu_graph import IPCDeviceGraph, ERRORS
from .ipc_gpu_nested_cg import NestedCG
from .ipc_gpu_preconditioner import diagonal_inverse
from . import ipc_gpu_control as ctl, ipc_gpu_kernels as k
from . import ipc_gpu_hessian as hk, ipc_gpu_contact_hessian as chk
from . import ipc_gpu_operator as op, ipc_gpu_friction_iteration as fi
from . import abd_gpu as ak
from . import abd_operator as ao
from . import abd_energy_increment as ai, ipc_energy_increment as ei
from . import abd_fused_operator as af
from . import ipc_contact_spectral as cs
from . import abd_friction_iteration as afi
from . import abd_block_rows as abr
from . import abd_mapping_precision as amp
from . import abd_guides as ag
from . import abd_convergence as ac
from . import abd_fixed_hessian as fh
from . import abd_path_policy as ap
from . import abd_temporal as at
from . import abd_material_projection as mp
from . import abd_rotation_search as ar
from . import abd_trial_paths as atp
from .abd_body_blocks import BodyBlocks

ERRORS[13]='ABD shape distortion exceeds configured bound'
ERRORS[14]='ABD rejected-trial budget exhausted; retry at smaller timestep'
ERRORS[15]='ABD temporal defect exceeds position tolerance; retry at smaller timestep'
ERRORS[32]='ABD coordinates do not match the input collision geometry'
ERRORS[57]='ABD surface tangent assembly has an invalid frame or missing matrix entry'


class ABDDeviceGraph(IPCDeviceGraph):
    stable_energy_differences=False
    short_circuit_trial_paths=True
    parallel_raw_body=True
    parallel_raw_material=False
    projected_material_block_dim=256
    reuse_retry_friction=True
    report_columns=48

    def launch(self,kernel,n,args):
        if kernel==cs.normals:
            # This branch can still execute a full 12x12 mollified-EE
            # eigensolve. Match the small blocks used for the original
            # contact kernel instead of inheriting the generic 256 threads.
            if n:wp.launch(kernel,n,args,device=self.device,block_dim=32)
        elif (self.projected_material_block_dim==32 and self.e.material_model=='log_neo_hookean'
                and self.e.material_hessian=='spectral'
                and kernel in (self.e.material_hessian_kernel,self.e.material_free_hessian_kernel)):
            # The spectral tensor kernel has substantial per-thread register
            # pressure. One warp per block distributes the per-tet work more
            # widely; arithmetic, clamp, damping and fixed filtering stay intact.
            from .ipc_precision import position_kernel
            if n:wp.launch(position_kernel(kernel,args[0]),n,args,device=self.device,block_dim=32)
        else:super().launch(kernel,n,args)

    def __init__(self,coupled):
        super().__init__(coupled.s)
        self.c=coupled; self.joint_cg=None; self.joint_signature=None
        self.assembled_cg=None
        self.rotation_contacts=wp.zeros(coupled.nb,dtype=int,device=self.device)
        self.rotation_eligible=wp.zeros(coupled.nb,dtype=int,device=self.device)
        self.rotation_linear_ccd=wp.zeros(1,dtype=int,device=self.device)
        self.rotation_changed=wp.zeros(1,dtype=int,device=self.device)
        self.rotation_counts=wp.zeros(2,dtype=int,device=self.device)
        self.surface_path_needed=wp.zeros(1,dtype=int,device=self.device)
        self.no_remaining=wp.zeros(1,dtype=int,device=self.device)
        self.assembled_matrix=None; self.assembled_base_signature=None
        self.schwarz=None;self.schwarz_matrix=None
        self.compact=None
        self.base_contacts=None
        self.ccd_step=None
        self.split_contact=None
        self.inner_tolerance=wp.zeros(1,dtype=wp.float64,device=self.device)
        self.friction_newton_mark=wp.zeros(1,dtype=int,device=self.device)
        self.correction_metrics=wp.zeros(6,dtype=wp.float64,device=self.device)
        self.correction_done=wp.zeros(1,dtype=int,device=self.device)
        self.stationary_cond=wp.zeros(1,dtype=int,device=self.device)
        self.advance_cond=wp.zeros(1,dtype=int,device=self.device)
        self.residual_guard_state=wp.zeros(2,dtype=wp.float64,device=self.device)
        self.seed_contact_changed=wp.zeros(1,dtype=int,device=self.device)
        self.material_projected=wp.zeros(1,dtype=int,device=self.device)
        self.material_retry=wp.zeros(1,dtype=int,device=self.device)
        self.material_projection_counts=wp.zeros(3,dtype=int,device=self.device)
        self.body_blocks=BodyBlocks(coupled.nb,self.device)
        import numpy as np
        coupled.aggregation_index=wp.array(np.arange(coupled.nd,dtype=np.int32),dtype=int,device=self.device)
        self.joint_operator=LinearOperator((3*coupled.nd,3*coupled.nd),wp.float64,self.device,self.matvec)
        self.assembled_operator=LinearOperator((3*coupled.nd,3*coupled.nd),wp.float64,self.device,self.assembled_matvec)
        self.joint_preconditioner=LinearOperator((3*coupled.nd,3*coupled.nd),wp.float64,self.device,self.precondition)

    def trial_surface_status(self):
        surface=getattr(self.e,'surface_contact',None)
        if surface is not None:
            from .ipc_surface_control import reject_trial
            self.launch(reject_trial,1,[self.status,self.e.stats,surface.trial_rejections])

    def seed_value(self,x,old,dt,c0,c1):
        # freeze(old) has just built the current normal field. The ABD seed
        # usually reconstructs exactly old, but rigid remapping can differ:
        # reuse ONLY on exactly equal coordinates, never a distance tolerance.
        if getattr(self.e,'surface_contact',None) is None:
            self.value(x,old,dt,c0,c1)
            return
        self.seed_contact_changed.zero_()
        self.launch(self.s._positions_changed,self.c.n,[old,x,self.seed_contact_changed])
        wp.capture_if(self.seed_contact_changed,
            lambda:self.value(x,old,dt,c0,c1),
            lambda:self._value_from_current_contacts(x,old,dt,c0,c1))

    def prepare(self,state):
        super().prepare(state)
        c,e=self.c,self.e
        if self.projected_material_block_dim not in (32,256):
            raise ValueError('projected material block dimension must be 32 or 256')
        if c.material_projection=='on_demand' and e.material_model!='log_neo_hookean':
            raise ValueError('on-demand material projection requires log Neo-Hookean')
        if c.coupled_projection:
            if c.material_projection!='on_demand':
                raise ValueError('coupled projection requires on_demand material projection')
            if (not c.split_contact_hessian or c.contact_projection!='low_rank'
                    or c.merge_contact_hessians or e.mesh.aggregation is not None
                    or getattr(e,'surface_contact',None) is not None):
                raise ValueError('coupled projection requires split low_rank contact without merging, aggregation or surface integration')
        if getattr(e,"surface_contact",None) is not None:
            if c.operator_mode!='assembled' or c.stable_energy:
                raise ValueError('surface contact requires assembled operator and stable_energy=False; unsupported modes must not omit its energy or tangent')
        if e.mesh.aggregation is not None and c.operator_mode!='assembled':
            raise ValueError('aggregated contact requires the assembled ABD operator')
        if c.cooperative_contact_projection and not c.split_contact_hessian:
            raise ValueError('cooperative projection requires split contact Hessians')
        if c.merge_contact_hessians and (not c.split_contact_hessian or e.mesh.aggregation is not None):
            raise ValueError('feature merging requires split Hessians without nonlinear contact aggregation')
        if c.split_contact_hessian:
            if c.contact_projection!='low_rank':raise ValueError('split contact Hessians require low_rank projection')
            if (self.split_contact is None or self.split_contact.capacity!=e.mesh.batch.capacity
                    or self.split_contact.cooperative!=c.cooperative_contact_projection
                    or (self.split_contact.features is not None)!=c.merge_contact_hessians):
                from .ipc_gpu_split_contact import SplitContactHessian
                self.split_contact=SplitContactHessian(e.mesh.batch.capacity,self.device,
                    cooperative=c.cooperative_contact_projection,merge_features=c.merge_contact_hessians,
                    node_count=self.s.model.particle_count)
        else:self.split_contact=None
        if c.ccd_safe_step:
            if not self.s.self_motion.parallel_pairs:
                raise ValueError('CCD safe steps require parallel CCD')
            if (self.ccd_step is None or self.ccd_step.audit is not self.s.self_motion
                    or self.ccd_step.min_gap != self.s.self_motion.min_gap
                    or self.ccd_step.candidate_skin != c.ccd_candidate_skin
                    or self.ccd_step.reject_prepass != c.trial_ccd_prepass
                    or self.ccd_step.cancel_rejected != c.trial_ccd_cancel
                    or (self.ccd_step.candidate_cache is not None and
                        self.ccd_step.candidate_cache.capacity != self.s.self_motion.pair_capacity)):
                from .ipc_gpu_safe_step import GPUCollisionSafeStep
                self.ccd_step=GPUCollisionSafeStep(self.s.self_motion,candidate_skin=c.ccd_candidate_skin,
                    reject_prepass=c.trial_ccd_prepass,cancel_rejected=c.trial_ccd_cancel)
        else:
            self.ccd_step=None
        if c.stable_energy and (self.base_contacts is None or self.base_contacts.capacity!=e.mesh.batch.capacity):
            self.base_contacts=ai.BaseContacts(e.mesh.batch,self.device)
        if c.operator_mode=='assembled':
            surface=getattr(e,'surface_contact',None)
            faces=None if surface is None else surface.faces
            base_signature=(id(e.matrix),e.matrix.offsets.ptr,e.matrix.columns.ptr,
                            id(surface),None if faces is None else (id(faces),faces.ptr))
            if self.assembled_matrix is None or base_signature!=self.assembled_base_signature:
                self.assembled_matrix=ao.JointMatrix(c.n,c.nb,e.matrix,self.device,surface_faces=faces)
                self.assembled_base_signature=base_signature
            self.assembled_matrix.prepare((e.mesh.batch,e.mesh_frame))
            self.assembled_matrix.cooperative_product=c.cooperative_matvec
        if c.preconditioner_mode=='schwarz':
            if (self.schwarz is None or self.schwarz_matrix is not self.assembled_matrix
                    or self.schwarz.cache_symmetry!=c.schwarz_symmetry_cache
                    or self.schwarz.cooperative_restriction!=c.schwarz_cooperative_restriction):
                from .abd_schwarz import JointSchwarz
                self.schwarz=JointSchwarz(c,self.assembled_matrix,cache_symmetry=c.schwarz_symmetry_cache,
                    cooperative_restriction=c.schwarz_cooperative_restriction)
                self.schwarz_matrix=self.assembled_matrix
        if c.compact_linear_dofs:
            if (self.compact is None or self.compact.schwarz is not self.schwarz
                    or self.compact.matrix is not self.assembled_matrix
                    or self.compact.batches!=(e.mesh.batch,e.mesh_frame)):
                from .abd_compact import CompactJointSystem
                self.compact=CompactJointSystem(c,self.assembled_matrix,self.schwarz,(e.mesh.batch,e.mesh_frame))
        else:self.compact=None
        # Velocity termination still needs a direction for every nonzero
        # gradient. A fixed 1e-10 N PCG floor can return zero iterations and
        # a zero direction near equilibrium, which correctly fails the strict
        # descent check and used to trigger futile timestep subdivision.
        # Solve that small system to the same relative tolerance instead;
        # do not relax descent, collision, or nonlinear stopping checks.
        linear_atol=0. if c.newton_convergence=='velocity' else 1.e-10
        signature=(c.material_projection,e.linear_tolerance,linear_atol,e.linear_max_iterations,c.operator_mode,c.skip_empty_contact_matvec,c.cg_group_size,c.preconditioner_mode,id(self.assembled_matrix),id(self.schwarz),id(self.compact),c.cooperative_matvec,id(e.mesh.aggregation),c.fused_pcg,c.fused_preconditioner)
        if self.joint_cg is None or signature!=self.joint_signature:
            a,assembled,rhs,direction,m=(self.joint_operator,self.assembled_operator,c.rhs,c.direction,self.joint_preconditioner)
            if self.compact is not None:
                v=self.compact
                a,assembled,rhs,direction,m=v.operator,v.assembled_operator,v.rhs,v.direction,v.preconditioner
            self.joint_cg=NestedCG(a,rhs,direction,M=m,
                tol=e.linear_tolerance,atol=linear_atol,maxiter=e.linear_max_iterations,
                check_every=0,use_cuda_graph=True)
            self.joint_cg.zero_initial_guess=True; self.joint_signature=signature
            self.joint_cg.group_size=c.cg_group_size
            self.joint_cg.check_positive_curvature=c.material_projection=='on_demand'
            self.assembled_cg=None
            if c.skip_empty_contact_matvec:
                self.assembled_cg=NestedCG(assembled,rhs,direction,M=m,
                    tol=e.linear_tolerance,atol=linear_atol,maxiter=e.linear_max_iterations,
                    check_every=0,use_cuda_graph=True)
                self.assembled_cg.zero_initial_guess=True
                self.assembled_cg.group_size=c.cg_group_size
                self.assembled_cg.check_positive_curvature=c.material_projection=='on_demand'
                # Only the device-selected EMPTY-tail branch can use this
                # product. Aggregated contacts add a separate matrix action.
                # Both cases retain the original complete-operator CG path.
                if c.fused_pcg and e.mesh.aggregation is None:
                    from .abd_warp_product import FusedProductDot
                    self.assembled_cg.fused_product_dot=FusedProductDot(self.compact)
            if c.fused_preconditioner:
                # This only changes M*r and its dot products; it is valid for
                # both complete-operator and empty-tail solves.
                from .abd_precondition_dot import FusedPreconditionDot
                callback=FusedPreconditionDot(self.compact)
                self.joint_cg.fused_precondition_dot=callback
                if self.assembled_cg is not None:self.assembled_cg.fused_precondition_dot=callback

    def _cache_key(self,*args):
        key,retained=super()._cache_key(*args)
        key=(key,self.c.temporal_tolerance_m,self.c.trial_ccd_prepass,self.c.trial_ccd_cancel,self.parallel_raw_material,self.projected_material_block_dim)
        arrays=[v for v in vars(self.c).values() if isinstance(v,wp.array)]
        arrays.extend([self.material_projected,self.material_retry,self.material_projection_counts,self.no_remaining,self.inner_tolerance,self.friction_newton_mark,
            self.correction_metrics,self.correction_done,self.stationary_cond,self.advance_cond,self.seed_contact_changed,self.residual_guard_state])
        if self.schwarz is not None:
            arrays += [v for v in vars(self.schwarz).values() if isinstance(v,wp.array)]
            arrays += list(self.schwarz.restriction_plan)+list(self.schwarz.assembly_plan)
        if self.compact is not None:
            arrays += [v for v in vars(self.compact).values() if isinstance(v,wp.array)]
            arrays += list(self.compact.restriction_plan)
        fused=getattr(self.assembled_cg,'fused_product_dot',None)
        if fused is not None:
            arrays += [fused.terms]+[level[2] for level in fused.levels]
        if self.split_contact is not None:
            arrays += [self.split_contact.pending,self.split_contact.count]
            if self.split_contact.features is not None:
                arrays += [v for v in vars(self.split_contact.features).values() if isinstance(v,wp.array)]
        arrays+=[v for v in vars(self.body_blocks).values() if isinstance(v,wp.array)]
        if self.base_contacts is not None:
            arrays+=[v for v in vars(self.base_contacts).values() if isinstance(v,wp.array)]
        if self.ccd_step is not None:
            arrays+=[v for v in vars(self.ccd_step).values() if isinstance(v,wp.array)]
            if self.ccd_step.candidate_cache is not None:
                arrays+=[v for v in vars(self.ccd_step.candidate_cache).values() if isinstance(v,wp.array)]
        if self.assembled_matrix is not None:
            arrays+=[v for v in vars(self.assembled_matrix).values() if isinstance(v,wp.array)]
            arrays+=[a for scratch in self.assembled_matrix.remaining.values() for a in scratch]
            arrays+=self.assembled_matrix.block_scan.prefixes+self.assembled_matrix.block_scan.totals
        return (key,self.reuse_retry_friction,self.parallel_raw_body,self.short_circuit_trial_paths,self.c.rotation_line_search,self.c.newton_limit_policy,self.c.material_projection,self.c.coupled_projection,id(getattr(self.seed_value,'__func__',self.seed_value)),self.c.final_path_policy,self.c.newton_convergence,self.c.velocity_tolerance_m_s,self.c.transform_rate_tolerance_s,self.c.velocity_residual_guard,self.c.velocity_stop_at_current,self.c.velocity_stop_factor,self.c.friction_policy,self.c.friction_passes,self.c.max_shape_error,tuple(self.c.gravity),self.c.shape_hessian,self.c.operator_mode,self.c.stable_energy,self.c.contact_projection,self.c.inexact_friction,self.c.friction_forcing,self.c.rowwise_blocks,self.c.reuse_contact_work,self.c.body_inverse_mode,self.c.has_guides,self.c.reduced_guides,self.c.fast_guided_blocks,self.c.skip_empty_contact_matvec,id(self.joint_cg),id(self.assembled_cg),
                self.c.skip_fixed_hessians,self.c.cg_group_size,self.c.preconditioner_mode,id(self.schwarz),id(self.compact),self.c.cooperative_matvec,self.c.effective_backtrack_budget,self.c.ccd_safe_step,id(self.ccd_step),self.c.split_contact_hessian,id(self.split_contact),self.c.cooperative_contact_projection,tuple((id(a),a.ptr) for a in arrays)),retained+arrays+[self.joint_cg,self.assembled_cg,self.assembled_matrix,self.schwarz,self.compact,self.ccd_step,self.split_contact]

    def attempt(self,*args,**kwargs):
        work_start=len(self.s._attempt_work);trace_start=len(self.s._trace)
        try:
            diagnostics=super().attempt(*args,**kwargs)
            return replace(diagnostics,converged=bool(self.s._attempt_work[-1]['converged']))
        finally:
            for row in self.s._attempt_work[work_start:]+self.s._trace[trace_start:]:
                row['force_tolerance_N']=float(self.s.force_tolerance) if self.c.newton_convergence=='force' else None
                row['friction_policy']=self.c.friction_policy
                row['friction_refreshes']=row['friction_iterations']
                row['friction_consistency_checked']=self.c.friction_policy!='lagged' and self.e.mesh.mu>0.
                if self.c.friction_policy=='lagged':
                    row['friction_solve_passes']=(1+row['friction_refreshes'] if row.get('code',0)==0 else None)
                    row['last_pre_solve_refresh_change_N']=(row['friction_update_error_N']
                        if row['friction_refreshes'] else None)
                    row['friction_update_error_N']=None

    def extra_report_fields(self,row):
        velocity=self.c.newton_convergence=='velocity'
        accepted=int(row[0])==0
        budget=velocity and int(row[21])==5
        reason=None
        if int(row[0])==0:
            reason=({1:'velocity_and_transform_rate',2:'zero_gradient',3:'velocity_at_current_iterate',4:'coordinate_resolution'}.get(int(row[21]))
                    if velocity else 'force_residual')
        return dict(newton_convergence=self.c.newton_convergence,convergence_reason=reason,
            newton_limit_policy=self.c.newton_limit_policy,newton_iteration_limit=self.s.iterations,
            converged=accepted and reason is not None,
            iteration_limit_reached=budget or int(row[0])==7,
            accepted_without_convergence=accepted and budget,
            termination_reason=('iteration_limit' if (accepted and budget) or int(row[0])==7 else reason),
            fem_residual_N=float(row[42]) if int(row[0])==0 else None,
            affine_residual_N=float(row[43]) if int(row[0])==0 else None,
            com_residual_N=float(row[44]) if int(row[0])==0 else None,
            shape_residual_N=float(row[45]) if int(row[0])==0 else None,
            material_projection=self.c.material_projection,coupled_projection=self.c.coupled_projection,
            parallel_raw_material=self.parallel_raw_material,
            projected_material_block_dim=self.projected_material_block_dim,
            reuse_retry_friction=self.reuse_retry_friction,
            rotation_line_search=self.c.rotation_line_search,
            rotation_trials=int(row[46]),rotation_accepted_updates=int(row[47]),
            unprojected_material_attempts=int(row[39]),
            material_preconditioner_retries=int(row[40]),material_linear_retries=int(row[41]),
            temporal_defect_m=float(row[38]) if self.c.temporal_tolerance_m else None,
            temporal_tolerance_m=self.c.temporal_tolerance_m,
            velocity_residual_guard=bool(velocity and self.c.velocity_residual_guard),
            velocity_stop_at_current=bool(velocity and self.c.velocity_stop_at_current),
            velocity_stop_factor=self.c.velocity_stop_factor if velocity and self.c.velocity_stop_at_current else None,
            best_residual_N=float(row[36]) if velocity and self.c.velocity_residual_guard else None,
            residual_guard_vetoes=int(row[37]) if velocity and self.c.velocity_residual_guard else 0,
            final_path_policy=self.c.final_path_policy,
            physical_segment_certified=bool(row[24]) if row[23] else None,
            physical_segment_min_J=float(row[25]) if row[23] else None,
            physical_segment_uncertified_pairs=int(row[26]) if row[23] else None,
            physical_segment_invalid_geometry=int(row[27]) if row[23] else None,
            physical_segment_invalid_bounds=int(row[28]) if row[23] else None,
            physical_segment_first_pair=(dict(kind='point_triangle' if int(row[31])==0 else 'edge_edge',
                particle_indices=[int(x) for x in row[32:36]]) if row[23] and row[31]>=0 else None),
            correction_velocity_m_s=float(row[18]) if velocity else None,
            correction_transform_rate_s=float(row[19]) if velocity else None,
            full_correction_path_feasible=bool(row[22]) if velocity else None,
            correction_alpha=float(row[20]) if velocity else None)

    def assembled_matvec(self,x,y,out,alpha,beta):
        # The GPU selector may use this only when BOTH remainder lists are empty.
        # Preserve unreduced guides' P H P + (I-P) wrapper as in the full operator.
        c=self.c
        aggregation=self.e.mesh.aggregation
        if aggregation is not None:aggregation.prepare_product(x,c.mobile_mass,c)
        if c.has_guides and not c.reduced_guides:
            self.launch(ag.project,c.nd,[x,c.guide_axes,c.n,c.guide_input])
            self.assembled_matrix.assembled_matvec(c.guide_input,c.guide_input,c.guide_product,1.,0.)
            self.launch(ag.finish,c.nd,[x,y,c.guide_product,c.guide_axes,c.n,alpha,beta,out])
        else:self.assembled_matrix.assembled_matvec(x,y,out,alpha,beta)
        if aggregation is not None:aggregation.add_product(out,alpha,c.mobile_mass,c)

    def matvec(self,x,y,out,alpha,beta):
        c=self.c
        if getattr(c,'has_guides',False) and not (c.reduced_guides and c.operator_mode=='assembled'):
            self.launch(ag.project,c.nd,[x,c.guide_axes,c.n,c.guide_input])
            self._unconstrained_matvec(c.guide_input,c.guide_input,c.guide_product,1.,0.)
            self.launch(ag.finish,c.nd,[x,y,c.guide_product,c.guide_axes,c.n,alpha,beta,out])
        else:self._unconstrained_matvec(x,y,out,alpha,beta)

    def _unconstrained_matvec(self,x,y,out,alpha,beta):
        c,e,m=self.c,self.e,self.s.model
        if getattr(c,'operator_mode','matrix_free')=='assembled':
            aggregation=e.mesh.aggregation
            if aggregation is not None:aggregation.prepare_product(x,c.mobile_mass,c)
            self.assembled_matrix.matvec(x,y,out,alpha,beta,(e.mesh.batch,e.mesh_frame))
            if aggregation is not None:aggregation.add_product(out,alpha,c.mobile_mass,c)
            return
        if e.mesh.aggregation is not None:
            raise ValueError('aggregated ABD contacts currently require assembled operator mode')
        if getattr(c,'operator_mode','matrix_free')=='fused':
            self.launch(ak.map_positions,c.n,[x,c.owner,c.weights,c.n,c.full_direction])
            mat=e.matrix; target=c.product.view(wp.float64).reshape((c.nd,3))
            self.launch(af.base_product,(c.nd,3),[x,mat.offsets,mat.product_counts,mat.product_columns,
                mat.product_values.view(wp.float64).reshape((len(mat.product_values),3,3)),c.body_h,c.n,target])
            for b in (e.mesh.batch,e.mesh_frame):
                lanes=min(b.capacity,256)
                self.launch(af.contact_product,(lanes,12),[c.full_direction,b.ids,b.active_indices,b.active_count,
                    b.h.view(wp.float64).reshape((b.capacity,12,12)),c.mobile_mass,c.owner,c.weights,c.n,lanes,target])
            self.launch(ak.combine,c.nd,[x,y,c.product,m.particle_mass,c.n,alpha,beta,out])
            return
        self.launch(ak.map_positions,c.n,[x,c.owner,c.weights,c.n,c.full_direction])
        e.matrix.product(c.full_direction,c.full_product,packed=True)
        self.launch(ak.clear_fixed,c.n,[c.full_product,m.particle_mass])
        for b in (e.mesh.batch,e.mesh_frame):
            lanes=min(b.capacity,256)
            self.launch(ak.contact_product,(lanes,12),[c.full_direction,b.ids,b.active_indices,
                b.active_count,b.h.view(wp.float64).reshape((b.capacity,12,12)),c.mobile_mass,
                c.full_product.view(wp.float64).reshape((c.n,3)),lanes])
        c.product.zero_()
        self.launch(ak.restrict_vector,c.n,[c.full_product,c.owner,c.weights,m.particle_mass,c.n,c.product])
        if getattr(c,'rowwise_blocks',False):
            self.launch(abr.body_product,(c.nb,12),[c.body_h.view(wp.float64).reshape((c.nb,12,12)),
                x,c.n,True,c.product.view(wp.float64).reshape((c.nd,3))])
        else:
            self.launch(ak.body_product,(c.nb,4),[c.body_h,x,c.n,c.product])
        self.launch(ak.combine,c.nd,[x,y,c.product,m.particle_mass,c.n,alpha,beta,out])

    def precondition(self,x,y,out,alpha,beta):
        c=self.c
        if getattr(c,'has_guides',False) and not c.reduced_guides:
            self.launch(ag.project,c.nd,[x,c.guide_axes,c.n,c.guide_pre_input])
            self._unconstrained_precondition(c.guide_pre_input,c.guide_pre_input,c.guide_pre_product,1.,0.)
            self.launch(ag.finish,c.nd,[x,y,c.guide_pre_product,c.guide_axes,c.n,alpha,beta,out])
        else:self._unconstrained_precondition(x,y,out,alpha,beta)

    def _unconstrained_precondition(self,x,y,out,alpha,beta):
        c=self.c
        if getattr(c,'preconditioner_mode','block')=='schwarz':
            self.schwarz.apply(x,y,out,alpha,beta)
            return
        if getattr(c,'rowwise_blocks',False):
            self.launch(abr.body_product,(c.nb,12),[c.body_inverse.view(wp.float64).reshape((c.nb,12,12)),
                x,c.n,False,c.preconditioned.view(wp.float64).reshape((c.nd,3))])
            self.launch(abr.finish_precondition,c.nd,[x,y,self.e.inverse,c.preconditioned,
                self.s.model.particle_mass,c.n,alpha,beta,out])
            return
        # Keep the small FEM and dense affine blocks in separate kernels.
        # The combined per-node branch is algebraically equivalent but costs
        # 234 us versus 44 us on the frozen fingertip fixture (RTX 5090).
        # This operation runs at EVERY PCG iteration, not only once/Newton.
        c.preconditioned.zero_()
        self.launch(ak.precondition_soft,c.n,[x,self.e.inverse,c.preconditioned])
        self.launch(ak.body_product,(c.nb,4),[c.body_inverse,x,c.n,c.preconditioned])
        self.launch(ak.combine,c.nd,[x,y,c.preconditioned,self.s.model.particle_mass,c.n,alpha,beta,out])

    def _value_from_current_contacts(self,x,old,dt,c0,c1):
        surface=getattr(self.e,'surface_contact',None)
        super()._value_from_current_contacts(x,old,dt,c0,c1,particle_residual=False,
            surface_tangent=surface is None or not surface.defer_tangent)
        c,e=self.c,self.e
        def finish():
            c.gradient.zero_()
            self.launch(ak.restrict_vector,c.n,[e.gradient,c.owner,c.weights,self.s.model.particle_mass,c.n,c.gradient])
            self.launch(ak.body_energy,c.nb,[c.u,c.predictor,c.radius,c.mass_matrix,c.kv,dt,c.n,c.gradient,e.stats])
            if c.has_guides:self.launch(ag.project,c.nd,[c.gradient,c.guide_axes,c.n,c.gradient])
            self.launch(k.residual,c.nd,[c.gradient,c.dof_mass,e.stats,c.rhs])
        self.alive(finish)

    def hessian(self,x,old,dt,c0,c1,*,project_material=True,reuse_friction=False):
        c,e,m=self.c,self.e,self.s.model
        if not project_material:
            if self.parallel_raw_material:
                from .ipc_log_material_blocks import hessian_blocks
                self.launch(hessian_blocks,(m.tet_count,4,4),[x,e.tets.ids,m.tet_poses,m.tet_materials,
                    dt,e.tets.h.view(wp.float64).reshape((m.tet_count,12,12)),c.mobile_mass,c.skip_fixed_hessians])
            else:
                from .ipc_log_material_raw import hessian_unprojected
                self.launch(hessian_unprojected,m.tet_count,[x,e.tets.ids,m.tet_poses,m.tet_materials,
                    dt,e.tets.h,c.mobile_mass,c.skip_fixed_hessians])
            if e.volume_barrier_enabled:
                self.launch(fh.add_hessian,m.tet_count,[x,e.tets.ids,m.tet_poses,m.tet_materials,
                    e.volume_min_jacobian,e.volume_barrier_activation,e.volume_barrier_scale,e.tets.h,c.mobile_mass])
        elif c.skip_fixed_hessians:
            material=e.material_free_hessian_kernel
            self.launch(material,m.tet_count,[x,e.tets.ids,m.tet_poses,m.tet_materials,dt,e.tets.h,c.mobile_mass])
            if e.volume_barrier_enabled:
                self.launch(fh.add_hessian,m.tet_count,[x,e.tets.ids,m.tet_poses,m.tet_materials,
                    e.volume_min_jacobian,e.volume_barrier_activation,e.volume_barrier_scale,e.tets.h,c.mobile_mass])
        else:
            self.launch(e.material_hessian_kernel,m.tet_count,[x,e.tets.ids,m.tet_poses,m.tet_materials,dt,e.tets.h])
            e.volume_barrier(x,hessian=True)
        b=e.mesh.batch
        # A vertex belonging to a dynamic ABD is free in the joint system,
        # although the original mesh adapter stores zero nodal FEM mass there.
        if self.split_contact is not None:
            if c.coupled_projection and not project_material:
                self.split_contact.build_unprojected(x,b,e.mesh,c.mobile_mass)
            else:self.split_contact.build(x,b,e.mesh,c.mobile_mass)
        elif c.contact_projection=='low_rank':
            self.launch(cs.normals,b.capacity,[x,b.ids,b.kinds,b.areas,b.eps,b.count,
                e.mesh.dhat,e.mesh.dmin,e.mesh.stiffness,b.h,c.mobile_mass])
        else:
            self.launch(chk.normals_free,b.capacity,[x,b.ids,b.kinds,b.areas,b.eps,b.count,c1,
                e.mesh.dhat,e.mesh.dmin,e.mesh.stiffness,0.,b.h,c.mobile_mass])
        if e.mesh.aggregation is not None:e.mesh.aggregation.add_hessian(c.mobile_mass)
        f=e.mesh_frame
        if not reuse_friction:
            self.launch(fh.friction if c.skip_fixed_hessians else hk.friction,f.capacity,
                [x,old,f.ids,f.count,c1,c0,f.weights,f.normals,f.loads,e.epsilon*dt,f.h]+([c.mobile_mass] if c.skip_fixed_hessians else []))
        if c.reuse_contact_work:
            # Dynamic lists below require ABD mobility, so the FEM-only lists
            # would immediately be overwritten. Joint assembly reads base
            # values directly and does not consume its packed matvec rows.
            e._assemble_matrix(compact_dynamic=False,packed_product=c.operator_mode!='assembled')
        else:
            e._assemble_matrix()
        if c.preconditioner_mode=='block':e.matrix.diagonal(e.diagonal)
        if c.fast_guided_blocks:
            self.launch(ag.translation_mass_blocks,(c.nb,12,12),[c.mass,dt,
                c.body_h.view(wp.float64).reshape((c.nb,12,12)),
                c.body_diagonal.view(wp.float64).reshape((c.nb,12,12))])
        elif c.coupled_projection and not project_material:
            if self.parallel_raw_body:
                from .abd_raw_rows import body_hessian_rows
                self.launch(body_hessian_rows,(c.nb,12,12),[c.u,c.radius,c.mass_matrix,c.kv,dt,c.n,
                    c.body_h.view(wp.float64).reshape((c.nb,12,12)),
                    c.body_diagonal.view(wp.float64).reshape((c.nb,12,12))])
            else:
                from .abd_coupled_tangent import raw_body_hessian
                self.launch(raw_body_hessian,c.nb,[c.u,c.radius,c.mass_matrix,c.kv,dt,c.n,c.body_h,c.body_diagonal])
        elif c.shape_hessian=='spectral_rows':
            self.body_blocks.shape(c,dt)
        else:
            body_kernel=ak.body_hessian_spectral if c.shape_hessian=='spectral' else ak.body_hessian
            self.launch(body_kernel,c.nb,[c.u,c.radius,c.mass_matrix,c.kv,dt,c.n,c.body_h,c.body_diagonal])
        for b in (e.mesh.batch,e.mesh_frame):
            b.active_count.zero_()
            self.launch(op.compact_nonzero_free,b.capacity,[b.h,b.ids,c.mobile_mass,b.count,b.active_indices,b.active_count])
            if c.preconditioner_mode=='block':
                self.launch(op.active_diagonal,b.capacity,[b.ids,b.active_indices,b.active_count,b.h,m.particle_mass,e.diagonal])
                lanes=min(b.capacity,128)
                self.launch(ak.contact_body_diagonal,(lanes,12,12),[b.ids,b.active_indices,b.active_count,
                    b.h.view(wp.float64).reshape((b.capacity,12,12)),c.owner,c.weights,
                    c.body_diagonal.view(wp.float64).reshape((c.nb,12,12)),lanes])
        if getattr(e,"surface_contact",None) is not None:
            from .ipc_surface_gpu import add_body_diagonal
            if e.surface_contact.defer_tangent:e.surface_contact.tangent(x,old,e.epsilon*dt)
            e.surface_contact.linearize()
            if c.preconditioner_mode=='block':
                e.surface_contact.add_diagonal(e.diagonal)
                add_body_diagonal(e.surface_contact,c)
        if c.preconditioner_mode=='block':self.launch(diagonal_inverse,c.n,[e.diagonal,e.inverse,e.stats])
        if c.fast_guided_blocks or c.preconditioner_mode=='schwarz':
            # reduced_inverse below writes every entry for every guided body.
            # No full inverse is consumed by the constrained linear system.
            pass
        elif c.body_inverse_mode=='tiled':
            self.body_blocks.inverse(c.body_diagonal,c.body_inverse,e.stats)
        else:
            self.launch(ak.invert_body,c.nb,[c.body_diagonal,c.body_inverse,e.stats])
        if c.has_guides and c.reduced_guides and c.preconditioner_mode=='block':
            # Invert the free axial block, instead of projecting an inverse
            # that includes compliance from the locked affine coordinates.
            self.launch(ag.reduced_inverse,c.nb,[c.body_diagonal.view(wp.float64).reshape((c.nb,12,12)),
                c.guide_axes,c.body_inverse.view(wp.float64).reshape((c.nb,12,12)),e.stats])
        if c.preconditioner_mode=='block':self.launch(ctl.preconditioner,1,[e.stats,self.status])
        if c.operator_mode=='assembled':
            self.assembled_matrix.assemble(e.matrix,c,(e.mesh.batch,e.mesh_frame))
            if getattr(e,'surface_contact',None) is not None:
                self.launch(ctl.flag_error,1,[self.assembled_matrix.surface_error,57,self.status])
            if self.compact is not None:self.compact.refresh()
            if c.preconditioner_mode=='schwarz':
                self.schwarz.build(self.assembled_matrix,(e.mesh.batch,e.mesh_frame),e.stats)
                if self.compact is not None:
                    from .abd_compact import validate_selection
                    self.launch(validate_selection,c.n,[m.particle_mass,self.compact.index,e.stats])
                self.launch(ctl.preconditioner,1,[e.stats,self.status])

    def _solve_joint_cg(self):
        c,e=self.c,self.e
        c.direction.zero_(); e.linear_stats.zero_()
        compact=getattr(self,'compact',None)
        if compact is not None:compact.begin()
        def run(solver):
            result=solver()
            self.launch(op.pack_cg_status,1,[result[0],result[1],e.linear_stats])
            if c.material_projection=='on_demand':
                self.launch(ctl.flag_error,1,[solver.curvature_failure,5,self.status])
        if c.skip_empty_contact_matvec:
            self.launch(ao.no_remaining_contacts,1,[self.assembled_matrix.remaining[e.mesh.batch][2],
                self.assembled_matrix.remaining[e.mesh_frame][2],self.no_remaining])
            # Decide once per assembled system, outside the PCG while loop.
            # No host readback; nonempty normal OR friction tails use full CG.
            wp.capture_if(self.no_remaining,lambda:run(self.assembled_cg),lambda:run(self.joint_cg))
        else:run(self.joint_cg)
        if compact is not None:compact.finish()

    def solve(self):
        c,e=self.c,self.e
        self._solve_joint_cg()
        if c.has_guides:self.launch(ag.project,c.nd,[c.direction,c.guide_axes,c.n,c.direction])
        # Independently check the result using the COMPLETE operator even when
        # the selected inner loop omitted empty kernels.
        self.matvec(c.direction,c.direction,c.applied,1.,0.)
        self.launch(k.check_linear,c.nd,[c.rhs,c.applied,c.direction,e.linear_stats])
        self.launch(ctl.linear,1,[e.linear_stats,e.linear_tolerance,self.status,self.values])

    def linearize_and_solve(self,x,old,dt,c0,c1):
        if self.c.material_projection=='always':
            self.hessian(x,old,dt,c0,c1)
            self.alive(self.solve)
            return
        # No positions, objective or frozen contacts change between these
        # alternatives. A failed raw direction is discarded, never published.
        # coupled_projection also tries raw normal-contact/affine-shape terms;
        # a failure restores ALL projected tangents, with the same checked CG.
        args=(x,old,dt,c0,c1)
        def retry_hessian():
            # Only this raw-to-projected retry may reuse friction. Both
            # assemblies use the identical x/old/dt and frozen frame, and
            # projection changes only elastic, normal and affine tangents.
            # Every new Newton iterate or friction refresh builds it afresh.
            self.hessian(*args,reuse_friction=self.reuse_retry_friction)
        self.launch(mp.begin,1,[self.material_projected,self.material_projection_counts])
        self.hessian(*args,project_material=False)
        def recover(kind):
            self.launch(mp.recover,1,[self.status,self.e.stats,kind,self.material_projected,
                self.material_retry,self.material_projection_counts])
        recover(4)
        wp.capture_if(self.material_retry,retry_hessian)
        self.alive(self.solve)
        recover(5)
        def projected():
            retry_hessian()
            self.alive(self.solve)
        wp.capture_if(self.material_retry,projected)

    def joint_paths(self,u0,u1,x0,x1,c0,c1,*,reuse_self_path=False):
        self.paths(x0,x1,c0,c1)
        self.launch(ak.body_paths,self.c.nb,[u0,u1,self.c.radius,self.c.n,self.s.diagnostic])
        # Only Newton trial paths may use the padded, endpoint-guarded
        # certificate. Seed and final physical paths always run a fresh audit.
        # FEM and affine determinant paths above are never cached here.
        def surface():
            if reuse_self_path and getattr(self,'ccd_step',None) is not None:
                self.ccd_step.check(x0,x1,self.values)
            elif reuse_self_path and self.path_cache is not None:
                self.path_cache.check(x0,x1,self.values)
            else:
                self.s.self_motion.check(x0,x1,readback=False,
                    stop_on_witness=reuse_self_path and self.c.trial_ccd_prepass,
                    stop_after_rejection=reuse_self_path and self.c.trial_ccd_cancel)
        if reuse_self_path and self.short_circuit_trial_paths:
            # A trial failing the analytic gap/J test cannot be accepted. Clear
            # stale motion results, then evaluate CCD only for viable trials.
            # Seed/final audits stay exhaustive; this never certifies a path.
            self.s.self_motion.stats.zero_()
            self.launch(atp.needs_surface_check,1,[self.s.diagnostic,
                self.s.contact.min_gap_m,self.s.min_jacobian,self.surface_path_needed])
            wp.capture_if(self.surface_path_needed,surface)
        else:surface()

    def accept_trial(self,dt):
        s,e,m,c=self.s,self.e,self.s.model,self.c
        if not c.stable_energy:
            return super().accept_trial(dt)
        args=[e.stats,s._line_stats,s.diagnostic,e.mesh.batch.count,self.status,self.values,self.report]
        self.launch(ai.increment_needed,1,[e.stats,self.values,s._line_stats,self.increment_cond])
        def ordinary():self.launch(ctl.accept_trial,1,args)
        def stable():
            self.energy_increment.zero_()
            self.launch(ei.inertia_change,c.n,[s._path_start,s.positions,s.predictor,m.particle_mass,dt,self.energy_increment])
            tet_change=ai.log_tet_change if e.material_model=='log_neo_hookean' else ei.tet_change
            self.launch(tet_change,m.tet_count,[s._path_start,s.positions,s.old,m.tet_indices,m.tet_poses,
                m.tet_materials,dt,e.volume_min_jacobian,e.volume_barrier_activation or 0.,e.volume_barrier_scale,self.energy_increment])
            self.launch(ai.body_change,c.nb,[c.base,c.u,c.predictor,c.radius,c.mass_matrix,c.kv,dt,c.n,self.energy_increment])
            if e.mesh.aggregation is not None:
                e.mesh.aggregation.energy_change(self.energy_increment)
            else:
                for batch,departed in ((e.mesh.batch,False),(self.base_contacts,True)):
                    self.launch(ai.normal_change,batch.capacity,[s._path_start,s.positions,batch.ids,batch.kinds,
                        batch.areas,batch.eps,batch.count,e.mesh.dmin,e.mesh.dhat,e.mesh.stiffness,departed,self.energy_increment])
            f=e.mesh_frame
            self.launch(ai.friction_change,f.capacity,[s._path_start,s.positions,s.old,f.ids,f.weights,
                f.normals,f.loads,f.count,e.epsilon*dt,self.energy_increment])
            self.launch(amp.energy_correction,c.n,[c.base,c.u,s._path_start,s.positions,
                c.line_physical_gradient,e.gradient,c.owner,c.weights,c.n,self.energy_increment])
            self.launch(ctl.accept_trial_increment,1,[*args,self.energy_increment,self.increment_disabled])
        wp.capture_if(self.increment_cond,stable,ordinary)

    def trial_feasible_and_energy(self,dt,c0,c1,velocity_mode):
        """Audit a Newton trial and evaluate its unchanged physical objective."""
        s,c=self.s,self.c
        self.joint_paths(c.base,c.u,s._path_start,s.positions,c1,c1,reuse_self_path=True)
        self.launch(ctl.trial_feasible,1,[s.diagnostic,s.self_motion.stats,s.contact.min_gap_m,s.min_jacobian,self.trial_cond])
        if velocity_mode:
            self.launch(ac.full_path,1,[self.values,self.trial_cond,self.correction_metrics])
        def energy():
            self.value(s.positions,s.old,dt,c0,c1)
            self.trial_surface_status()
            self.alive(lambda:self.accept_trial(dt))
        wp.capture_if(self.trial_cond,energy)

    def record(self,state,output,control,dt,c0,c1):
        s,e,m,c=self.s,self.e,self.s.model,self.c
        rotation_search=c.rotation_line_search and not c.all_guided
        self.rotation_counts.zero_()
        if self.ccd_step is not None:self.ccd_step.reset()
        if self.path_cache is not None:
            # This zero is captured too: each replay/adaptive retry starts
            # without a certificate from the preceding physical attempt.
            self.path_cache.reset()
        self.launch(ctl.initialize,1,[self.status,self.values]); e.dt_device.fill_(dt)
        self.launch(ap.reset_observation,1,[self.report])
        from .abd_residual_report import reset
        self.launch(reset,1,[self.report])
        if c.temporal_tolerance_m:self.launch(at.reset,1,[self.report])
        velocity_mode=c.newton_convergence=='velocity'
        if velocity_mode:
            self.correction_metrics.zero_();self.correction_done.zero_()
        self.increment_disabled.zero_()
        self.material_projection_counts.zero_()
        inexact=c.inexact_friction and e.mesh.mu>0. and (c.friction_policy=='strict' or
            (c.friction_policy=='lagged' and c.friction_passes>1))
        per_newton=c.friction_policy=='newton' and e.mesh.mu>0.
        if per_newton:self.launch(afi.newton_frame_start,1,[self.status,self.friction_newton_mark])
        self.inner_tolerance.fill_(s.force_tolerance*(100 if inexact else 1))
        self.initial(state,control,c0)
        self.finite([c.old_q,c.old_qd],24)
        self.launch(ak.check_bindings,c.n,[state.particle_q,c.old_q,c.owner,c.weights,self.status])
        def seed():
            wp.copy(s.old,state.particle_q)
            self.launch(s.initialize_kernel,c.n,[s.old,state.particle_qd,state.particle_f,m.particle_mass,m.gravity,
                m.particle_world,s.backend.radii,c0,c1,s.contact.sphere_radius_m,
                s.contact.activation_distance_m,s.contact.barrier_stiffness,dt,s.contact.min_gap_m,
                s.positions,s.predictor,s.lagged_normals,s.lagged_loads])
            self.launch(ak.seed,c.nd,[s.old,c.old_q,c.old_qd,c.external,c.mass,wp.vec3d(*c.gravity),dt,c.n,c.u,c.predictor])
            wp.copy(c.uold,c.u)
            self.launch(ak.map_positions,c.n,[c.u,c.owner,c.weights,c.n,s.positions])
            self.joint_paths(c.uold,c.u,s.old,s.positions,c0,c1)
            self.check_path(1)
        self.alive(seed)
        self.alive(lambda:self.freeze(s.old,dt,c0))
        self.alive(lambda:self.seed_value(s.positions,s.old,dt,c0,c1))
        self.alive(lambda:self.launch(ctl.objective,1,[e.stats,e.mesh.batch.count,self.status,self.values]))
        if velocity_mode and c.velocity_residual_guard:
            self.launch(ac.start_residual_guard,1,[self.values,self.residual_guard_state])
        if inexact and c.friction_forcing:
            self.alive(lambda:self.launch(afi.relative_tolerance,1,[self.values,1,s.force_tolerance,c.friction_forcing,self.inner_tolerance]))
        def condition():
            if velocity_mode:
                self.launch(ac.condition,1,[self.status,self.values,self.correction_done,s.iterations,self.newton_cond])
            elif per_newton:
                self.launch(afi.one_newton_condition,1,[self.status,self.values,self.friction_newton_mark,
                    s.force_tolerance,s.iterations,self.newton_cond])
            elif inexact:
                self.launch(afi.condition,1,[self.status,self.values,self.inner_tolerance,s.force_tolerance,s.iterations,self.newton_cond])
            else:
                self.launch(ctl.newton_condition,1,[self.status,self.values,s.force_tolerance,s.iterations,self.newton_cond])
        def iteration():
            self.linearize_and_solve(s.positions,s.old,dt,c0,c1)
            if rotation_search:
                self.rotation_contacts.zero_();self.rotation_linear_ccd.fill_(1)
                b=e.mesh.batch
                self.launch(ar.contact_bodies,b.capacity,[b.ids,b.count,c.owner,self.rotation_contacts])
                self.launch(ar.eligibility,c.nb,[c.direction,c.radius,c.guide_axes,self.rotation_contacts,c.n,
                    self.rotation_eligible,self.rotation_linear_ccd])
            if velocity_mode:
                self.correction_metrics.zero_()
                self.launch(ac.vertex_correction,c.n,[c.direction,c.owner,c.weights,m.particle_mass,c.n,self.correction_metrics])
                self.launch(ac.affine_correction,c.nb,[c.direction,c.radius,c.n,self.correction_metrics])
                if c.velocity_stop_at_current:
                    self.launch(ac.coordinate_resolution,c.nd,[c.u,c.direction,c.dof_mass,self.correction_metrics])
            if c.stable_energy:
                self.base_contacts.copy(e.mesh.batch)
                if e.mesh.aggregation is not None:e.mesh.aggregation.save_energy()
                wp.copy(c.line_physical_gradient,e.gradient)
            wp.copy(c.base,c.u); wp.copy(s._path_start,s.positions); wp.copy(c.line_gradient,c.gradient)
            def advance():
                if self.ccd_step is not None:
                    def safe_step():
                        s._line_stats.zero_()
                        self.launch(ctl.trial,c.nd,[c.base,c.direction,c.line_gradient,c.dof_mass,self.values,c.u,s._line_stats])
                        self.launch(ak.map_positions,c.n,[c.u,c.owner,c.weights,c.n,self.ccd_step.endpoint])
                        self.ccd_step.compute(s._path_start,self.ccd_step.endpoint,self.values)
                    if rotation_search:
                        # A previous iteration's straight-ray certificate is
                        # not the ray followed by retracted trial endpoints.
                        self.ccd_step.valid.zero_()
                        wp.capture_if(self.rotation_linear_ccd,lambda:self.alive(safe_step))
                    else:self.alive(safe_step)
                self.launch(ctl.line_condition,1,[self.status,self.values,False,self.line_cond])
                def trial():
                    s._line_stats.zero_()
                    self.launch(ctl.trial,c.nd,[c.base,c.direction,c.line_gradient,c.dof_mass,self.values,c.u,s._line_stats])
                    if rotation_search:
                        self.rotation_changed.zero_()
                        self.launch(ar.change,c.nb,[c.base,c.direction,c.line_gradient,c.radius,
                            self.rotation_eligible,self.values,c.n,c.u,self.rotation_changed])
                        s._line_stats.zero_()
                        self.launch(ar.trial_stats,c.nd,[c.base,c.u,c.direction,c.line_gradient,s._line_stats])
                    self.launch(ak.map_positions,c.n,[c.u,c.owner,c.weights,c.n,s.positions])
                    self.launch(ctl.trial_direction,1,[s._line_stats,self.status])
                    def feasible():
                        self.trial_feasible_and_energy(dt,c0,c1,velocity_mode)
                    self.alive(feasible)
                    if rotation_search:
                        self.launch(ar.observe,1,[self.rotation_changed,self.status,self.correction_metrics,self.rotation_counts])
                    self.launch(ctl.line_condition,1,[self.status,self.values,True,self.line_cond])
                    if c.effective_backtrack_budget:
                        self.launch(afi.backtrack_budget,1,[self.status,self.line_cond,c.effective_backtrack_budget])
                wp.capture_while(self.line_cond,trial)
                if velocity_mode:
                    self.launch(ac.check,1,[self.status,self.values,self.correction_metrics,dt,
                        c.velocity_tolerance_m_s,c.transform_rate_tolerance_s,self.correction_done])
                    if c.velocity_residual_guard:
                        self.launch(ac.residual_guard,1,[self.status,self.values,self.residual_guard_state,self.correction_done])
            if velocity_mode and c.velocity_stop_at_current:
                self.launch(ac.stationary_candidate,1,[self.status,self.values,self.correction_metrics,
                    self.residual_guard_state,c.velocity_residual_guard,dt,
                    c.velocity_stop_factor*c.velocity_tolerance_m_s,
                    c.velocity_stop_factor*c.transform_rate_tolerance_s,self.stationary_cond])
                def certify_current():
                    s._line_stats.zero_()
                    self.launch(ctl.trial,c.nd,[c.base,c.direction,c.line_gradient,c.dof_mass,self.values,c.u,s._line_stats])
                    self.launch(ak.map_positions,c.n,[c.u,c.owner,c.weights,c.n,s.positions])
                    self.joint_paths(c.base,c.u,s._path_start,s.positions,c1,c1,reuse_self_path=True)
                    self.launch(ctl.trial_feasible,1,[s.diagnostic,s.self_motion.stats,s.contact.min_gap_m,
                        s.min_jacobian,self.trial_cond])
                    self.launch(ac.stationary_accept,1,[self.status,self.trial_cond,s._line_stats,
                        self.correction_metrics,self.correction_done])
                    # The certificate tests an unscaled proposal, but stopping
                    # keeps the base coordinates AND their already evaluated
                    # energy/gradient/forces. No stale trial is published.
                    wp.copy(c.u,c.base);wp.copy(s.positions,s._path_start)
                wp.capture_if(self.stationary_cond,certify_current)
                self.launch(ac.unfinished,1,[self.status,self.correction_done,self.advance_cond])
                wp.capture_if(self.advance_cond,advance)
            else:
                advance()
            condition()
        condition(); wp.capture_while(self.newton_cond,iteration)
        if c.newton_limit_policy=='accept_feasible':
            self.launch(ac.accept_iteration_limit,1,[self.status,self.values,self.correction_done,s.iterations])
        self.friction_count.zero_(); self.friction_change.zero_()
        if e.mesh.mu>0.:
            if c.friction_policy=='lagged':
                self.launch(afi.finite_pass_condition,1,[self.status,self.friction_count,c.friction_passes,self.friction_cond])
            else:self.launch(fi.start,1,[self.status,self.friction_cond])
            def friction_update():
                self.launch(fi.save_forces,c.n,[e.sphere_force,e.self_force,self.previous_contact_force])
                if c.reuse_contact_work:
                    self._freeze_from_current_contacts(s.positions,dt,c1)
                else:
                    self.freeze(s.positions,dt,c1,initial=False)
                self.alive(lambda:self._value_from_current_contacts(s.positions,s.old,dt,c0,c1))
                self.alive(lambda:self.launch(ctl.objective,1,[e.stats,e.mesh.batch.count,self.status,self.values]))
                self.friction_change.zero_()
                self.launch(fi.force_change,c.n,[e.sphere_force,e.self_force,self.previous_contact_force,self.friction_change])
                if c.friction_policy=='lagged':
                    self.launch(afi.record_refresh,1,[self.status,self.friction_change,self.friction_count])
                else:
                    # One refreshed frame per Newton step; permit a final check
                    # after the last allowed Newton, without a second inner budget.
                    maximum=max(s.friction_iterations,s.iterations+1) if per_newton else s.friction_iterations
                    self.launch(fi.check,1,[self.status,self.values,self.friction_change,self.friction_count,
                        self.friction_cond,maximum,s.force_tolerance])
                if per_newton:self.launch(afi.newton_frame_start,1,[self.status,self.friction_newton_mark])
                if inexact:
                    if c.friction_forcing:
                        self.launch(afi.relative_tolerance,1,[self.friction_change,0,s.force_tolerance,c.friction_forcing,self.inner_tolerance])
                    else:
                        self.launch(afi.update_tolerance,1,[self.friction_change,s.force_tolerance,self.inner_tolerance])
                if inexact and c.friction_policy=='lagged':
                    self.launch(afi.final_pass_tolerance,1,[self.friction_count,c.friction_passes,s.force_tolerance,self.inner_tolerance])
                condition(); wp.capture_while(self.newton_cond,iteration)
                if c.friction_policy=='lagged':
                    self.launch(afi.finite_pass_condition,1,[self.status,self.friction_count,c.friction_passes,self.friction_cond])
            wp.capture_while(self.friction_cond,friction_update)
        self.alive(lambda:self.final_geometry(c0,c1))
        def finish():
            s._sphere_publish_stats.zero_(); e.stats.zero_()
            self.launch(k.publish_force,c.n,[e.sphere_force,s._trial_contact_forces,s._sphere_publish_stats])
            self.launch(k.publish_force,c.n,[e.self_force,s._trial_self_contact_forces,e.stats])
            self.launch(ctl.forces,1,[s._sphere_publish_stats,e.stats,self.status,self.values])
            self.launch(s._kernels.finalize,c.n,[s.old,s.positions,dt,output.particle_q,output.particle_qd])
            wp.copy(output.body_q,state.body_q); wp.copy(output.body_qd,state.body_qd)
            self.launch(ak.publish,c.nb,[c.u,c.uold,c.radius,c.com,c.n,dt,c.next_q,c.next_qd,c.next_poses,c.next_velocities])
            self.finite([output.particle_q,output.particle_qd,c.next_q,c.next_qd],10)
        self.alive(finish)
        if c.temporal_tolerance_m:
            self.alive(lambda:self.launch(at.measure,c.n,[state.particle_qd,output.particle_qd,
                m.particle_mass,dt,self.report,self.status]))
            self.alive(lambda:self.launch(at.measure_affine,c.n,[c.old_qd,c.next_qd,
                c.owner,c.weights,dt,self.report,self.status]))
            self.launch(at.check,1,[self.report,self.status,c.temporal_tolerance_m])
        self.launch(ctl.report,1,[self.status,self.values,self.report])
        self.launch(ar.report,1,[self.rotation_counts,self.report])
        self.launch(ctl.path_report,1,[s.diagnostic,s.self_motion.stats,self.report])
        self.launch(fi.report,1,[self.friction_count,self.friction_change,self.report])
        self.launch(mp.report,1,[self.material_projection_counts,self.report])
        from .abd_residual_report import measure
        self.alive(lambda:self.launch(measure,c.nd,[c.gradient,m.particle_mass,c.n,self.report]))
        if velocity_mode:
            self.launch(ac.report,1,[self.correction_metrics,self.correction_done,dt,self.report])
            if c.velocity_residual_guard:
                self.launch(ac.report_residual_guard,1,[self.residual_guard_state,self.report])

    def final_geometry(self,c0,c1):
        s,c=self.s,self.c
        # Always observe the old-to-new physical chord, even when the selected
        # discrete integration contract does not make it an acceptance gate.
        self.joint_paths(c.uold,c.u,s.old,s.positions,c0,c1)
        self.launch(ap.observe_segment,1,[s.diagnostic,s.self_motion.stats,s.self_motion.witness,
            s.self_motion.faces,s.self_motion.edges,s.contact.min_gap_m,s.min_jacobian,self.report])
        if c.final_path_policy=='incremental':
            # Seed and ALL Newton segments already went through the original
            # CCD/J guards. Independently certify the final configuration too.
            # A failed diagnostic chord is not a certified physical trajectory.
            self.joint_paths(c.u,c.u,s.positions,s.positions,c1,c1)
        self.check_path(8)
        self.launch(ak.reset_shape,1,[c.shape_stats])
        self.launch(ak.shape_check,c.nb,[c.u,c.radius,c.n,c.max_shape_error,c.shape_stats,self.status])
