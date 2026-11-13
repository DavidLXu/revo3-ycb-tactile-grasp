"""One captured submission per IPC attempt, including Newton/PCG/line search.

Python records the graph; all numerical decisions execute in device kernels.
Only the completed report is downloaded. Overflow returns before any truncated
contact set is consumed and retries the complete attempt with larger buffers.
"""
import numpy as np
from collections import OrderedDict
from time import perf_counter
import warp as wp
import newton
from . import ipc_gpu_control as ctl
from . import ipc_gpu_kernels as k
from . import ipc_gpu_hessian as hk
from . import ipc_gpu_operator as op
from . import ipc_gpu_seed as seed_k
from . import ipc_energy_increment as energy_increment
from . import ipc_gpu_friction_iteration as fi
from .ipc_gpu_nested_cg import NestedCG
from .ipc_gpu_preconditioner import diagonal_inverse
from .ipc_gpu_contacts import ContactBatchGPU
from .ipc_self_audit import _bounds, _tet_pairs
from .ipc_precision import position_kernel, pose_kernel
from .ipc_vbd import (IPCStepRejected, IPCInfeasibleTarget,
                      IPCInitialContactCapacityExceeded, ImplicitStepDiagnostics)


ERRORS = {
    51: "GPU surface contact has degenerate rigid feature geometry",
    52: "GPU surface clipping polygon capacity exceeded; no partial force accepted",
    53: "GPU surface integration reaches invalid contact clearance",
    54: "GPU surface quadrature did not meet its accuracy target",
    55: "GPU surface pair/sample capacity exceeded; no partial force accepted",
    56: "GPU adaptive quadrature queue capacity exceeded; no partial force accepted",
    1: "GPU IPC seed crosses surfaces or invalid tet path",
    2: "GPU contact capacity exceeded; no pairs discarded",
    3: "Nonfinite GPU unified objective or collision bounds",
    4: "GPU block preconditioner is invalid",
    5: "GPU PCG rejected: true residual, tolerance, or descent check failed",
    6: "GPU IPC line search stalled or nonfinite search direction",
    7: "GPU unified IPC Newton convergence criterion not met within iteration budget",
    8: "GPU IPC physical surface path is not certified",
    9: "GPU IPC force is not finite in float32",
    10: "Nonfinite staged state or contact force; output unchanged",
    11: "Prescribed sphere target violates fixed backing clearance",
    12: "GPU friction fixed-point iteration did not converge",
    20: "sphere pose and velocity must be finite",
    21: "sphere orientation must be a unit quaternion",
    22: "rotating sphere contact is unsupported; prescribe translation only",
    23: "sphere_center must match the input state's sphere pose",
    24: "particle positions, velocities and external forces must be finite",
    25: "active material activation is unsupported by the IPC adapter",
    26: "Initial state has invalid tet geometry or sphere/surface overlap",
    27: "Initial soft volume overlaps or cannot be separated",
    28: "Initial soft surfaces intersect or lack self-contact clearance",
    31: "Initial rigid/soft solids overlap or contain one another",
    30: "Prescribed mesh/backing state does not match its accepted pose",
    29: "force tolerance must be finite and positive; iterations must be positive",
}


class IPCDeviceGraph:
    # Diagnostic A/B control. Production always chooses from current GPU
    # active matrices; this never disables contact detection or CCD.
    specialize_contact_cg = True
    stable_energy_differences = True
    cache_capacity = 8

    def __init__(self, solver):
        self.s, self.e = solver, solver.energy
        self.device = solver.model.device
        self.status = wp.zeros(16, dtype=int, device=self.device)
        self.values = wp.zeros(16, dtype=wp.float64, device=self.device)
        self.gate = wp.zeros(1, dtype=int, device=self.device)
        self.newton_cond = wp.zeros(1, dtype=int, device=self.device)
        self.friction_cond = wp.zeros(1, dtype=int, device=self.device)
        self.friction_count = wp.zeros(1, dtype=int, device=self.device)
        self.friction_change = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.previous_contact_force = wp.zeros(solver.model.particle_count, dtype=wp.vec3d, device=self.device)
        self.line_cond = wp.zeros(1, dtype=int, device=self.device)
        self.trial_cond = wp.zeros(1, dtype=int, device=self.device)
        self.seed_cond = wp.zeros(1, dtype=int, device=self.device)
        self.repair_cond = wp.zeros(1, dtype=int, device=self.device)
        self.increment_cond = wp.zeros(1, dtype=int, device=self.device)
        self.increment_disabled = wp.zeros(1, dtype=int, device=self.device)
        self.energy_increment = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.seed_delta = wp.zeros(solver.model.particle_count, dtype=wp.vec3d, device=self.device)
        self.seed_count = wp.zeros(solver.model.particle_count, dtype=wp.float64, device=self.device)
        self.report = None
        self.cg = None
        self.fixed_cg = None
        self.full_cg_cond = wp.zeros(1, dtype=int, device=self.device)
        self.fixed_cg_cond = wp.zeros(1, dtype=int, device=self.device)
        self.cg_signature = None
        self.initialized = False
        self.last_report = None
        self.path_cache = None
        self.graph = None
        self.pose_old = wp.zeros(1, dtype=wp.vec3d, device=self.device)
        self.pose_new = wp.zeros_like(self.pose_old)
        self.pose_delta = wp.zeros_like(self.pose_old)
        self.cache_enabled = True
        self.capture_count = 0
        self.cache_hits = 0
        self._graphs = OrderedDict()
        self.last_capture_ms = 0.0

    def launch(self, kernel, n, args):
        if n:
            # Contact eigensolves have high per-thread register pressure and
            # sparse active work. One warp per block distributes that work
            # across more SMs without changing any per-contact arithmetic.
            block_dim = 32 if kernel in (hk.normals, self.e.normal_hessian_kernel) else 256
            if args and isinstance(args[0], wp.array):
                kernel = position_kernel(kernel, args[0])
            wp.launch(pose_kernel(kernel), n, args, device=self.device, block_dim=block_dim)

    def _cache_key(self, state, output, control, dt):
        """Track captured constants and array ownership, never array contents.

        Input values may change in-place. Rebound arrays, buffer growth, changed
        solver settings and fault-injection hooks require a different graph.
        Keep strong references alongside each graph so old pointers cannot be
        recycled into a false cache hit. The LRU bounds uncommon caller buffers.
        """
        s, e, m, c = self.s, self.e, self.s.model, self.s.contact
        if e.material_model != "stable_neo_hookean" and self.stable_energy_differences:
            raise ValueError("log_neo_hookean requires stable_energy_differences=False; incremental SNH energy is incompatible")
        surface = getattr(e, "surface_contact", None)
        surface_key = None if surface is None else (
            id(surface), surface.dhat, surface.dmin, surface.stiffness, surface.mu,
            surface.rtol, surface.max_depth, surface.capacity, surface.pair_capacity,
            surface.work_capacity, surface.reuse_zero_friction,
            surface.bucket_refinement, surface.defer_tangent, surface.refinement_group_size)
        trial_hook=getattr(self,"trial_surface_status",None)
        trial_hook=getattr(trial_hook,"__func__",trial_hook)
        constants = (surface_key, trial_hook, dt, s.sphere_enabled, None if hasattr(s,'_physical_path_gap_device') else getattr(s, "physical_path_gap", None), s.iterations, s.force_tolerance, s.min_jacobian, s.seed_repair_iterations,
            getattr(s,"curved_path",None),getattr(getattr(s,"_curved_path_audit",None),"max_checks",None),
            s.repair_volume_seed, s.friction_iterations,
            getattr(s._seed_boundary, "__func__", s._seed_boundary),
            e.volume_min_jacobian, e.volume_barrier_activation, e.volume_barrier_scale,
            s.seed_includes_faces,
            s._has_certified_positions, c.sphere_radius_m, c.activation_distance_m,
            c.barrier_stiffness, c.min_gap_m, e.offset, e.barrier_offset, e.sphere_mu,
            e.epsilon, e.mesh.dhat, e.mesh.dmin, e.mesh.stiffness, e.mesh.mu,
            e.mesh.candidate_skin, e.mesh.max_capacity, e.mesh.parallel_search, e.mesh.aggregation_power, e.mesh.convergent_contact, e.mesh.taccel_discrete, e.mesh.taccel_rigid_area, e.mesh.mollifier_coefficient, e.mesh.ee_multiplicity,
            e.linear_tolerance, e.linear_max_iterations, e.material_model, e.material_value_kernel,
            e.material_hessian_kernel, e.material_free_hessian_kernel, e.normal_hessian_kernel,
            s.self_motion.min_gap, s.self_motion.query_block_dim, s.self_motion.parallel_pairs,getattr(s.self_motion,'fast_reject',False),
            s.self_motion.pair_capacity, m.particle_count, m.tet_count, m.joint_count,
            s.self_path_reuse_skin_m,
            id(e.multilevel), id(e.operator), id(e.fixed_operator), id(e.preconditioner),
            self.specialize_contact_cg,
            self.stable_energy_differences,
            getattr(self.cg, "zero_initial_guess", False),
            getattr(self, "warm_friction", False),
            getattr(self, "enabled", None), getattr(self, "reuse_seed", None))
        objects = (state, output, control, s, e, m, s.backend, s.self_motion,
            s.self_volume, e.mesh, e.tets, e.sphere, e.sphere_frame, e.mesh.batch,
            surface, getattr(surface, "frame", None),
            e.mesh_frame, e.mesh.candidate_cache, e.mesh.parallel_query, e.mesh.aggregation, e.mesh.convergent_surface,
            e.matrix, e.multilevel, self.cg, self.fixed_cg, self.path_cache,
            getattr(s,"_curved_path_audit",None))
        arrays, handles, resources = [], [], []
        def collect(value):
            if isinstance(value, wp.array):
                arrays.append(value)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, wp.Bvh):
                handles.append(value.id)
                resources.append(value)
        for obj in objects:
            if obj is not None:
                for name, value in vars(obj).items():
                    # Work reports contain host scalars only; do not rescan
                    # prior adaptive attempts while looking for graph buffers.
                    if obj is s and name in ('_attempt_work', 'last_step_work'):
                        continue
                    collect(value)
        # Include subclass scratch arrays, e.g. a guarded velocity predictor.
        for name, value in vars(self).items():
            if name != "velocity" and isinstance(value, wp.array):
                arrays.append(value)
        bindings = tuple((id(a), a.ptr, a.shape, a.strides, a.dtype) for a in arrays)
        hooks = tuple(getattr(getattr(self, name), "__func__", getattr(self, name))
            for name in ("record", "initial", "paths", "freeze", "_freeze_from_current_contacts", "value", "_value_from_current_contacts", "hessian", "solve", "accept_trial"))
        # BVHs and CG reductions also own internal device buffers that are not
        # exposed as direct array attributes on the solver.
        retained = arrays + resources + [self.cg, self.fixed_cg, e.multilevel]
        return (constants, bindings, tuple(handles), hooks), retained

    def alive(self, callback):
        self.launch(ctl.gate, 1, [self.status, self.gate])
        wp.capture_if(self.gate, callback)

    def finite(self, arrays, code):
        s = self.s
        s._data_invalid.zero_()
        for a in arrays:
            scalar = wp.float64 if a.dtype in (wp.vec3d, wp.float64) else wp.float32
            a = a.view(scalar).flatten()
            kernel = s._kernels.validate_finite_data_double if scalar == wp.float64 else s._kernels.validate_finite_data
            self.launch(kernel, a.size, [a, s._data_invalid])
        self.launch(ctl.flag_error, 1, [s._data_invalid, code, self.status])

    def prepare(self, state):
        """Allocate outside conditional bodies; initialize BVH spatial topology."""
        s, e = self.s, self.e
        if getattr(e, "surface_contact", None) is not None and not hasattr(self, 'c'):
            raise ValueError('surface contact currently requires the monolithic ABD device graph')
        if getattr(e, "surface_contact", None) is not None:
            if (e.surface_contact.dhat,e.surface_contact.dmin,e.surface_contact.stiffness,e.surface_contact.mu) != (e.mesh.dhat,e.mesh.dmin,e.mesh.stiffness,e.mesh.mu):
                raise ValueError('contact parameters changed; rebuild the surface backend explicitly')
        if s.self_path_reuse_skin_m:
            from .ipc_gpu_path_cache import GPUPathPrefixCache
            if self.path_cache is None or self.path_cache.skin != s.self_path_reuse_skin_m:
                self.path_cache = GPUPathPrefixCache(s.self_motion, s.self_path_reuse_skin_m)
        else:
            self.path_cache = None
        e.mesh.prepare_candidate_cache()
        report_shape = (s.iterations + 1, max(getattr(self,'report_columns',0),
            21 if getattr(s,"curved_path",False) else 18 if s.friction_iterations else 16))
        if self.report is None or self.report.shape != report_shape:
            self.report = wp.zeros(report_shape, dtype=wp.float64, device=self.device)
        if e.mesh_frame.capacity != e.mesh.batch.capacity:
            e.mesh_frame = ContactBatchGPU(e.mesh.batch.capacity, self.device).frame()
        signature = (e.linear_tolerance, e.linear_max_iterations, id(e.mesh.aggregation))
        if self.cg is None or signature != self.cg_signature:
            self.cg = NestedCG(e.operator, e.rhs, e.direction, M=e.preconditioner,
                tol=e.linear_tolerance, atol=1e-10, maxiter=e.linear_max_iterations,
                check_every=0, use_cuda_graph=True)
            self.cg.zero_initial_guess = True  # solve() zeros direction before every call.
            self.fixed_cg = NestedCG(e.fixed_operator, e.rhs, e.direction, M=e.preconditioner,
                tol=e.linear_tolerance, atol=1e-10, maxiter=e.linear_max_iterations,
                check_every=0, use_cuda_graph=True)
            self.fixed_cg.zero_initial_guess = True
            self.cg_signature = signature
        if not self.initialized:
            # Rebuild once outside capture; later refits allocate no memory.
            for features, bounds, bvh in zip((e.mesh.faces, e.mesh.edges), e.mesh.bounds, e.mesh.bvhs):
                self.launch(position_kernel(_bounds, state.particle_q), len(features),
                    [state.particle_q, state.particle_q, features, features.shape[1],
                     float((e.mesh.dhat+e.mesh.dmin)/2), *bounds, e.mesh.invalid])
                bvh.rebuild()
            e.mesh._bvh_initialized = True
            motion = s.self_motion
            for features, bounds, bvh in zip((motion.faces, motion.edges), motion.bounds[1:],
                                            (motion.face_bvh, motion.edge_bvh)):
                self.launch(position_kernel(_bounds, state.particle_q), len(features),
                    [state.particle_q, state.particle_q, features, features.shape[1],
                     motion.min_gap, *bounds, motion.invalid])
                bvh.rebuild()
            motion._bvh_initialized = True
            vol = s.self_volume
            self.launch(position_kernel(_bounds, state.particle_q), len(vol.tets),
                [state.particle_q, state.particle_q, vol.tets, 4, 0.0, vol.lower, vol.upper, vol.invalid])
            vol.bvh.rebuild()
            vol._bvh_initialized = True
            self.initialized = True

    def paths(self, old, new, c0, c1, containment=False):
        s, m = self.s, self.s.model
        self.launch(ctl.reset_path, 1, [s.diagnostic])
        if s.sphere_enabled:
            self.launch(position_kernel(s._kernels.validate_sphere_paths, old), m.particle_count,
                [old, new, s.backend.radii, s.contact.sphere_radius_m, c0, c1, s.diagnostic])
        self.launch(position_kernel(s._kernels.validate_tet_paths, old), m.tet_count,
            [old, new, m.tet_indices, m.tet_poses, s.diagnostic])
        self.launch(position_kernel(s._surface_kernels.validate_surface_paths, old), len(s.surface_triangles),
            [old, new, s.surface_triangles, c0, c1, self.e.offset, s.contact.min_gap_m, s.diagnostic])
        if containment and s.sphere_enabled:
            self.launch(position_kernel(s._surface_kernels.validate_sphere_outside_tets, old), m.tet_count,
                [old, m.tet_indices, c0, s.diagnostic])

    def check_path(self, code):
        s = self.s
        self.launch(ctl.check_path, 1, [s.diagnostic, s.self_motion.stats,
            s.contact.min_gap_m, s.min_jacobian, code, self.status, self.values])

    def initial(self, state, control, c0):
        s, m = self.s, self.s.model
        if s.sphere_enabled:
            self.launch(ctl.validate_body, 1, [state.body_q, state.body_qd, c0, self.status])
        self.finite([state.particle_q, state.particle_qd, state.particle_f], 24)
        if hasattr(s, "_extra_initial_bindings"):
            s._extra_initial_bindings(state.particle_q, self)
        s._control_invalid.zero_()
        if control is not None:
            for field in ("tet_activations", "tri_activations"):
                a = getattr(control, field, None)
                if a is not None:
                    self.launch(s._kernels.validate_passive_control, a.size, [a, s._control_invalid])
        self.launch(ctl.flag_error, 1, [s._control_invalid, 25, self.status])

        def geometry():
            self.paths(state.particle_q, state.particle_q, c0, c0, containment=True)
            s.self_motion.stats.zero_()
            self.check_path(26)
            if s._has_certified_positions:
                s._geometry_changed.zero_()
                self.launch(position_kernel(s._positions_changed, state.particle_q), m.particle_count,
                    [state.particle_q, s._certified_positions, s._geometry_changed])
            else:
                s._geometry_changed.fill_(1)

            def changed():
                vol = s.self_volume
                vol.invalid.zero_()
                vol.stats.zero_()
                self.launch(position_kernel(_bounds, state.particle_q), len(vol.tets),
                    [state.particle_q, state.particle_q, vol.tets, 4, 0.0, vol.lower, vol.upper, vol.invalid])
                vol.bvh.refit()
                self.launch(position_kernel(_tet_pairs, state.particle_q), len(vol.tets),
                    [state.particle_q, vol.tets, vol.lower, vol.upper, vol.bvh.id, vol.stats, vol.witnesses])
                self.launch(ctl.volume_status, 1, [vol.invalid, vol.stats, self.status])
                s.self_motion.check(state.particle_q, state.particle_q, readback=False)
                self.check_path(28)
                s._extra_initial_geometry(state.particle_q, self)
            self.alive(lambda: wp.capture_if(s._geometry_changed, changed))
        self.alive(geometry)

    def build(self, x, *, initial=False):
        e = self.e
        e.mesh.build(x, readback=False)
        if getattr(e, "surface_contact", None) is not None:
            from .ipc_surface_gpu import mask_old
            surface = getattr(e, "surface_contact", None)
            mask_old(surface, e.mesh)
            surface.build(x)
            # Any overflow, underresolved integral or invalid feature fails
            # the attempt explicitly; never retain a partial quadrature.
            from .ipc_surface_gpu import contact_status
            self.launch(contact_status, 1, [surface.invalid, self.status])
        self.launch(ctl.capacity, 1, [e.mesh.batch.count, e.mesh.invalid,
                                    e.mesh.batch.capacity, initial, self.status])

    def freeze(self, old, dt, c0, *, initial=True):
        self.build(old, initial=initial)
        self._freeze_from_current_contacts(old, dt, c0)

    def _freeze_from_current_contacts(self, old, dt, c0):
        """Relag only after a successful evaluation at these unchanged positions."""
        e = self.e
        def finish():
            if getattr(e, "surface_contact", None) is not None:e.surface_contact.freeze(old)
            e.mesh_frame.copy_candidates(e.mesh.batch, e.mesh.batch.capacity)
            for batch, dhat, dmin, stiffness, offset, mu in (
                (e.sphere_frame, e.contact.activation_distance_m, 0., e.contact.barrier_stiffness, e.barrier_offset, e.sphere_mu),
                (e.mesh_frame, e.mesh.dhat, e.mesh.dmin, e.mesh.stiffness, 0., e.mesh.mu)):
                self.launch(k.freeze_contact, batch.capacity, [old, batch.ids, batch.kinds, batch.areas,
                    batch.eps, batch.count, c0, dhat, dmin, stiffness, offset, mu,
                    batch.weights, batch.normals, batch.loads])
            from .taccel_contact import adjust_friction
            adjust_friction(e.mesh,old,e.mesh_frame,self.launch)
        self.alive(finish)

    def value(self, x, old, dt, c0, c1):
        self.build(x)
        self._value_from_current_contacts(x, old, dt, c0, c1)

    def _value_from_current_contacts(self, x, old, dt, c0, c1, *, particle_residual=True, surface_tangent=True):
        """Evaluate after build/freeze at exactly x, with no intervening motion.

        Internal graph optimization only. A relagged friction frame does not
        move vertices or change active normal contacts, so a second BVH query
        at the identical positions is redundant. All forces and energies are
        still recomputed with the new frame, including fixed-node reactions.
        """
        e, m = self.e, self.s.model
        def finish():
            e.stats.zero_()
            e.sphere_force.zero_()
            e.self_force.zero_()
            self.launch(k.inertia, e.n, [x, self.s.predictor, m.particle_mass, dt, e.gradient, e.stats])
            self.launch(e.material_value_kernel, m.tet_count, [x, old, e.tets.ids, m.tet_poses,
                m.tet_materials, dt, False, e.gradient, e.stats, e.tets.h])
            e.volume_barrier(x, value=True)
            for batch, dhat, dmin, stiffness, offset, force in (
                (e.sphere, e.contact.activation_distance_m, 0., e.contact.barrier_stiffness, e.barrier_offset, e.sphere_force),
                (e.mesh.batch, e.mesh.dhat, e.mesh.dmin, e.mesh.stiffness, 0., e.self_force)):
                self.launch(k.normals, batch.capacity, [x, batch.ids, batch.kinds, batch.areas,
                    batch.eps, batch.count, c1, dhat, dmin, stiffness, offset, False,
                    e.gradient, force, e.stats, batch.h])
            if e.mesh.aggregation is not None:e.mesh.aggregation.correct_energy(e.stats)
            for batch, force, constant in ((e.sphere_frame, e.sphere_force, False), (e.mesh_frame, e.self_force, True)):
                self.launch(k.friction, batch.capacity, [x, old, batch.ids, batch.count, c1, c0,
                    batch.weights, batch.normals, batch.loads, e.epsilon*dt, constant, False,
                    e.gradient, force, e.stats, batch.h])
            if getattr(e, "surface_contact", None) is not None:
                e.surface_contact.value(x, old, e.epsilon*dt, e.gradient, e.self_force, e.stats,compute_tangent=surface_tangent)
                from .ipc_surface_gpu import contact_status
                self.launch(contact_status, 1, [e.surface_contact.frame.invalid, self.status])
            # Coupled coordinates build their own complete residual after
            # restricting this physical gradient. The particle RHS is unused
            # there; its free-node norm is already included in that residual.
            if particle_residual:
                self.launch(k.residual, e.n, [e.gradient, m.particle_mass, e.stats, e.rhs])
        self.alive(finish)

    def hessian(self, x, old, dt, c0, c1):
        e, m = self.e, self.s.model
        self.launch(e.material_hessian_kernel, m.tet_count, [x, e.tets.ids, m.tet_poses, m.tet_materials, dt, e.tets.h])
        e.volume_barrier(x, hessian=True)
        for batch, dhat, dmin, stiffness, offset in (
            (e.sphere, e.contact.activation_distance_m, 0., e.contact.barrier_stiffness, e.barrier_offset),
            (e.mesh.batch, e.mesh.dhat, e.mesh.dmin, e.mesh.stiffness, 0.)):
            e.normal_hessian(self.launch, x, batch, c1, dhat, dmin, stiffness, offset)
        for batch in (e.sphere_frame, e.mesh_frame):
            self.launch(hk.friction, batch.capacity, [x, old, batch.ids, batch.count, c1, c0,
                batch.weights, batch.normals, batch.loads, e.epsilon*dt, batch.h])
        e._assemble_matrix()
        e.matrix.diagonal(e.diagonal)
        for b in (e.mesh.batch, e.mesh_frame):
            self.launch(op.active_diagonal, b.capacity,
                [b.ids, b.active_indices, b.active_count, b.h, m.particle_mass, e.diagonal])
        self.launch(diagonal_inverse, e.n, [e.diagonal, e.inverse, e.stats])
        if e.multilevel is not None:
            e.multilevel.build(e.matrix, (e.mesh.batch, e.mesh_frame), e.stats)
        self.launch(ctl.preconditioner, 1, [e.stats, self.status])

    def solve(self):
        e = self.e
        e.direction.zero_()
        e.linear_stats.zero_()
        def run(cg):
            result = cg()
            self.launch(op.pack_cg_status, 1, [result[0], result[1], e.linear_stats])
        if self.specialize_contact_cg:
            self.launch(ctl.select_contact_cg, 1,
                [e.mesh.batch.active_count, e.mesh_frame.active_count, self.full_cg_cond, self.fixed_cg_cond])
            wp.capture_if(self.full_cg_cond, lambda: run(self.cg))
            wp.capture_if(self.fixed_cg_cond, lambda: run(self.fixed_cg))
        else:
            run(self.cg)
        # Always check the true residual using the full operator, including
        # self-contact, independently of the selected iterative path.
        e._matvec(e.direction, e.direction, e.applied, 1.0, 0.0)
        self.launch(k.check_linear, e.n, [e.rhs, e.applied, e.direction, e.linear_stats])
        self.launch(ctl.linear, 1, [e.linear_stats, e.linear_tolerance, self.status, self.values])

    def check_increment_contacts(self):
        e, m = self.e, self.s.model
        for batch in (e.mesh.batch, e.mesh_frame):
            self.launch(energy_increment.free_contacts, batch.capacity,
                [batch.ids, batch.count, m.particle_mass, self.increment_disabled])

    def accept_trial(self, dt):
        s, e, m = self.s, self.e, self.s.model
        args = [e.stats, s._line_stats, s.diagnostic, e.mesh.batch.count,
                self.status, self.values, self.report]
        if not self.stable_energy_differences or s.positions.dtype != wp.vec3d:
            self.launch(ctl.accept_trial, 1, args)
            return
        self.launch(energy_increment.needed, 1, [e.stats, self.values, self.increment_cond])
        def ordinary():
            self.launch(ctl.accept_trial, 1, args)
        def stable():
            # The base contacts were recorded before any trial rebuilt them.
            # Keep this flag sticky through all backtracks of this iteration.
            self.check_increment_contacts()
            def calculate():
                self.energy_increment.zero_()
                self.launch(energy_increment.inertia_change, m.particle_count,
                    [s._path_start, s.positions, s.predictor, m.particle_mass, dt, self.energy_increment])
                self.launch(energy_increment.tet_change, m.tet_count,
                    [s._path_start, s.positions, s.old, m.tet_indices, m.tet_poses,
                     m.tet_materials, dt, e.volume_min_jacobian, e.volume_barrier_activation or 0.,
                     e.volume_barrier_scale, self.energy_increment])
                f = e.sphere_frame
                self.launch(energy_increment.sphere_change, e.sphere.capacity,
                    [s._path_start, s.positions, s.old, e.sphere.ids, e.sphere.areas,
                     self.pose_new, self.pose_old, e.barrier_offset, s.contact.activation_distance_m,
                     s.contact.barrier_stiffness, f.weights, f.normals, f.loads,
                     e.epsilon*dt, self.energy_increment])
                self.launch(ctl.accept_trial_increment, 1,
                    [*args, self.energy_increment, self.increment_disabled])
            # Avoid computing a partial increment only to discard it.
            wp.capture_if(self.increment_disabled, ordinary, calculate)
        wp.capture_if(self.increment_cond, stable, ordinary)

    def record(self, state, output, control, dt, c0, c1):
        s, e, m, c = self.s, self.e, self.s.model, self.s.contact
        if self.path_cache is not None:
            # The guard verifies BOTH endpoints against the certified tube.
            # Its proof also covers a nearby path from a later Newton iterate;
            # unconditional per-iteration invalidation wastes that certificate.
            # Reset at each physical attempt to bound configuration lifetime.
            self.path_cache.reset()
        if getattr(s,"curved_path",False):
            s._curved_path_audit.work.zero_()
        self.launch(ctl.initialize, 1, [self.status, self.values])
        e.dt_device.fill_(dt)
        self.initial(state, control, c0)

        def seed():
            wp.copy(s.old, state.particle_q)
            self.launch(s.initialize_kernel, m.particle_count,
                [s.old, state.particle_qd, state.particle_f, m.particle_mass, m.gravity,
                 m.particle_world, s.backend.radii, c0, c1, c.sphere_radius_m,
                 c.activation_distance_m, c.barrier_stiffness, dt, c.min_gap_m,
                 s.positions, s.predictor, s.lagged_normals, s.lagged_loads])
            # Prescribed backing motion is a Dirichlet condition. The hook can
            # also seed free nodes; it must preserve the world inertia predictor.
            s._seed_boundary()
            # Test the requested fixed geometry before repairing any free
            # nodes or recursively spending work on an impossible target.
            self.launch(seed_k.fixed_sphere_target, len(s.surface_triangles),
                [s.positions, s.surface_triangles, m.particle_mass, c1,
                 e.offset, c.min_gap_m, self.status])
            self.alive(repair_seed)

        def repair_seed():
            # Moving mounts already build a face-aware seed. Reapplying the
            # generic seed and then this hook just overwrote it with itself.
            if not s.seed_includes_faces:
                self.launch(ctl.reset_path, 1, [s.diagnostic])
                self.launch(s._surface_kernels.validate_surface_paths, len(s.surface_triangles),
                    [s.positions, s.positions, s.surface_triangles, c1, c1, e.offset, c.min_gap_m, s.diagnostic])
                self.launch(ctl.seed_needed, 1, [s.diagnostic, c.min_gap_m, self.seed_cond])
                def repair():
                    s._seed_face_weights.zero_()
                    args = [c0, self.pose_delta, e.offset, c.activation_distance_m, c.min_gap_m, s._seed_face_weights]
                    self.launch(k.sphere_seed_face_weights, len(s.surface_triangles), [s.old, s.surface_triangles, *args])
                    self.launch(k.sphere_seed_apply, m.particle_count, [s.old, m.particle_mass, *args, s.positions])
                    s._seed_boundary()
                wp.capture_if(self.seed_cond, repair)
            self.paths(s.old, s.positions, c0, c1)
            s.self_motion.check(s.old, s.positions, readback=False)
            if s.seed_repair_iterations:
                def repair_condition():
                    self.launch(seed_k.condition, 1, [s.diagnostic, s.self_motion.stats,
                        c.min_gap_m, s.min_jacobian, s.seed_repair_iterations,
                        s.repair_volume_seed, self.status, self.repair_cond])
                def project():
                    self.seed_delta.zero_()
                    self.seed_count.zero_()
                    self.launch(seed_k.reset_progress, 1, [self.status])
                    if s.repair_volume_seed:
                        self.launch(seed_k.volume, m.tet_count, [s.old, s.positions,
                            m.tet_indices, m.tet_poses, m.particle_mass, s.min_jacobian,
                            self.seed_delta, self.seed_count])
                    self.launch(seed_k.sphere, len(s.surface_triangles), [s.positions,
                        s.surface_triangles, m.particle_mass, c1, e.offset, c.min_gap_m,
                        self.seed_delta, self.seed_count, self.status])
                    self.launch(seed_k.self_witness, s.self_motion.witness.shape[0],
                        [s.old, s.positions, s.self_motion.faces, s.self_motion.edges, m.particle_mass,
                         s.self_motion.witness, s.self_motion.stats, s.self_motion.min_gap,
                         self.seed_delta, self.seed_count])
                    self.launch(seed_k.apply, m.particle_count,
                        [s.positions, m.particle_mass, self.seed_delta, self.seed_count, self.status])
                    self.paths(s.old, s.positions, c0, c1)
                    s.self_motion.check(s.old, s.positions, readback=False)
                    repair_condition()
                repair_condition()
                wp.capture_while(self.repair_cond, project)
            self.check_path(1)
        self.alive(seed)
        self.alive(lambda: self.freeze(s.old, dt, c0))
        self.alive(lambda: self.value(s.positions, s.old, dt, c0, c1))
        self.alive(lambda: self.launch(ctl.objective, 1, [e.stats, e.mesh.batch.count, self.status, self.values]))

        def newton_condition():
            self.launch(ctl.newton_condition, 1, [self.status, self.values, s.force_tolerance, s.iterations, self.newton_cond])

        def iteration():
            if self.stable_energy_differences and s.positions.dtype == wp.vec3d:
                self.increment_disabled.zero_()
                self.check_increment_contacts()
            self.hessian(s.positions, s.old, dt, c0, c1)
            self.alive(self.solve)
            wp.copy(s._path_start, s.positions)
            wp.copy(s._line_gradient, e.gradient)
            self.launch(ctl.line_condition, 1, [self.status, self.values, False, self.line_cond])
            def trial():
                s._line_stats.zero_()
                self.launch(ctl.trial, m.particle_count, [s._path_start, e.direction, s._line_gradient,
                    m.particle_mass, self.values, s.positions, s._line_stats])
                self.launch(ctl.trial_direction, 1, [s._line_stats, self.status])
                def feasible():
                    self.paths(s._path_start, s.positions, c1, c1)
                    if self.path_cache is None:
                        s.self_motion.check(s._path_start, s.positions, readback=False)
                    else:
                        self.path_cache.check(s._path_start, s.positions, self.values)
                    self.launch(ctl.trial_feasible, 1, [s.diagnostic, s.self_motion.stats,
                        c.min_gap_m, s.min_jacobian, self.trial_cond])
                    def energy():
                        self.value(s.positions, s.old, dt, c0, c1)
                        self.alive(lambda: self.accept_trial(dt))
                    wp.capture_if(self.trial_cond, energy)
                self.alive(feasible)
                self.launch(ctl.line_condition, 1, [self.status, self.values, True, self.line_cond])
            wp.capture_while(self.line_cond, trial)
            newton_condition()
        newton_condition()
        wp.capture_while(self.newton_cond, iteration)

        if s.friction_iterations:
            self.friction_count.zero_()
            self.friction_change.zero_()
            if e.sphere_mu > 0. or e.mesh.mu > 0.:
                self.launch(fi.start, 1, [self.status, self.friction_cond])
                def friction_update():
                    self.launch(fi.save_forces, e.n,
                        [e.sphere_force, e.self_force, self.previous_contact_force])
                    # Freeze at x, not old. Friction still measures x-old.
                    self.freeze(s.positions, dt, c1, initial=False)
                    self.alive(lambda: self._value_from_current_contacts(s.positions, s.old, dt, c0, c1))
                    self.alive(lambda: self.launch(ctl.objective, 1,
                        [e.stats, e.mesh.batch.count, self.status, self.values]))
                    self.friction_change.zero_()
                    self.launch(fi.force_change, e.n,
                        [e.sphere_force, e.self_force, self.previous_contact_force, self.friction_change])
                    self.launch(fi.check, 1, [self.status, self.values, self.friction_change,
                        self.friction_count, self.friction_cond, s.friction_iterations, s.force_tolerance])
                    newton_condition()
                    wp.capture_while(self.newton_cond, iteration)
                wp.capture_while(self.friction_cond, friction_update)

        def final_path():
            if hasattr(s,"_check_physical_path"):
                s._check_physical_path(self,c0,c1)
            else:
                self.paths(s.old, s.positions, c0, c1)
                s.self_motion.check(s.old, s.positions, readback=False,
                                    gap_override=getattr(s, "_physical_path_gap_device",getattr(s, "physical_path_gap", None)))
            self.check_path(8)
        self.alive(final_path)
        def finish():
            s._sphere_publish_stats.zero_()
            e.stats.zero_()
            self.launch(k.publish_force, e.n, [e.sphere_force, s._trial_contact_forces, s._sphere_publish_stats])
            self.launch(k.publish_force, e.n, [e.self_force, s._trial_self_contact_forces, e.stats])
            self.launch(ctl.forces, 1, [s._sphere_publish_stats, e.stats, self.status, self.values])
            self.launch(s._kernels.finalize, m.particle_count, [s.old, s.positions, dt, output.particle_q, output.particle_qd])
            wp.copy(output.body_q, state.body_q)
            wp.copy(output.body_qd, state.body_qd)
            if s.sphere_enabled:
                self.launch(ctl.publish_sphere, 1, [c1, self.pose_delta, dt, output.body_q, output.body_qd])
            if m.joint_count:
                newton.eval_ik(m, output, output.joint_q, output.joint_qd)
            arrays = [output.particle_q, output.particle_qd, output.body_q, output.body_qd]
            if m.joint_count:
                arrays += [output.joint_q, output.joint_qd]
            self.finite(arrays, 10)
        self.alive(finish)
        self.launch(ctl.report, 1, [self.status, self.values, self.report])
        self.launch(ctl.path_report, 1, [s.diagnostic, s.self_motion.stats, self.report])
        if s.friction_iterations:
            self.launch(fi.report, 1, [self.friction_count, self.friction_change, self.report])
        if getattr(s,"curved_path",False):
            from .rigid_curved_path import report_work
            self.launch(report_work,1,[s._curved_path_audit.work,self.report])

    def attempt(self, state, output, control, dt, c0, c1):
        s, e = self.s, self.e
        while True:  # Only exceptional capacity growth returns here.
            started_attempt = perf_counter()
            self.last_report = None
            self.prepare(state)
            e._evaluation, e._hessian_ready = None, False
            e.dt, e.old_center = dt, np.array(c0, dtype=float)
            # This small launch updates parameters read by all cached kernels.
            wp.launch(ctl.update_pose, 1, [wp.vec3d(*c0), wp.vec3d(*c1),
                self.pose_old, self.pose_new, self.pose_delta], device=self.device)
            try:
                key, retained = self._cache_key(state, output, control, dt)
                cached = self._graphs.get(key) if self.cache_enabled else None
                self.last_capture_ms = 0.0
                if cached is None:
                    started = perf_counter()
                    with wp.ScopedCapture(device=self.device) as capture:
                        self.record(state, output, control, dt, self.pose_old, self.pose_new)
                    self.graph = capture.graph
                    self.last_capture_ms = 1000*(perf_counter()-started)
                    self.capture_count += 1
                    if self.cache_enabled:
                        # CG may allocate persistent scratch during first recording.
                        key, retained = self._cache_key(state, output, control, dt)
                        self._graphs[key] = (self.graph, retained)
                        while len(self._graphs) > self.cache_capacity:
                            self._graphs.popitem(last=False)
                else:
                    self.graph = cached[0]
                    self._graphs.move_to_end(key)
                    self.cache_hits += 1
                wp.capture_launch(self.graph)
                rows = self.report.numpy()  # The only readback in a normal physical step.
            finally:
                # Recording H nodes does not mean those conditional nodes ran.
                # The independently callable host energy API must reevaluate.
                e._evaluation, e._hessian_ready = None, False
            r = rows[0]
            self.last_report = r.copy()
            code = int(r[0])
            path_diagnostics = None
            if code in (1, 6, 7, 8):
                j, node_gap, surface_gap, invalid, self_flags = rows[1, 9:14]
                failed = []
                for name, value, limit in (("tet_volume", j, s.min_jacobian),
                        ("particle_sphere_gap", node_gap, s.contact.min_gap_m),
                        ("surface_sphere_gap", surface_gap, s.contact.min_gap_m)):
                    # An empty minimum is +inf and passes the GPU path check.
                    if np.isnan(value) or value <= limit:
                        failed.append(name)
                if invalid: failed.append("uncertified_geometry")
                if int(self_flags) & 1: failed.append("self_surface_path")
                if int(self_flags) & 2: failed.append("invalid_self_bounds")
                path_diagnostics = dict(failed_checks=failed,
                    min_tet_jacobian=float(j), min_particle_gap_m=float(node_gap),
                    min_surface_gap_bound_m=float(surface_gap),
                    uncertified_geometry_count=int(invalid), self_path_flags=int(self_flags))
            s._attempt_work.append(dict(dt_s=float(dt), code=code, residual_N=float(r[2]), **self.extra_report_fields(r),
                curved_path_work=([int(v) for v in r[18:21]] if getattr(s,"curved_path",False) else None),
                warm_seed_accepted=int(rows[1,16]) if getattr(self,'has_warm_seed',False) else None,
                friction_iterations=int(r[16]) if s.friction_iterations else 0,
                friction_update_error_N=float(r[17]) if s.friction_iterations else None,
                newton_iterations=int(r[1]), backtracks=int(r[13]),
                hessians=int(r[14]), pcg_iterations=int(r[15]),
                capacity_required=int(r[12]),
                capacity_phase=('initial' if rows[1, 14] else 'iterate') if code == 2 else None,
                path_diagnostics=path_diagnostics,
                capture_ms=self.last_capture_ms,
                wall_ms=1000*(perf_counter()-started_attempt)))
            if code == 2 and r[12] <= e.mesh.max_capacity:
                required = int(r[12])
                capacity = min(e.mesh.max_capacity, max(2*e.mesh.batch.capacity, 1 << (required-1).bit_length()))
                e.mesh.batch = ContactBatchGPU(capacity, self.device)
                e.mesh.growths += 1
                continue
            if code == 2 and rows[1, 14]:
                # Every first half-step freezes these same input contacts.
                # Only exhausted storage is terminal; permitted growth above
                # and overflow at a new iterate retain their existing retries.
                raise IPCInitialContactCapacityExceeded(r[12], e.mesh.max_capacity)
            if code:
                message = ERRORS.get(code, f"Unknown GPU IPC failure {code}")
                if code >= 20 and code not in (53,54):
                    raise ValueError(message)
                if code == 11:
                    raise IPCInfeasibleTarget(message)
                raise IPCStepRejected(message)
            break
        iterations = int(r[1])
        history = []
        for row in rows[1:iterations+1]:
            history.append(dict(energy_before_J=float(row[0]), energy_after_J=float(row[1]),
                alpha=float(row[2]), residual_before_N=float(row[3]), contacts=int(row[4]), min_jacobian=float(row[5]),
                linear=dict(iterations=int(row[6]), true_residual_N=float(row[7]), tolerance_N=float(row[8]))))
        if history:
            e.last_linear = history[-1]["linear"].copy()
        e.mesh.last_count = int(r[3])
        e._evaluation, e._hessian_ready = None, False
        s._trace.append(dict(dt_s=dt, iterations=iterations, residual_N=float(r[2]), contact_count=int(r[3]), **self.extra_report_fields(r),
            friction_iterations=int(r[16]) if s.friction_iterations else 0,
            friction_update_error_N=float(r[17]) if s.friction_iterations else None,
            max_self_force_N=float(r[8]), internal_force_sum_N=r[9:12].tolist(), energy_iterations=history,
            control="device", backtracks=int(r[13]), hessian_evaluations=int(r[14]), pcg_iterations=int(r[15]),
            seed_repair_iterations=int(rows[1, 15]),
            graph_capture_ms=self.last_capture_ms, graph_reused=cached is not None))
        return ImplicitStepDiagnostics(1, iterations, float(r[2]), float(r[4]), float(r[5]),
            0., True, float(r[6]), True, int(r[7]))

    def extra_report_fields(self, row):
        """Subclass diagnostics carried in the existing single GPU readback."""
        return {}
