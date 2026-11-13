"""Newton adapter with a GPU-resident unified IPC numerical pipeline.

The default device-control path captures Newton, PCG, line search and validation
in a conditional CUDA graph. The host reads the completed report once. An
explicit device_control=False reference retains the original host decisions.
"""

from time import perf_counter
import numpy as np
import warp as wp
from .ipc_precision import position_kernel, copy_positions
from .ipc import positive_scalar, points_and_center
from .ipc_vbd import SolverIPCVBD, IPCStepRejected, ImplicitStepDiagnostics
from .ipc_gpu_energy import UnifiedEnergyGPU
from . import ipc_gpu_kernels as k
from .ipc_gpu_operator import merge_invalid


class SolverUnifiedIPCGPU(SolverIPCVBD):
    is_gpu_unified = True
    seed_includes_faces = False

    @property
    def friction_mu(self):
        """Sphere friction shares storage with the GPU energy configuration."""
        return self.energy.sphere_mu if hasattr(self, "energy") else self._friction_mu

    @friction_mu.setter
    def friction_mu(self, value):
        self._friction_mu = value
        if hasattr(self, "energy"):
            self.energy.sphere_mu = value

    @property
    def friction_epsilon(self):
        return self.energy.epsilon if hasattr(self, "energy") else self._friction_epsilon

    @friction_epsilon.setter
    def friction_epsilon(self, value):
        self._friction_epsilon = value
        if hasattr(self, "energy"):
            self.energy.epsilon = value

    def __init__(
        self,
        model,
        contact,
        *,
        self_activation_distance_m=0.0002,
        self_barrier_stiffness=1000.0,
        self_friction_coefficient=0.0,
        contact_capacity=4096,
        max_contact_capacity=262144,
        contact_candidate_skin_m=0.0,
        self_path_reuse_skin_m=0.0,
        linear_tolerance=1e-5,
        linear_max_iterations=2000,
        state_precision="float64",
        device_control=True,
        seed_repair_iterations=32,
        repair_volume_seed=False,
        material_hessian="spectral",
        contact_hessian="free",
        linear_preconditioner="jacobi",
        volume_barrier_activation=None,
        volume_barrier_scale=1.,
        friction_iterations=0,
        **kwargs,
    ):
        if not isinstance(device_control, bool):
            raise ValueError("device_control must be a bool")
        self.device_control = device_control
        if isinstance(friction_iterations, (bool, np.bool_)) or not isinstance(friction_iterations, (int, np.integer)) or not 0 <= friction_iterations < 2**31-1:
            raise ValueError("friction_iterations must be a nonnegative integer")
        if friction_iterations and not device_control:
            raise ValueError("iterated friction requires GPU device control")
        self.friction_iterations = int(friction_iterations)
        self.self_path_reuse_skin_m = positive_scalar(self_path_reuse_skin_m, "self path reuse skin", allow_zero=True)
        if isinstance(seed_repair_iterations, bool) or not isinstance(seed_repair_iterations, int) or seed_repair_iterations < 0:
            raise ValueError("seed repair iterations must be a nonnegative integer")
        self.seed_repair_iterations = seed_repair_iterations
        if not isinstance(repair_volume_seed, bool):
            raise ValueError("repair_volume_seed must be a bool")
        # Prefer adaptive time refinement to changing a severely compressed
        # initial guess at the coarse step. Volume projection is experimental.
        self.repair_volume_seed = repair_volume_seed
        self._device_graph = None
        self._active_control = None
        if state_precision not in ("float32", "float64"):
            raise ValueError("state_precision must be float32 or float64")
        self.state_precision = state_precision
        self.position_dtype = wp.vec3d if state_precision == "float64" else wp.vec3
        if not model.device.is_cuda:
            raise ValueError("Unified IPC GPU requires a CUDA Newton model")
        if kwargs.get("require_convergence", True) is not True:
            raise ValueError("Unified GPU IPC requires force-residual convergence")
        linear_tolerance = positive_scalar(linear_tolerance, "linear tolerance")
        if linear_tolerance >= 1:
            raise ValueError("linear tolerance must be below one")
        if (
            isinstance(linear_max_iterations, bool)
            or int(linear_max_iterations) != linear_max_iterations
            or linear_max_iterations < 1
        ):
            raise ValueError("linear max iterations must be a positive integer")
        super().__init__(model, contact, **kwargs)
        if not self.sphere_enabled and not device_control:
            raise ValueError("mesh IPC currently requires device_control=True")
        self.energy = UnifiedEnergyGPU(
            model,
            contact,
            self.surface_triangles.numpy(),
            collision_faces=self.self_motion.host_faces,
            rigid_groups=kwargs.get("rigid_groups"),
            self_dhat=self_activation_distance_m,
            self_stiffness=self_barrier_stiffness,
            self_min_gap=kwargs.get("self_contact_gap_m", 1e-7),
            self_mu=self_friction_coefficient,
            sphere_mu=self.friction_mu,
            friction_epsilon=self.friction_epsilon,
            contact_capacity=contact_capacity,
            max_contact_capacity=max_contact_capacity,
            contact_candidate_skin=contact_candidate_skin_m,
            linear_tolerance=linear_tolerance,
            linear_max_iterations=linear_max_iterations,
            material_hessian=material_hessian,
            contact_hessian=contact_hessian,
            linear_preconditioner=linear_preconditioner,
            volume_min_jacobian=self.min_jacobian,
            volume_barrier_activation=volume_barrier_activation,
            volume_barrier_scale=volume_barrier_scale,
        )
        self.mesh_contact = self.energy.mesh
        self.surface_collision_radius_m = self.energy.offset
        self.positions = wp.empty(
            model.particle_count, dtype=self.position_dtype, device=model.device
        )
        self.old = wp.empty_like(self.positions)
        self._path_start = wp.empty_like(self.positions)
        self._certified_positions = wp.empty_like(self.positions)
        self._staged = self.state()
        self._line_gradient = wp.empty_like(self.energy.gradient)
        self._line_stats = wp.zeros(3, dtype=wp.float64, device=model.device)
        self._sphere_publish_stats = wp.zeros(6, dtype=wp.float64, device=model.device)
        self._seed_face_weights = wp.zeros(model.particle_count, dtype=wp.float64, device=model.device)
        self.self_contact_forces = wp.zeros_like(model.particle_q)
        self._trial_self_contact_forces = wp.zeros_like(model.particle_q)
        self.last_unified_trace = ()
        self._trace = []
        self._attempt_work = []
        self.last_step_work = None

    def state(self):
        """Newton State with canonical GPU positions/velocities at solver precision.

        Use this factory instead of model.state(). Float64 coordinates persist
        between steps; there is no hidden float32 mirror driving the physics.
        External force and rigid-body buffers retain Newton's native layouts.
        """
        result = self.model.state()
        if not self.model.body_count:
            # Newton omits these arrays for a pure particle model. Keep the
            # transactional state layout uniform without inventing a body.
            for field, dtype in (("body_q", wp.transform), ("body_qd", wp.spatial_vector),
                                 ("body_f", wp.spatial_vector)):
                setattr(result, field, wp.empty(0, dtype=dtype, device=self.model.device))
        if self.position_dtype == wp.vec3d:
            for field in ("particle_q", "particle_qd"):
                source = getattr(result, field)
                target = wp.empty(
                    source.shape, dtype=wp.vec3d, device=self.model.device
                )
                copy_positions(source, target)
                setattr(result, field, target)
        return result

    def to_float32_state(self, state):
        """Return an independent, lossy Newton state for float32-only consumers.

        This is an export, never the input for continuing a float64 simulation.
        The exported coordinates do not inherit the physical residual guarantee.
        """
        self._validate_states(state, state)
        result = self.model.state()
        for field in ("particle_q", "particle_qd"):
            source, target = getattr(state, field), getattr(result, field)
            copy_positions(source, target)
        for field in (
            "particle_f",
            "body_q",
            "body_qd",
            "body_f",
            "joint_q",
            "joint_qd",
        ):
            source, target = getattr(state, field, None), getattr(result, field, None)
            if source is not None and target is not None and source.size:
                wp.copy(target, source)
        return result

    def _validate_states(self, state_in, state_out):
        for state in (state_in, state_out):
            for field in ("particle_q", "particle_qd"):
                if (
                    getattr(getattr(state, field, None), "dtype", None)
                    != self.position_dtype
                ):
                    raise ValueError(
                        f"GPU IPC {self.state_precision} requires solver.state() buffers; "
                        f"{field} has the wrong dtype"
                    )
        return super()._validate_states(state_in, state_out)

    def _attempt(self, state_in, state_out, dt, c0, c1):
        if self.device_control:
            if self._device_graph is None:
                from .ipc_gpu_graph import IPCDeviceGraph
                self._device_graph = IPCDeviceGraph(self)
            return self._device_graph.attempt(state_in, state_out, self._active_control, dt, c0, c1)
        m, c = self.model, self.contact
        wp.copy(self.old, state_in.particle_q)
        wp.launch(
            position_kernel(self._kernels.initialize, self.old),
            m.particle_count,
            inputs=[
                self.old,
                state_in.particle_qd,
                state_in.particle_f,
                m.particle_mass,
                m.gravity,
                m.particle_world,
                self.backend.radii,
                wp.vec3d(*c0),
                wp.vec3d(*c1),
                c.sphere_radius_m,
                c.activation_distance_m,
                c.barrier_stiffness,
                dt,
                c.min_gap_m,
                self.positions,
                self.predictor,
                self.lagged_normals,
                self.lagged_loads,
            ],
            device=m.device,
        )
        self._seed_boundary()
        # Keep the local vertex seed when it is already in the surface barrier
        # domain: translating extra vertices can unnecessarily close a separate
        # soft/soft gap. Repair only the missing face-contact case on the GPU.
        self._reset_diagnostic()
        wp.launch(
            position_kernel(self._surface_kernels.validate_surface_paths, self.positions),
            len(self.surface_triangles),
            inputs=[self.positions, self.positions, self.surface_triangles,
                    wp.vec3d(*c1), wp.vec3d(*c1), self.energy.offset, c.min_gap_m,
                    self.diagnostic], device=m.device,
        )
        surface_seed = self.diagnostic.numpy()
        if surface_seed[0] or surface_seed[5] <= c.min_gap_m:
            self._seed_face_weights.zero_()
            seed_args = [wp.vec3d(*c0), wp.vec3d(*(c1 - c0)), self.energy.offset,
                         c.activation_distance_m, c.min_gap_m, self._seed_face_weights]
            wp.launch(position_kernel(k.sphere_seed_face_weights, self.old), len(self.surface_triangles),
                      inputs=[self.old, self.surface_triangles, *seed_args], device=m.device)
            wp.launch(position_kernel(k.sphere_seed_apply, self.old), m.particle_count,
                      inputs=[self.old, m.particle_mass, *seed_args, self.positions], device=m.device)
            self._seed_boundary()
        seed = self._validate_paths(self.old, self.positions, c0, c1)
        if not self.self_motion.check(self.old, self.positions)["certified"]:
            raise IPCStepRejected("GPU IPC seed crosses soft surfaces")
        self.energy.begin_step(self.old, dt, c0)
        min_gap, min_j, min_surface = seed[2], seed[3], seed[5]
        history = []
        e, residual, count = self.energy.evaluate(
            self.positions, self.old, self.predictor, c1, hessian=False
        )
        for iteration in range(self.iterations + 1):
            if residual <= self.force_tolerance:
                break
            if iteration == self.iterations:
                raise IPCStepRejected(
                    f"GPU unified IPC residual {residual:.6g} N after {iteration} iterations"
                )
            self.energy.linearize()
            direction = self.energy.solve()
            wp.copy(self._path_start, self.positions)
            wp.copy(self._line_gradient, self.energy.gradient)
            accepted = False
            alpha = 1.0
            for backtrack in range(60):
                self._line_stats.zero_()
                wp.launch(
                    position_kernel(k.trial, self.positions),
                    m.particle_count,
                    inputs=[
                        self._path_start,
                        direction,
                        self._line_gradient,
                        m.particle_mass,
                        alpha,
                        self.positions,
                        self._line_stats,
                    ],
                    device=m.device,
                )
                slope, change, invalid = self._line_stats.numpy()
                if invalid or not np.all(np.isfinite([slope, change])):
                    raise IPCStepRejected(
                        "Nonfinite GPU IPC search direction or candidate state"
                    )
                if slope >= 0 or change == 0:
                    break
                try:
                    path = self._validate_paths(
                        self._path_start, self.positions, c1, c1
                    )
                    if not self.self_motion.check(self._path_start, self.positions)[
                        "certified"
                    ]:
                        raise IPCStepRejected("GPU IPC iterate crosses soft surfaces")
                    trial_e, trial_residual, trial_count = self.energy.evaluate(
                        self.positions, self.old, self.predictor, c1, hessian=False
                    )
                    if trial_e <= e + 1e-4 * slope:
                        accepted = True
                        break
                except (IPCStepRejected, ValueError):
                    pass
                alpha *= 0.5
            if not accepted:
                raise IPCStepRejected(
                    f"GPU IPC line search stalled at residual {residual:.6g} N"
                )
            history.append(
                {
                    "energy_before_J": e,
                    "energy_after_J": trial_e,
                    "alpha": alpha,
                    "residual_before_N": residual,
                    "contacts": count,
                    "min_jacobian": float(path[3]),
                    "linear": dict(self.energy.last_linear),
                }
            )
            min_gap, min_j = min(min_gap, path[2]), min(min_j, path[3])
            min_surface = min(min_surface, path[5])
            # The accepted trial already computed E, gradient, forces, residual
            # and the exact contact set at this position. Keep them for the next
            # convergence check and build only H if another update is needed.
            e, residual, count = trial_e, trial_residual, trial_count
        # Independently certify the physical segment, not just optimization moves.
        path = self._validate_paths(self.old, self.positions, c0, c1)
        audit = self.self_motion.check(self.old, self.positions)
        if not audit["certified"]:
            raise IPCStepRejected("GPU IPC physical self-contact path is not certified")
        self.energy.stats.zero_()
        self._sphere_publish_stats.zero_()
        wp.launch(
            k.publish_force,
            m.particle_count,
            inputs=[
                self.energy.sphere_force,
                self._trial_contact_forces,
                self._sphere_publish_stats,
            ],
            device=m.device,
        )
        wp.launch(
            k.publish_force,
            m.particle_count,
            inputs=[
                self.energy.self_force,
                self._trial_self_contact_forces,
                self.energy.stats,
            ],
            device=m.device,
        )
        wp.launch(merge_invalid, 1, [self._sphere_publish_stats, self.energy.stats], device=m.device)
        force_stats = self.energy.stats.numpy()
        if force_stats[2]:
            raise IPCStepRejected("GPU IPC force is not finite in float32")
        wp.launch(
            position_kernel(self._kernels.finalize, self.old),
            m.particle_count,
            inputs=[
                self.old,
                self.positions,
                dt,
                state_out.particle_q,
                state_out.particle_qd,
            ],
            device=m.device,
        )
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)
        wp.launch(
            self._kernels.set_sphere_state,
            1,
            inputs=[
                wp.vec3(*c1),
                wp.vec3(*((c1 - c0) / dt)),
                state_out.body_q,
                state_out.body_qd,
            ],
            device=m.device,
        )
        self._trace.append(
            {
                "dt_s": dt,
                "iterations": iteration,
                "residual_N": residual,
                "contact_count": count,
                "max_self_force_N": float(force_stats[1]),
                "internal_force_sum_N": force_stats[3:6].tolist(),
                "energy_iterations": history,
            }
        )
        return ImplicitStepDiagnostics(
            1,
            iteration,
            residual,
            min(min_gap, path[2]),
            min(min_j, path[3]),
            0.0,
            True,
            min(min_surface, path[5]),
            True,
            audit["candidate_pairs"],
        )

    def _seed_boundary(self):
        """Optional prescribed motion of mass-zero nodes (default: fixed)."""

    def _extra_initial_geometry(self, positions, graph):
        """Optional GPU initial solid-containment/binding validation."""

    @property
    def initialize_kernel(self):
        if self.sphere_enabled:
            return self._kernels.initialize
        from .rigid_mesh_gpu import initialize_mesh
        return initialize_mesh

    def _advance(self, *args):
        mark = len(self._trace)
        try:
            return super()._advance(*args)
        except Exception:
            del self._trace[mark:]
            raise

    def _device_step(self, state_in, state_out, control, contacts, dt, *, sphere_center, sphere_target):
        if contacts is not None:
            raise ValueError("IPC adapter owns sphere contacts; Newton contact buffers are unsupported")
        dt = positive_scalar(dt, "dt")
        self._validate_states(state_in, state_out)
        _, c0 = points_and_center(np.empty((0, 3)), sphere_center)
        _, c1 = points_and_center(np.empty((0, 3)), sphere_target)
        maximum = np.finfo(np.float32).max
        if np.any(np.abs(c0) > maximum) or np.any(np.abs(c1) > maximum):
            raise ValueError("sphere centers must be representable in float32")
        c0, c1 = c0.astype(np.float32).astype(float), c1.astype(np.float32).astype(float)
        with np.errstate(over="ignore", invalid="ignore"):
            velocity = (c1-c0)/dt
        if not np.all(np.isfinite(velocity)) or np.any(np.abs(velocity) > maximum):
            raise ValueError("prescribed sphere velocity must be finite in float32")
        if control is not None:
            for field, count in (("tet_activations", self.model.tet_count), ("tri_activations", self.model.tri_count)):
                a = getattr(control, field, None)
                if a is not None and (a.shape != (count,) or a.dtype != wp.float32 or a.device != self.model.device):
                    raise ValueError("material activation arrays must match the model")
        self._active_control = control
        staged = self._staged
        diagnostic = self._advance(state_in, staged, dt, c0, c1, 0)
        # The completed graph checked every staged field before this publication.
        fields = ["particle_q", "particle_qd", "body_q", "body_qd"]
        if self.model.joint_count:
            fields += ["joint_q", "joint_qd"]
        for field in fields:
            wp.copy(getattr(state_out, field), getattr(staged, field))
        wp.copy(self.contact_forces, self._trial_contact_forces)
        wp.copy(self._certified_positions, staged.particle_q)
        self._has_certified_positions = True
        self.last_diagnostics = diagnostic
        return diagnostic

    def _validate_runtime_parameters(self):
        """Reject invalid mutable settings before graph allocation/publication."""
        skin = positive_scalar(self.self_path_reuse_skin_m, "self path reuse skin", allow_zero=True)
        if skin+self.self_motion.min_gap > np.finfo(np.float32).max / 4:
            raise ValueError("self path reuse skin is too large for conservative bounds")
        for owner, name, minimum in ((self, "iterations", 1), (self, "max_depth", 0),
                (self, "seed_repair_iterations", 0), (self, "friction_iterations", 0), (self.energy, "linear_max_iterations", 1)):
            value = getattr(owner, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                    or value < minimum or value >= 2**31-1):
                raise ValueError(f"{name} must be an integer in [{minimum}, {2**31-2}]")
        positive_scalar(self.force_tolerance, "force tolerance")
        if self.friction_iterations and not self.device_control:
            raise ValueError("iterated friction requires GPU device control")
        if positive_scalar(self.min_jacobian, "minimum tet Jacobian") >= 1:
            raise ValueError("minimum tet Jacobian must be smaller than one")
        if positive_scalar(self.energy.linear_tolerance, "linear tolerance") >= 1:
            raise ValueError("linear tolerance must be below one")
        for name in ("device_control", "repair_volume_seed"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if self.require_convergence is not True:
            raise ValueError("Unified GPU IPC requires force-residual convergence")
        if self.contact is not self.energy.contact or self.contact is not self.backend.contact:
            raise ValueError("contact object cannot be replaced during a run; rebuild the solver")
        # Both feasibility floors and their constitutive barriers must agree.
        # All checks use host settings, with no GPU download or host convergence.
        self.energy.volume_min_jacobian = self.min_jacobian
        self.energy.validate_volume_barrier()
        self.energy.sync_contact_parameters()
        gap = self.energy.mesh.dmin
        if gap > np.finfo(np.float32).max / 4 or np.float32(gap) <= 0:
            raise ValueError("self-contact gap must be representable in float32")
        if self.self_motion.min_gap != gap:
            self.self_motion.min_gap = gap
            # The previous certificate only covered the previous clearance.
            # Recheck unchanged positions too, before constructing a new seed.
            self._has_certified_positions = False

    def _step_impl(self, *args, **kwargs):
        """Adapter validation belongs here, inside the per-call work report."""
        return self._device_step(*args, **kwargs) if self.device_control else super().step(*args, **kwargs)

    def _record_step_work(self, started, *, result=None, error=None, attempts=()):
        """Publish work independently of the last accepted physical state.

        An adapter rejecting a command before entering step uses the default
        empty attempts. Invalid mode values must neither be truth-tested nor
        leak non-serializable mutable objects into a retained report.
        """
        mode = self.device_control if isinstance(self.device_control, bool) else None
        attempts = tuple(attempts)
        self.last_step_work = dict(
            step_accepted=result is not None, error=error, wall_ms=1000*(perf_counter()-started),
            device_control=mode,
            # Intermediate accepted half steps can be discarded by a later
            # rejection; only a completed outer call publishes substeps.
            gpu_launches=len(attempts),
            failed_gpu_launches=sum(a['code'] != 0 for a in attempts),
            published_substeps=result.internal_substeps if result is not None else 0,
            attempts=attempts,
            **{name: sum(a[name] for a in attempts) if mode is True else None
               for name in ('newton_iterations', 'backtracks', 'hessians', 'pcg_iterations', 'capture_ms')})

    def step(self, *args, **kwargs):
        # Accepted-state diagnostics intentionally survive a rejected step.
        # Work diagnostics describe THIS call, including discarded substeps
        # and failed/capacity-retry GPU launches, using existing report data.
        started = perf_counter()
        self._attempt_work = []
        self._trace = []
        accepted, error = False, None
        try:
            self._validate_runtime_parameters()
            result = self._step_impl(*args, **kwargs)
            wp.copy(self.self_contact_forces, self._trial_self_contact_forces)
            self.last_unified_trace = tuple(self._trace)
            accepted = True
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            self._record_step_work(started, result=result if accepted else None,
                                   error=error, attempts=self._attempt_work)
