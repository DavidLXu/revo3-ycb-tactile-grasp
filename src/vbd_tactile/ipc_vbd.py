"""Versioned Newton adapter: implicit tetrahedral VBD with sphere IPC.

This isolated subclass uses Newton's finalized model/coloring/adjacency. It
replaces the particle iteration with energy-decreasing colored Newton blocks
that assemble IPC and lagged friction Hessians together with elasticity.
Unsupported model features/versions fail closed. No installed Newton files or
process-global solver methods are patched.
"""

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp
from .ipc_precision import position_kernel

from .ipc import points_and_center, positive_scalar
from .ipc_surface import boundary_triangles
from .ipc_warp import WarpSphereContact

SUPPORTED_NEWTON = "1.6.1"
SUPPORTED_WARP = "1.17.0"
# Keep the previously validated pair available for saved EA reproductions.
# Cross-pair mixtures are not covered by either environment validation.
SUPPORTED_RUNTIME_PAIRS = {(SUPPORTED_NEWTON, SUPPORTED_WARP), ("1.5.2", "1.16.0")}


class IPCStepRejected(RuntimeError):
    """A step needs a smaller physical interval or a larger iteration budget."""

    retryable = True

    def __init__(self, reason, *, retry_failure=None):
        self.reason = reason
        self.retry_failure = retry_failure
        message = reason
        if retry_failure is not None:
            message += f"; adaptive retry failed: {retry_failure}"
        super().__init__(message)


class IPCInfeasibleTarget(IPCStepRejected):
    """A prescribed target violates a constraint on immovable geometry."""

    retryable = False


class IPCInitialContactCapacityExceeded(IPCStepRejected):
    """The unchanged input contact set exceeds the configured storage limit.

    Reducing dt cannot change this set. Raising the limit can permit the same
    state to run; this is not a proof that the geometric target is infeasible.
    """

    retryable = False

    def __init__(self, required, maximum):
        self.required, self.maximum = int(required), int(maximum)
        super().__init__(f"Initial GPU contact capacity exceeded: {self.required} > "
                         f"{self.maximum}; no pairs discarded; increase max_contact_capacity")


@dataclass(frozen=True)
class ImplicitStepDiagnostics:
    internal_substeps: int
    iterations: int
    max_residual_N: float
    min_swept_gap_m: float
    min_tet_jacobian: float
    max_local_energy_increase_J: float
    converged: bool
    # Conservative surface bound; unified solvers include particle padding.
    min_surface_gap_m: float = float("inf")
    self_contact_checked: bool = False
    self_contact_candidate_pairs: int = 0


class SolverIPCVBD(newton.solvers.SolverVBD):
    """Implicit Euler solver for tetrahedra and one prescribed translating sphere.

    Normal/friction load frames are frozen only for friction. Normal barriers
    are re-evaluated inside every vertex block. Full physical step paths are
    checked after nonlinear solves; rejected steps retry as two half steps.
    Caller states are untouched if all attempts fail.
    """

    def __init__(
        self,
        model,
        contact,
        *,
        iterations=128,
        force_tolerance_N=1.0e-6,
        min_tet_jacobian=0.05,
        max_subdivision_depth=4,
        friction_coefficient=0.0,
        friction_velocity_epsilon_m_s=0.005,
        require_convergence=True,
        block_dim=32,
        self_contact_gap_m=1e-7,
        sphere_enabled=True,
        collision_triangles=None,
        rigid_groups=None,
    ):
        if (newton.__version__, wp.__version__) not in SUPPORTED_RUNTIME_PAIRS:
            raise RuntimeError(
                f"IPC adapter requires a validated Newton/Warp pair {sorted(SUPPORTED_RUNTIME_PAIRS)}; "
                f"found {newton.__version__}/{wp.__version__}"
            )
        # Import private Newton utilities only after checking the version.
        from . import ipc_surface_warp, ipc_vbd_kernels
        from .ipc_self_audit import (
            SurfaceMotionAudit,
            TetVolumeAudit,
            positions_changed,
        )

        self._kernels = ipc_vbd_kernels
        self._surface_kernels = ipc_surface_warp
        if (
            isinstance(iterations, bool)
            or int(iterations) != iterations
            or iterations < 1
        ):
            raise ValueError("iterations must be a positive integer")
        if (
            isinstance(max_subdivision_depth, bool)
            or int(max_subdivision_depth) != max_subdivision_depth
            or max_subdivision_depth < 0
        ):
            raise ValueError("subdivision depth must be a non-negative integer")
        self.force_tolerance = positive_scalar(force_tolerance_N, "force tolerance")
        self.min_jacobian = positive_scalar(min_tet_jacobian, "minimum tet Jacobian")
        if self.min_jacobian >= 1:
            raise ValueError("minimum tet Jacobian must be smaller than one")
        self.max_depth = int(max_subdivision_depth)
        self.friction_mu = positive_scalar(
            friction_coefficient, "friction coefficient", allow_zero=True
        )
        self.friction_epsilon = positive_scalar(
            friction_velocity_epsilon_m_s, "friction velocity epsilon"
        )
        self.require_convergence = bool(require_convergence)
        if isinstance(block_dim, bool) or block_dim not in (32, 64, 128, 256):
            raise ValueError("block_dim must be one of 32, 64, 128, 256")
        self.block_dim = int(block_dim)
        if not isinstance(sphere_enabled, bool):
            raise ValueError("sphere_enabled must be a bool")
        if not sphere_enabled and not getattr(self, "is_gpu_unified", False):
            raise ValueError("disabled sphere is supported only by unified GPU IPC")
        self.sphere_enabled = sphere_enabled
        faces = self._validate_model(model, sphere_enabled=sphere_enabled)
        if sphere_enabled and not np.isclose(
            float(model.shape_scale.numpy()[0, 0]),
            contact.sphere_radius_m,
            rtol=1.0e-6,
            atol=1.0e-9,
        ):
            raise ValueError("IPC radius must match the model sphere radius")
        super().__init__(
            model, iterations=int(iterations), particle_enable_self_contact=False,
            friction_epsilon=self.friction_epsilon,
            **({"rigid_compliant_alm": False} if newton.__version__ == "1.6.1" else {}),
        )
        self.contact = contact
        self.surface_collision_radius_m = contact.sphere_radius_m
        self.backend = WarpSphereContact(contact, model.particle_count, model.device)
        if not np.allclose(
            self.backend.radii.numpy(),
            model.particle_radius.numpy(),
            rtol=1.0e-6,
            atol=1.0e-9,
        ):
            raise ValueError("IPC particle radii must match the model particle radii")
        self.positions = wp.empty_like(model.particle_q)
        self.old = wp.empty_like(model.particle_q)
        self.predictor = wp.empty(
            model.particle_count, dtype=wp.vec3d, device=model.device
        )
        self.lagged_normals = wp.empty_like(self.predictor)
        self.lagged_loads = wp.empty(
            model.particle_count, dtype=wp.float64, device=model.device
        )
        self.surface_triangles = wp.array(faces if sphere_enabled else np.empty((0, 3), np.int32),
                                          dtype=int, device=model.device)
        self.diagnostic = wp.empty(6, dtype=wp.float64, device=model.device)
        self.contact_forces = wp.zeros_like(model.particle_q)
        self._trial_contact_forces = wp.empty_like(model.particle_q)
        self._staged = model.state()
        self.last_diagnostics = None
        self._control_invalid = wp.zeros(1, dtype=int, device=model.device)
        self._data_invalid = wp.zeros(1, dtype=int, device=model.device)
        self.self_motion = SurfaceMotionAudit(
            faces if collision_triangles is None else collision_triangles,
            model.particle_count,
            device=model.device,
            min_gap_m=self_contact_gap_m,
            rigid_groups=rigid_groups,
        )
        self.self_volume = TetVolumeAudit(
            model.tet_indices.numpy(), model.particle_count, device=model.device
        )
        self._positions_changed = positions_changed
        self._certified_positions = wp.empty_like(model.particle_q)
        self._has_certified_positions = False
        self._geometry_changed = wp.zeros(1, dtype=int, device=model.device)

    def _validate_initial_soft_geometry(self, positions):
        # Cache geometry by value, never by object/pointer: callers may modify a
        # previously accepted state in place. Model topology/rest data are fixed.
        if self._has_certified_positions:
            self._geometry_changed.zero_()
            wp.launch(
                position_kernel(self._positions_changed, positions),
                self.model.particle_count,
                inputs=[positions, self._certified_positions, self._geometry_changed],
                device=self.model.device,
            )
            if not self._geometry_changed.numpy()[0]:
                return
        volume = self.self_volume.check(positions)
        if not volume["valid"]:
            raise ValueError(
                "Initial soft volume overlaps or cannot be separated; "
                f"tet pairs: {volume['witnesses']}"
            )
        surface = self.self_motion.check(positions, positions)
        if not surface["certified"]:
            raise ValueError(
                "Initial soft surfaces intersect or lack self-contact clearance; "
                f"{surface['uncertified_pairs']} uncertified feature pairs"
            )
        wp.copy(self._certified_positions, positions)
        self._has_certified_positions = True

    def _validate_states(self, state_in, state_out):
        # Validate every copy before starting: Warp copies can fail after an
        # earlier field has already been published, or overwrite another field.
        fields = [
            "particle_q",
            "particle_qd",
            "particle_f",
            "body_q",
            "body_qd",
            "body_f",
        ]
        if self.model.joint_count:
            fields.extend(("joint_q", "joint_qd"))
        buffers = []
        for label, state in (("state_in", state_in), ("state_out", state_out)):
            for field in fields:
                value = getattr(state, field, None)
                reference = getattr(self._staged, field)
                if (
                    not isinstance(value, wp.array)
                    or value.shape != reference.shape
                    or value.dtype != reference.dtype
                    or value.device != reference.device
                    or not value.is_contiguous
                ):
                    raise ValueError(
                        f"{label}.{field} must match the model's shape, dtype, device "
                        "and contiguous storage"
                    )
                if value.size:
                    begin = value.ptr
                    end = begin + value.size * value.strides[0]
                    for other_field, other_begin, other_end in buffers:
                        # Exact aliases of the same field permit in-place
                        # stepping. Cross-field or partial aliases do not.
                        if (
                            begin < other_end
                            and other_begin < end
                            and (field, begin, end)
                            != (
                                other_field,
                                other_begin,
                                other_end,
                            )
                        ):
                            raise ValueError(
                                f"state buffers overlap: {field} and {other_field}"
                            )
                    buffers.append((field, begin, end))

    def _all_finite(self, arrays):
        self._data_invalid.zero_()
        for array in arrays:
            scalar = wp.float64 if array.dtype in (wp.vec3d, wp.float64) else wp.float32
            values = array.view(scalar).flatten()
            if values.size:
                wp.launch(
                    self._kernels.validate_finite_data_double
                    if scalar == wp.float64
                    else self._kernels.validate_finite_data,
                    values.size,
                    inputs=[values, self._data_invalid],
                    device=self.model.device,
                )
        return not self._data_invalid.numpy()[0]

    def state(self):
        """Create a Newton state with the precision required by this solver."""
        return self.model.state()

    @staticmethod
    def _validate_model(model, *, sphere_enabled=True):
        if not sphere_enabled and (model.body_count or model.shape_count or model.joint_count):
            raise ValueError("mesh IPC prescribes particle boundaries; Newton rigid bodies are unsupported")
        if sphere_enabled and (model.body_count != 1 or model.shape_count != 1):
            raise ValueError(
                "IPC adapter supports exactly one centered kinematic sphere"
            )
        if model.joint_count and (
            model.joint_count != 1
            or int(model.joint_type.numpy()[0]) != int(newton.JointType.FREE)
        ):
            raise ValueError(
                "the prescribed sphere supports only an unrestrained free joint"
            )
        for field in ("tet_activations", "tri_activations"):
            value = getattr(model, field, None)
            if value is not None and np.any(value.numpy() != 0):
                raise ValueError(
                    "active material activation is unsupported by the IPC adapter"
                )
        if sphere_enabled and (
            int(model.shape_type.numpy()[0]) != int(newton.GeoType.SPHERE)
            or int(model.shape_body.numpy()[0]) != 0
        ):
            raise ValueError("IPC adapter requires a sphere attached to body zero")
        if sphere_enabled and np.any(model.shape_transform.numpy()[0, :3] != 0):
            raise ValueError("sphere shape must be centered at its body origin")
        if model.particle_count == 0 or model.tet_count == 0:
            raise ValueError("IPC adapter requires a tetrahedral particle model")
        if model.spring_count:
            raise ValueError("springs are not supported by the IPC tet adapter")
        if model.tri_count and np.any(model.tri_materials.numpy()[:, :3] != 0):
            raise ValueError(
                "membrane materials are not supported by the IPC tet adapter"
            )
        if model.edge_count and np.any(model.edge_bending_properties.numpy() != 0):
            raise ValueError(
                "bending materials are not supported by the IPC tet adapter"
            )
        if model.body_count and np.any(
            (model.body_flags.numpy() & int(newton.BodyFlags.KINEMATIC)) == 0
        ):
            raise ValueError("only kinematic rigid bodies are supported")
        masses = model.particle_mass.numpy()
        if not np.all(np.isfinite(masses)) or np.any(masses < 0):
            raise ValueError("particle masses must be finite and non-negative")
        if np.any((masses > 0) & ((model.particle_flags.numpy() & 1) == 0)):
            raise ValueError("inactive particles with nonzero mass are unsupported")
        material = model.tet_materials.numpy()
        if (
            not np.all(np.isfinite(material))
            or np.any(material < 0)
            or np.any(material[:, 0] + material[:, 1] <= 0)
        ):
            raise ValueError(
                "tet materials must have positive total stiffness and non-negative damping"
            )
        poses = model.tet_poses.numpy().astype(np.float64)
        if not np.all(np.isfinite(poses)) or np.any(np.linalg.det(poses) <= 0):
            raise ValueError("tet rest poses must be positively oriented")
        indices = model.tet_indices.numpy()
        faces = boundary_triangles(indices)
        if np.any(indices >= model.particle_count):
            raise ValueError("tet indices must reference existing particles")
        if not len(faces):
            raise ValueError("tet model must have a nonempty boundary")
        # Validate topology before Newton builds GPU adjacency, which assumes
        # valid indices. Parallel per-color updates also need distinct colors.
        colors = model.particle_colors.numpy()[indices]
        if np.any(np.diff(np.sort(colors, axis=1), axis=1) == 0):
            raise ValueError(
                "tet vertices must have distinct colors; call builder.color()"
            )
        return faces

    def _reset_diagnostic(self):
        self.diagnostic.assign(
            np.array([0.0, 0.0, np.inf, np.inf, 0.0, np.inf], dtype=np.float64)
        )

    def _validate_paths(self, old, new, c0, c1, *, check_containment=False):
        self._reset_diagnostic()
        m = self.model
        wp.launch(
            position_kernel(self._kernels.validate_sphere_paths, old),
            m.particle_count,
            inputs=[
                old,
                new,
                self.backend.radii,
                self.contact.sphere_radius_m,
                wp.vec3d(*c0),
                wp.vec3d(*c1),
                self.diagnostic,
            ],
            device=m.device,
        )
        wp.launch(
            position_kernel(self._kernels.validate_tet_paths, old),
            m.tet_count,
            inputs=[old, new, m.tet_indices, m.tet_poses, self.diagnostic],
            device=m.device,
        )
        wp.launch(
            position_kernel(self._surface_kernels.validate_surface_paths, old),
            len(self.surface_triangles),
            inputs=[
                old,
                new,
                self.surface_triangles,
                wp.vec3d(*c0),
                wp.vec3d(*c1),
                self.surface_collision_radius_m,
                self.contact.min_gap_m,
                self.diagnostic,
            ],
            device=m.device,
        )
        if check_containment:
            wp.launch(
                position_kernel(
                    self._surface_kernels.validate_sphere_outside_tets, old
                ),
                m.tet_count,
                inputs=[old, m.tet_indices, wp.vec3d(*c0), self.diagnostic],
                device=m.device,
            )
        values = self.diagnostic.numpy()
        if (
            values[0]
            or values[2] <= self.contact.min_gap_m
            or values[3] <= self.min_jacobian
            or values[5] <= self.contact.min_gap_m
        ):
            raise IPCStepRejected(
                f"Unsafe step path: particle gap={values[2]:.3g} m, "
                f"surface gap lower bound={values[5]:.3g} m, min det(F)={values[3]:.3g}; "
                "surface overlap, containment or incomplete path certification"
            )
        return values

    def _arguments(self, dt, c0, c1):
        m, c = self.model, self.contact
        return [
            self.positions,
            self.old,
            self.predictor,
            m.particle_mass,
            self.backend.radii,
            m.tet_indices,
            m.tet_poses,
            m.tet_materials,
            self.particle_adjacency,
            dt,
            wp.vec3d(*c1),
            wp.vec3d(*c0),
            c.sphere_radius_m,
            c.activation_distance_m,
            c.barrier_stiffness,
            self.lagged_normals,
            self.lagged_loads,
            self.friction_mu,
            self.friction_epsilon * dt,
        ]

    def _attempt(self, state_in, state_out, dt, c0, c1):
        m, c = self.model, self.contact
        wp.copy(self.old, state_in.particle_q)
        wp.launch(
            self._kernels.initialize,
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
        seed = self._validate_paths(self.old, self.positions, c0, c1)
        min_gap, min_j = float(seed[2]), float(seed[3])
        min_surface_gap = float(seed[5])
        max_increase = 0.0
        residual = float("inf")
        args = self._arguments(dt, c0, c1)
        for iteration in range(1, self.iterations + 1):
            if iteration % 8 == 1:
                self._reset_diagnostic()
            for group in m.particle_color_groups:
                wp.launch(
                    self._kernels.solve_color,
                    len(group),
                    inputs=[
                        group,
                        *args,
                        c.min_gap_m,
                        self.min_jacobian,
                        24,
                        self.diagnostic,
                    ],
                    device=m.device,
                    block_dim=self.block_dim,
                )
            if iteration % 8 == 0 or iteration == self.iterations:
                wp.launch(
                    self._kernels.residual,
                    m.particle_count,
                    inputs=[*args, self.diagnostic],
                    device=m.device,
                    block_dim=self.block_dim,
                )
                diagnostic = self.diagnostic.numpy()
                if diagnostic[0]:
                    raise IPCStepRejected("Invalid nonlinear iterate")
                min_gap = min(min_gap, float(diagnostic[2]))
                min_j = min(min_j, float(diagnostic[3]))
                max_increase = max(max_increase, float(diagnostic[1]))
                residual = float(diagnostic[4])
                if residual <= self.force_tolerance:
                    break
        converged = residual <= self.force_tolerance
        if self.require_convergence and not converged:
            raise IPCStepRejected(
                f"IPC force residual {residual:.3g} N exceeds {self.force_tolerance:.3g} N after {iteration} iterations"
            )
        path = self._validate_paths(self.old, self.positions, c0, c1)
        self_path = self.self_motion.check(self.old, self.positions)
        if not self_path["certified"]:
            raise IPCStepRejected(
                "Unsafe soft/soft or self-contact path: "
                f"{self_path['uncertified_pairs']} uncertified feature pairs"
            )
        min_gap, min_j = min(min_gap, float(path[2])), min(min_j, float(path[3]))
        min_surface_gap = min(min_surface_gap, float(path[5]))
        wp.launch(
            self._kernels.finalize,
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
        if m.body_count:
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
        wp.launch(
            self._kernels.contact_response,
            m.particle_count,
            inputs=[
                self.positions,
                self.old,
                self.backend.radii,
                wp.vec3d(*c1),
                wp.vec3d(*c0),
                c.sphere_radius_m,
                c.activation_distance_m,
                c.barrier_stiffness,
                self.lagged_normals,
                self.lagged_loads,
                self.friction_mu,
                self.friction_epsilon * dt,
                self._trial_contact_forces,
            ],
            device=m.device,
        )
        return ImplicitStepDiagnostics(
            1,
            iteration,
            residual,
            min_gap,
            min_j,
            max_increase,
            converged,
            min_surface_gap,
            True,
            self_path["candidate_pairs"],
        )

    def _advance(self, state_in, state_out, dt, c0, c1, depth):
        try:
            return self._attempt(state_in, state_out, dt, c0, c1)
        except IPCStepRejected as initial:
            if not initial.retryable or depth >= self.max_depth:
                raise
            # Each retry has its own caller-visible buffers; no partial step is
            # published if a later half ultimately fails.
            middle = self.state()
            target = ((c0 + c1) * 0.5).astype(np.float32).astype(np.float64)
            try:
                first = self._advance(state_in, middle, dt * 0.5, c0, target, depth + 1)
                # External loads are applied over both physical half steps.
                wp.copy(middle.particle_f, state_in.particle_f)
                second = self._advance(
                    middle, state_out, dt * 0.5, target, c1, depth + 1
                )
            except IPCStepRejected as retry:
                if not retry.retryable:
                    raise
                # Keep the original collision diagnosis instead of hiding it
                # behind a later, smaller-step convergence/roundoff failure.
                raise IPCStepRejected(
                    initial.reason, retry_failure=retry.retry_failure or retry.reason
                ) from retry
            return ImplicitStepDiagnostics(
                first.internal_substeps + second.internal_substeps,
                first.iterations + second.iterations,
                max(first.max_residual_N, second.max_residual_N),
                min(first.min_swept_gap_m, second.min_swept_gap_m),
                min(first.min_tet_jacobian, second.min_tet_jacobian),
                max(
                    first.max_local_energy_increase_J,
                    second.max_local_energy_increase_J,
                ),
                first.converged and second.converged,
                min(first.min_surface_gap_m, second.min_surface_gap_m),
                first.self_contact_checked and second.self_contact_checked,
                first.self_contact_candidate_pairs
                + second.self_contact_candidate_pairs,
            )

    def step(
        self,
        state_in,
        state_out,
        control,
        contacts,
        dt,
        *,
        sphere_center,
        sphere_target,
    ):
        if contacts is not None:
            raise ValueError(
                "IPC adapter owns sphere contacts; Newton contact buffers are unsupported"
            )
        dt = positive_scalar(dt, "dt")
        self._validate_states(state_in, state_out)
        _, c0 = points_and_center(np.empty((0, 3)), sphere_center)
        _, c1 = points_and_center(np.empty((0, 3)), sphere_target)
        float_max = np.finfo(np.float32).max
        if np.any(np.abs(c0) > float_max) or np.any(np.abs(c1) > float_max):
            raise ValueError("sphere centers must be representable in float32")
        c0, c1 = (
            c0.astype(np.float32).astype(np.float64),
            c1.astype(np.float32).astype(np.float64),
        )
        pose, velocity = state_in.body_q.numpy()[0], state_in.body_qd.numpy()[0]
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(velocity)):
            raise ValueError("sphere pose and velocity must be finite")
        rotation = pose[3:].astype(np.float64)
        if not np.isclose(np.dot(rotation, rotation), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("sphere orientation must be a unit quaternion")
        with np.errstate(over="ignore", invalid="ignore"):
            prescribed_velocity = (c1 - c0) / dt
        if not np.all(np.isfinite(prescribed_velocity)) or np.any(
            np.abs(prescribed_velocity) > float_max
        ):
            raise ValueError("prescribed sphere velocity must be finite in float32")
        if np.any(velocity[3:] != 0):
            raise ValueError(
                "rotating sphere contact is unsupported; prescribe translation only"
            )
        if not np.array_equal(pose[:3], c0.astype(np.float32)):
            raise ValueError("sphere_center must match the input state's sphere pose")
        if not self._all_finite((state_in.particle_qd, state_in.particle_f)):
            raise ValueError("particle velocities and external forces must be finite")
        if control is not None:
            self._control_invalid.zero_()
            for field, count in (
                ("tet_activations", self.model.tet_count),
                ("tri_activations", self.model.tri_count),
            ):
                values = getattr(control, field, None)
                if values is not None:
                    if (
                        values.shape != (count,)
                        or values.dtype != wp.float32
                        or values.device != self.model.device
                    ):
                        raise ValueError(
                            "material activation arrays must match the model"
                        )
                    if count:
                        wp.launch(
                            self._kernels.validate_passive_control,
                            count,
                            inputs=[values, self._control_invalid],
                            device=self.model.device,
                        )
            if self._control_invalid.numpy()[0]:
                raise ValueError(
                    "active material activation is unsupported by the IPC adapter"
                )
        self.backend.evaluate(state_in.particle_q, c0)
        try:
            self._validate_paths(
                state_in.particle_q, state_in.particle_q, c0, c0, check_containment=True
            )
        except IPCStepRejected as exc:
            raise ValueError(
                "Initial state has invalid tet geometry or sphere/surface overlap"
            ) from exc
        self._validate_initial_soft_geometry(state_in.particle_q)
        # Stage output transactionally even when the caller uses the same state
        # object for both input and output.
        staged = self._staged
        diagnostic = self._advance(state_in, staged, dt, c0, c1, 0)
        # Keep generalized coordinates consistent with the prescribed rigid pose.
        # Otherwise a later eval_fk() silently restores the old sphere position.
        if self.model.joint_count:
            newton.eval_ik(self.model, staged, staged.joint_q, staged.joint_qd)
        published = [
            staged.particle_q,
            staged.particle_qd,
            staged.body_q,
            staged.body_qd,
            self._trial_contact_forces,
        ]
        if self.model.joint_count:
            published.extend((staged.joint_q, staged.joint_qd))
        if not self._all_finite(published):
            raise IPCStepRejected(
                "Nonfinite staged state or contact force; output unchanged"
            )
        wp.copy(state_out.particle_q, staged.particle_q)
        wp.copy(state_out.particle_qd, staged.particle_qd)
        if self.model.body_count:
            wp.copy(state_out.body_q, staged.body_q)
            wp.copy(state_out.body_qd, staged.body_qd)
        if self.model.joint_count:
            wp.copy(state_out.joint_q, staged.joint_q)
            wp.copy(state_out.joint_qd, staged.joint_qd)
        wp.copy(self.contact_forces, self._trial_contact_forces)
        wp.copy(self._certified_positions, staged.particle_q)
        self.last_diagnostics = diagnostic
        return diagnostic
