"""Device-resident unified potential and preconditioned conjugate gradients.

NumPy is used only for immutable model setup. Evaluation, sparse element-block
assembly, matrix products, preconditioning and vector reductions run in Warp.
Only bounded counters and convergence diagnostics are copied to the host.
"""

import math
import numpy as np
import warp as wp
from .ipc_precision import position_kernel
from .ipc_gpu_preconditioner import diagonal_inverse
from warp.optim.linear import LinearOperator, cg
from . import ipc_gpu_kernels as k
from .ipc_gpu_contacts import MeshContactGPU, ContactBatchGPU
from . import ipc_gpu_operator as op
from . import ipc_gpu_hessian as hk
from . import ipc_gpu_contact_hessian as chk
from . import ipc_volume_barrier as vb
from .ipc_vbd import IPCStepRejected
from .ipc import positive_scalar


class UnifiedEnergyGPU:
    def __init__(
        self,
        model,
        contact,
        faces,
        *,
        collision_faces=None,
        rigid_groups=None,
        self_dhat,
        self_stiffness,
        self_min_gap,
        self_mu=0.0,
        sphere_mu=0.0,
        friction_epsilon=0.005,
        contact_capacity=4096,
        max_contact_capacity=262144,
        contact_candidate_skin=0.0,
        linear_tolerance=1e-5,
        linear_max_iterations=2000,
        material_hessian="spectral",
        material_model="stable_neo_hookean",
        contact_hessian="free",
        linear_preconditioner="jacobi",
        volume_min_jacobian=.05,
        volume_barrier_activation=None,
        volume_barrier_scale=1.,
    ):
        self.model, self.contact, self.device = model, contact, model.device
        self.n = model.particle_count
        self.volume_min_jacobian = float(volume_min_jacobian)
        self.volume_barrier_activation = None if volume_barrier_activation is None else float(volume_barrier_activation)
        self.volume_barrier_scale = float(volume_barrier_scale)
        self.validate_volume_barrier()
        self.material_hessian = material_hessian
        self.material_model = material_model
        self.linear_preconditioner = linear_preconditioner
        self.sphere_mu, self.epsilon = sphere_mu, friction_epsilon
        radii = contact.radii(self.n)
        if not np.all(radii == radii[0]):
            raise ValueError(
                "Unified GPU sphere contact requires uniform particle radii"
            )
        self.offset = contact.sphere_radius_m + float(radii[0])
        self._sphere_radius_m = contact.sphere_radius_m
        self._particle_radius_m = contact.particle_radius_m.copy()
        # Diverge at the same clearance enforced by feasibility checks. Passing
        # this shifted offset with dmin=0 reuses the exact linear-gap kernels.
        self.barrier_offset = self.offset + contact.min_gap_m
        self.mesh = MeshContactGPU(
            model.particle_q.numpy(),
            faces if collision_faces is None else collision_faces,
            rigid_groups=rigid_groups,
            device=self.device,
            dhat=self_dhat,
            stiffness=self_stiffness,
            min_gap=self_min_gap,
            friction_mu=self_mu,
            initial_capacity=contact_capacity,
            max_capacity=max_contact_capacity,
            candidate_skin=contact_candidate_skin,
        )
        self.tets = ContactBatchGPU(model.tet_count, self.device)
        self.tets.ids.assign(model.tet_indices.numpy())
        self.tets.count.assign(np.array([model.tet_count], np.int32))
        self.sphere = ContactBatchGPU(len(faces), self.device)
        self.sphere.ids.assign(
            np.column_stack((np.full(len(faces), -1), faces)).astype(np.int32)
        )
        tri = model.particle_q.numpy()[faces].astype(np.float64)
        self.sphere.areas.assign(np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)*.5)
        self.sphere.count.assign(np.array([len(faces)], np.int32))
        self.sphere_frame = ContactBatchGPU(len(faces), self.device).frame()
        self.sphere_frame.copy_candidates(self.sphere, len(faces))
        self.mesh_frame = ContactBatchGPU(contact_capacity, self.device).frame()
        self.gradient = wp.zeros(self.n, dtype=wp.vec3d, device=self.device)
        self.sphere_force = wp.zeros_like(self.gradient)
        self.self_force = wp.zeros_like(self.gradient)
        self.rhs = wp.zeros_like(self.gradient)
        self.direction = wp.zeros_like(self.gradient)
        self.mv_work = wp.zeros_like(self.gradient)
        self.applied = wp.zeros_like(self.gradient)
        self.diagonal = wp.zeros(self.n, dtype=wp.mat33d, device=self.device)
        self.inverse = wp.zeros_like(self.diagonal)
        self.dt_device = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.stats = wp.zeros(6, dtype=wp.float64, device=self.device)
        self.linear_stats = wp.zeros(5, dtype=wp.float64, device=self.device)
        self.matrix = op.FixedBlockMatrix(self.n, model.tet_indices.numpy(), faces, self.device)
        self.multilevel = None
        self.preconditioner = self.inverse
        if linear_preconditioner == "multilevel":
            from .ipc_gpu_multilevel import MultilevelSchwarz
            self.multilevel = MultilevelSchwarz(model.particle_q.numpy(), model.particle_mass.numpy(),
                model.tet_indices.numpy(), self.matrix, self.device)
            self.preconditioner = self.multilevel.operator
        self.linear_tolerance = float(linear_tolerance)
        self.linear_max_iterations = int(linear_max_iterations)
        self.operator = LinearOperator(
            (3 * self.n, 3 * self.n), wp.float64, self.device, self._matvec
        )
        self.fixed_operator = LinearOperator(
            (3 * self.n, 3 * self.n), wp.float64, self.device, self._fixed_matvec
        )
        self._cg = None
        self._cg_signature = None
        self._cg_graph = None
        self.last_linear = {}
        self.old_center = np.zeros(3)
        self.dt = 0.0
        self.surface_contact = None
        self._evaluation = None
        self._hessian_ready = False
        self._hessian_revision = -1
        self._evaluated_x = self._evaluated_old = None
        self.contact_hessian = contact_hessian

    def enable_surface_contact(self, **options):
        """Opt in before graph capture; immutable CPU setup, CUDA runtime only.

        Currently exposed through the monolithic ABD graph with block
        preconditioning. Existing self contacts and all CCD stay unchanged.
        """
        from .ipc_surface_gpu import SurfaceContactGPU
        if self.mesh.convergent_contact or self.mesh.aggregation is not None or self.mesh.aggregation_power or self.mesh.taccel_discrete:
            raise ValueError('surface integral cannot combine with other contact laws')
        self.surface_contact = SurfaceContactGPU(self.mesh._rest_reference,
            self.mesh.faces.numpy(), self.mesh.groups.numpy(), device=self.device,
            dhat=self.mesh.dhat, min_gap=self.mesh.dmin, stiffness=self.mesh.stiffness,
            mu=self.mesh.mu, **options)
        return self.surface_contact

    @property
    def material_model(self):
        return self._material_model

    @material_model.setter
    def material_model(self, value):
        if value not in ("stable_neo_hookean", "log_neo_hookean"):
            raise ValueError("material_model must be stable_neo_hookean or log_neo_hookean")
        if value != getattr(self, "_material_model", None):
            self._evaluation = None
            self._hessian_ready = False
            self._hessian_revision = -1
        self._material_model = value

    @property
    def material_value_kernel(self):
        if self.material_model == "log_neo_hookean":
            from . import ipc_log_material as log_material
            return log_material.tetrahedra
        return k.tetrahedra

    @property
    def material_free_hessian_kernel(self):
        if self.material_model == "log_neo_hookean":
            from . import ipc_log_material as log_material
            return log_material.hessian_spectral_free if self.material_hessian == "spectral" else log_material.hessian_free
        from . import abd_fixed_hessian as fixed
        return fixed.tetrahedra_spectral if self.material_hessian == "spectral" else fixed.tetrahedra

    @property
    def material_hessian(self):
        return self._material_hessian

    @material_hessian.setter
    def material_hessian(self, value):
        if value not in ("spectral", "jacobi"):
            raise ValueError("material_hessian must be spectral or jacobi")
        if value != getattr(self, "_material_hessian", None):
            self._hessian_ready = False
            self._hessian_revision = -1
        self._material_hessian = value

    @property
    def material_hessian_kernel(self):
        # Both host-controlled linearization and CUDA graph capture must use
        # the reported algorithm. The graph cache keys this kernel as well.
        if self.material_model == "log_neo_hookean":
            from . import ipc_log_material as log_material
            return log_material.hessian_spectral if self.material_hessian == "spectral" else log_material.hessian
        return hk.tetrahedra_spectral if self.material_hessian == "spectral" else hk.tetrahedra

    @property
    def linear_preconditioner(self):
        return self._linear_preconditioner

    @linear_preconditioner.setter
    def linear_preconditioner(self, value):
        if value not in ("jacobi", "multilevel"):
            raise ValueError("linear_preconditioner must be jacobi or multilevel")
        if value != getattr(self, "_linear_preconditioner", value):
            # The selected hierarchy owns device arrays and is retained by
            # nested CG captures. A name change alone cannot replace it.
            raise ValueError("linear_preconditioner cannot change during a run; rebuild the solver")
        self._linear_preconditioner = value

    @property
    def contact_hessian(self):
        return self._contact_hessian

    @contact_hessian.setter
    def contact_hessian(self, value):
        if value not in ("full", "free", "reduced", "constrained"):
            raise ValueError("contact_hessian must be full, free, reduced or constrained")
        if value != getattr(self, "_contact_hessian", None):
            self._hessian_ready = False
            self._hessian_revision = -1
        self._contact_hessian = value

    @property
    def normal_hessian_kernel(self):
        return {"full": hk.normals, "free": chk.normals_free,
                "reduced": chk.normals_reduced,"constrained":chk.normals_constrained}[self.contact_hessian]

    def normal_hessian(self, launch, x, batch, center, dhat, dmin, stiffness, offset):
        args = [x, batch.ids, batch.kinds, batch.areas, batch.eps, batch.count,
                center, dhat, dmin, stiffness, offset, batch.h]
        if self.contact_hessian != "full":
            args.append(self.model.particle_mass)
        launch(self.normal_hessian_kernel, batch.capacity, args)
        if batch is self.mesh.batch and self.mesh.aggregation is not None:
            self.mesh.aggregation.add_hessian(self.model.particle_mass)

    @property
    def volume_barrier_enabled(self):
        return self.volume_barrier_activation is not None and self.volume_barrier_scale > 0

    def sync_contact_parameters(self):
        """Validate host configuration and update the derived sphere barrier.

        Geometry is fixed for a solver's lifetime; scalar contact settings may
        change between steps. This performs no device readback or geometry solve.
        """
        c = self.contact
        if (c.sphere_radius_m != self._sphere_radius_m
                or not np.array_equal(c.particle_radius_m, self._particle_radius_m)):
            raise ValueError("contact radii cannot change during a run; rebuild the model and solver")
        gap = positive_scalar(c.min_gap_m, "min gap")
        activation = positive_scalar(c.activation_distance_m, "activation distance")
        positive_scalar(c.barrier_stiffness, "barrier stiffness")
        if gap >= activation:
            raise ValueError("min gap must be smaller than activation distance")
        positive_scalar(self.sphere_mu, "sphere friction", allow_zero=True)
        positive_scalar(self.epsilon, "friction velocity epsilon")
        self.mesh.validate_parameters()
        self.barrier_offset = self.offset + gap

    def validate_volume_barrier(self):
        if not math.isfinite(self.volume_barrier_scale) or self.volume_barrier_scale < 0:
            raise ValueError("volume barrier scale must be finite and nonnegative")
        if self.volume_barrier_activation is not None and not (
            0 < self.volume_min_jacobian < self.volume_barrier_activation < 1
        ):
            raise ValueError("volume barrier requires 0 < minimum Jacobian < activation < 1")

    def volume_barrier(self, x, *, value=False, hessian=False):
        if not self.volume_barrier_enabled:
            return
        args = [x, self.tets.ids, self.model.tet_poses, self.model.tet_materials,
                self.volume_min_jacobian, self.volume_barrier_activation, self.volume_barrier_scale]
        if value:
            self._launch(vb.evaluate, self.model.tet_count, [*args, self.gradient, self.stats])
        if hessian:
            self._launch(vb.add_hessian, self.model.tet_count, [*args, self.tets.h])

    @property
    def batches(self):
        return (
            self.tets,
            self.sphere,
            self.sphere_frame,
            self.mesh.batch,
            self.mesh_frame,
        )

    def _launch(self, kernel, n, args):
        if n:
            wp.launch(
                position_kernel(kernel, args[0]), n, inputs=args, device=self.device,
                block_dim=32 if kernel in (hk.normals, chk.normals_free, chk.normals_reduced, chk.normals_constrained) else 256,
            )

    def begin_step(self, old, dt, center):
        if self.surface_contact is not None:
            raise ValueError("surface contact currently requires the monolithic ABD device graph")
        self.sync_contact_parameters()
        self._evaluation = None
        self._hessian_ready = False
        self.dt = float(dt)
        self.dt_device.assign(np.array([dt], dtype=np.float64))
        self.old_center = np.asarray(center, dtype=float).copy()
        count = self.mesh.build(old)
        if self.mesh_frame.capacity < count:
            self.mesh_frame = ContactBatchGPU(
                self.mesh.batch.capacity, self.device
            ).frame()
        self.mesh_frame.copy_candidates(self.mesh.batch, count)
        for batch, dhat, dmin, stiffness, offset, mu in (
            (
                self.sphere_frame,
                self.contact.activation_distance_m,
                0.0,
                self.contact.barrier_stiffness,
                self.barrier_offset,
                self.sphere_mu,
            ),
            (
                self.mesh_frame,
                self.mesh.dhat,
                self.mesh.dmin,
                self.mesh.stiffness,
                0.0,
                self.mesh.mu,
            ),
        ):
            self._launch(
                k.freeze_contact,
                batch.capacity,
                [
                    old,
                    batch.ids,
                    batch.kinds,
                    batch.areas,
                    batch.eps,
                    batch.count,
                    wp.vec3d(*center),
                    dhat,
                    dmin,
                    stiffness,
                    offset,
                    mu,
                    batch.weights,
                    batch.normals,
                    batch.loads,
                ],
            )

        from .taccel_contact import adjust_friction
        adjust_friction(self.mesh, old, self.mesh_frame, self._launch)

    def evaluate(self, x, old, predictor, center, *, hessian=True):
        if self.surface_contact is not None:
            raise ValueError("surface contact currently requires the monolithic ABD device graph")
        self._evaluation = None
        self._hessian_ready = False
        self.stats.zero_()
        self.sphere_force.zero_()
        self.self_force.zero_()
        self._launch(
            k.inertia,
            self.n,
            [
                x,
                predictor,
                self.model.particle_mass,
                self.dt,
                self.gradient,
                self.stats,
            ],
        )
        self._launch(
            self.material_value_kernel,
            self.model.tet_count,
            [
                x,
                old,
                self.tets.ids,
                self.model.tet_poses,
                self.model.tet_materials,
                self.dt,
                hessian and self.material_hessian == "jacobi",
                self.gradient,
                self.stats,
                self.tets.h,
            ],
        )
        if hessian and self.material_hessian == "spectral":
            self._launch(self.material_hessian_kernel, self.model.tet_count,
                [x, self.tets.ids, self.model.tet_poses, self.model.tet_materials, self.dt, self.tets.h])
        self.volume_barrier(x, value=True, hessian=hessian)
        count = self.mesh.build(x)
        for batch, dhat, dmin, stiffness, offset, force in (
            (
                self.sphere,
                self.contact.activation_distance_m,
                0.0,
                self.contact.barrier_stiffness,
                self.barrier_offset,
                self.sphere_force,
            ),
            (
                self.mesh.batch,
                self.mesh.dhat,
                self.mesh.dmin,
                self.mesh.stiffness,
                0.0,
                self.self_force,
            ),
        ):
            self._launch(
                k.normals,
                batch.capacity,
                [
                    x,
                    batch.ids,
                    batch.kinds,
                    batch.areas,
                    batch.eps,
                    batch.count,
                    wp.vec3d(*center),
                    dhat,
                    dmin,
                    stiffness,
                    offset,
                    hessian and self.contact_hessian == "full",
                    self.gradient,
                    force,
                    self.stats,
                    batch.h,
                ],
            )
            if hessian and self.contact_hessian != "full":
                self.normal_hessian(self._launch, x, batch, wp.vec3d(*center),
                                    dhat, dmin, stiffness, offset)
            elif hessian and batch is self.mesh.batch and self.mesh.aggregation is not None:
                self.mesh.aggregation.add_hessian(self.model.particle_mass)
        if self.mesh.aggregation is not None:self.mesh.aggregation.correct_energy(self.stats)
        for batch, force, constant in (
            (self.sphere_frame, self.sphere_force, False),
            (self.mesh_frame, self.self_force, True),
        ):
            self._launch(
                k.friction,
                batch.capacity,
                [
                    x,
                    old,
                    batch.ids,
                    batch.count,
                    wp.vec3d(*center),
                    wp.vec3d(*self.old_center),
                    batch.weights,
                    batch.normals,
                    batch.loads,
                    self.epsilon * self.dt,
                    constant,
                    hessian,
                    self.gradient,
                    force,
                    self.stats,
                    batch.h,
                ],
            )
        self._launch(
            k.residual,
            self.n,
            [self.gradient, self.model.particle_mass, self.stats, self.rhs],
        )
        values = self.stats.numpy()
        if values[2] or not np.all(np.isfinite(values[:2])):
            raise IPCStepRejected("Nonfinite GPU unified objective")
        if hessian:
            self._assemble_matrix()
        else:
            # Keep immutable device snapshots, not just caller buffer pointers:
            # line-search trials overwrite x in place. No bulk host download.
            if self._evaluated_x is None or self._evaluated_x.dtype != x.dtype:
                self._evaluated_x = wp.empty_like(x)
                self._evaluated_old = wp.empty_like(old)
            wp.copy(self._evaluated_x, x)
            wp.copy(self._evaluated_old, old)
            self._evaluation = (self.mesh.revision, wp.vec3d(*center))
        return float(values[0]), float(values[1]), count

    def _assemble_matrix(self, *, compact_dynamic=True, packed_product=True):
        batches=self.batches[1:] if compact_dynamic else (self.sphere,self.sphere_frame)
        for b in batches:
            b.active_count.zero_()
            wp.launch(op.compact_nonzero_free, b.capacity,
                [b.h, b.ids, self.model.particle_mass, b.count, b.active_indices, b.active_count], device=self.device)
        self.matrix.assemble(self.model.particle_mass, self.dt_device,
                             self.tets, self.sphere, self.sphere_frame, packed_product=packed_product)
        self._hessian_revision = self.mesh.revision
        self._hessian_ready = True

    def linearize(self):
        """Build H at the last successful value evaluation, retaining its RHS.

        Only the solver's immediately accepted evaluation may be reused. A new
        step, another evaluation (including a failed one), or an independent
        contact build invalidates it. Physical material/contact parameters must
        remain unchanged between evaluation and linearization. Changing the
        Hessian algorithm invalidates only the cached matrix, not the evaluated
        potential or gradient of the same constitutive model.
        """
        if self._evaluation is None or self._evaluation[0] != self.mesh.revision:
            self._hessian_ready = False
            raise IPCStepRejected("No current evaluated configuration for GPU Hessian")
        if self._hessian_ready:
            return
        x, old = self._evaluated_x, self._evaluated_old
        center = self._evaluation[1]
        self._launch(self.material_hessian_kernel, self.model.tet_count,
            [x, self.tets.ids, self.model.tet_poses, self.model.tet_materials, self.dt, self.tets.h])
        self.volume_barrier(x, hessian=True)
        for batch, dhat, dmin, stiffness, offset in (
            (self.sphere, self.contact.activation_distance_m, 0., self.contact.barrier_stiffness, self.barrier_offset),
            (self.mesh.batch, self.mesh.dhat, self.mesh.dmin, self.mesh.stiffness, 0.),
        ):
            self.normal_hessian(self._launch, x, batch, center, dhat, dmin, stiffness, offset)
        for batch in (self.sphere_frame, self.mesh_frame):
            self._launch(hk.friction, batch.capacity,
                [x, old, batch.ids, batch.count, center, wp.vec3d(*self.old_center),
                 batch.weights, batch.normals, batch.loads, self.epsilon*self.dt, batch.h])
        self._assemble_matrix()

    def _matvec(self, x, y, z, alpha, beta):
        self._product(x, y, z, alpha, beta, dynamic=True)

    def _fixed_matvec(self, x, y, z, alpha, beta):
        """Use only when both dynamic self-contact matrix lists are empty."""
        self._product(x, y, z, alpha, beta, dynamic=False)

    def _product(self, x, y, z, alpha, beta, *, dynamic):
        # PCG's hot path is z=A*x with disjoint x/z storage. Accumulate there
        # directly instead of writing a work vector and copying it in another
        # kernel at every linear iteration. Keep general axpby/alias semantics.
        disjoint = (x.ptr + x.shape[0]*x.strides[0] <= z.ptr or
                    z.ptr + z.shape[0]*z.strides[0] <= x.ptr)
        direct = alpha == 1.0 and beta == 0.0 and disjoint
        out = z if direct else self.mv_work
        aggregation = self.mesh.aggregation if dynamic else None
        if aggregation is not None:
            aggregation.prepare_product(x, self.model.particle_mass)
        self.matrix.product(x, out, packed=True)
        for b in ((self.mesh.batch, self.mesh_frame) if dynamic else ()):
            self._launch(
                op.active_product_rows,
                (b.capacity, 12),
                [x, b.ids, b.active_indices, b.active_count,
                 b.h.view(wp.float64).reshape((b.capacity, 12, 12)), self.model.particle_mass,
                 out.view(wp.float64).reshape((self.n, 3))],
            )
        if aggregation is not None:
            aggregation.add_product(out, 1., self.model.particle_mass)
        if not direct:
            self._launch(k.axpby, self.n, [out, y, alpha, beta, z])

    def solve(self):
        if not self._hessian_ready or self._hessian_revision != self.mesh.revision:
            raise IPCStepRejected("GPU Hessian is not ready; linearize the evaluated state first")
        self.matrix.diagonal(self.diagonal)
        for b in (self.mesh.batch, self.mesh_frame):
            self._launch(
                op.active_diagonal,
                b.capacity,
                [b.ids, b.active_indices, b.active_count, b.h, self.model.particle_mass, self.diagonal],
            )
        self._launch(
            diagonal_inverse, self.n, [self.diagonal, self.inverse, self.stats]
        )
        if self.multilevel is not None:
            self.multilevel.build(self.matrix, (self.mesh.batch, self.mesh_frame), self.stats)
        if self.stats.numpy()[2]:
            raise IPCStepRejected("GPU block preconditioner is invalid")
        # CG captures both array bindings and termination settings. Rebuild
        # when contacts resize or callers tune the linear solve between steps.
        signature = (
            id(self.mesh.aggregation),
            self.linear_tolerance,
            self.linear_max_iterations,
            tuple((b.ids.ptr, b.active_indices.ptr, b.active_count.ptr, b.h.ptr, b.capacity) for b in self.batches),
        )
        if self._cg is None or signature != self._cg_signature:
            self._cg = cg(
                self.operator,
                self.rhs,
                self.direction,
                M=self.preconditioner,
                tol=self.linear_tolerance,
                atol=1e-10,
                maxiter=self.linear_max_iterations,
                check_every=0,
                use_cuda_graph=True,
                run=False,
            )
            # Load the conditional-loop kernels before capture. The captured
            # graph then reuses buffers and live matrix values across solves.
            self.direction.zero_()
            self._cg()
            with wp.ScopedCapture(device=self.device) as capture:
                self.direction.zero_()
                self._cg_result = self._cg()
            self._cg_graph = capture.graph
            self._cg_signature = signature
        wp.capture_launch(self._cg_graph)
        # Recompute the true residual with a fresh product, never trust only the
        # recursive PCG residual or accept an exhausted linear iteration budget.
        self._matvec(self.direction, self.direction, self.applied, 1.0, 0.0)
        self.linear_stats.zero_()
        self._launch(
            k.check_linear,
            self.n,
            [self.rhs, self.applied, self.direction, self.linear_stats],
        )
        wp.launch(op.pack_cg_status, 1, [self._cg_result[0], self._cg_result[1], self.linear_stats], device=self.device)
        r2, b2, slope, iterations, reported = self.linear_stats.numpy()
        tolerance = max(1e-10, self.linear_tolerance * math.sqrt(max(float(b2), 0.0)))
        actual = math.sqrt(max(float(r2), 0.0))
        if (
            not np.all(np.isfinite([r2, b2, slope, reported]))
            or actual > tolerance * 1.05
            or slope >= 0
        ):
            raise IPCStepRejected(
                f"GPU PCG rejected: true residual {actual:.6g}, tolerance {tolerance:.6g}, slope {slope:.6g}"
            )
        self.last_linear = {
            "iterations": int(iterations),
            "true_residual_N": actual,
            "tolerance_N": tolerance,
        }
        return self.direction
