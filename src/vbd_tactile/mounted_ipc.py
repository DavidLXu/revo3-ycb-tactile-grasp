"""Moving rigid backings for the existing world-coordinate unified GPU IPC.

Only backing poses and the analytic sphere are prescribed. This is one-way
kinematic coupling, not a coupled rigid articulation/contact solver.
"""
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation, Slerp

from .ipc_unified_gpu import SolverUnifiedIPCGPU
from .ipc_vbd import IPCStepRejected, ImplicitStepDiagnostics
from .ipc_primitives_warp import _weights, _delta


def rigid_poses(value):
    poses = np.asarray(value, dtype=np.float64)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses)
            or not np.isfinite(poses).all()
            or not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or not np.allclose(poses[:, :3, :3] @ poses[:, :3, :3].transpose(0, 2, 1),
                               np.eye(3), atol=1e-9, rtol=0)
            or not np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-9, rtol=0)):
        raise ValueError("backing poses must be finite proper rigid 4x4 transforms")
    return poses.copy()


def midpoint(a, b):
    out = (a+b)*.5
    for i in range(len(out)):
        out[i, :3, :3] = Slerp([0, 1], Rotation.from_matrix([a[i, :3, :3], b[i, :3, :3]]))(.5).as_matrix()
    return out


@wp.kernel
def prescribe_backing(ids: wp.array(dtype=int), owners: wp.array(dtype=int),
                      local: wp.array(dtype=wp.vec3d), poses: wp.array(dtype=wp.mat44d),
                      positions: wp.array(dtype=wp.vec3d)):
    j = wp.tid()
    p = poses[owners[j]] * wp.vec4d(local[j][0], local[j][1], local[j][2], 1.0)
    positions[ids[j]] = wp.vec3d(p[0], p[1], p[2])


@wp.func
def transform_point(matrix: wp.mat44d, point: wp.vec3d):
    p = matrix * wp.vec4d(point[0], point[1], point[2], 1.0)
    return wp.vec3d(p[0], p[1], p[2])


@wp.kernel
def mount_face_seed(old: wp.array(dtype=wp.vec3d), faces: wp.array2d(dtype=int),
                    owners: wp.array(dtype=int), delta: wp.array(dtype=wp.mat44d),
                    sphere: wp.array(dtype=wp.vec3d), offset: wp.float64,
                    dhat: wp.float64, min_gap: wp.float64, weights: wp.array(dtype=wp.float64)):
    f = wp.tid()
    a, b, c = old[faces[f, 0]], old[faces[f, 1]], old[faces[f, 2]]
    old_center = sphere[0]
    motion = sphere[1]-transform_point(delta[owners[faces[f, 0]]], old_center)
    bary = _weights(old_center, a, b, c, 0)
    gap = wp.length(_delta(bary, old_center, a, b, c))-offset
    inner = min_gap+wp.float64(2)*wp.length(motion)
    weight = wp.clamp((inner+wp.float64(2)*dhat-gap)/(wp.float64(2)*dhat), wp.float64(0), wp.float64(1))
    for j in range(3):
        wp.atomic_max(weights, faces[f, j], weight)


@wp.kernel
def mount_particle_seed(old: wp.array(dtype=wp.vec3d), mass: wp.array(dtype=float),
                        owners: wp.array(dtype=int), delta: wp.array(dtype=wp.mat44d),
                        sphere: wp.array(dtype=wp.vec3d), offset: wp.float64,
                        dhat: wp.float64, min_gap: wp.float64, weights: wp.array(dtype=wp.float64),
                        positions: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if mass[i] > 0.0:
        matrix = delta[owners[i]]
        motion = sphere[1]-transform_point(matrix, sphere[0])
        gap = wp.length(old[i]-sphere[0])-offset
        inner = min_gap+wp.float64(2)*wp.length(motion)
        weight = wp.clamp((inner+wp.float64(2)*dhat-gap)/(wp.float64(2)*dhat), wp.float64(0), wp.float64(1))
        positions[i] = transform_point(matrix, old[i])+wp.max(weight, weights[i])*motion


class SolverMountedIPCGPU(SolverUnifiedIPCGPU):
    """World-space inertia and GPU constraints, with synchronized retry motion.

    Mass-zero nodes are transformed exactly to the requested link/sensor pose.
    Free nodes keep their inertial predictor in WORLD coordinates. Thus a moving
    base accelerates/deforms the gel; it does not merely move its rendering.
    Recursive retries bisect BOTH the sphere path and every backing pose.
    """

    seed_includes_faces = True

    def __init__(self, model, contact, *, pin_ids, pin_owners, pin_local, mount_poses,
                 particle_owners=None, seed_subdivision_depth=None, **kwargs):
        if kwargs.get("state_precision", "float64") != "float64":
            raise ValueError("moving backing currently requires float64 state")
        self.mount_poses = rigid_poses(mount_poses)
        ids, owners, local = np.asarray(pin_ids), np.asarray(pin_owners), np.asarray(pin_local, float)
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids)
                or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= model.particle_count
                or owners.shape != ids.shape or not np.issubdtype(owners.dtype, np.integer)
                or owners.min() < 0 or owners.max() >= len(self.mount_poses)
                or local.shape != (len(ids), 3) or not np.isfinite(local).all()):
            raise ValueError("invalid moving backing node bindings")
        masses = model.particle_mass.numpy()
        if not np.array_equal(np.sort(ids), np.flatnonzero(masses == 0)):
            raise ValueError("backing bindings must cover exactly all mass-zero nodes")
        initial = np.einsum('nij,nj->ni', self.mount_poses[owners, :3, :3], local)+self.mount_poses[owners, :3, 3]
        if not np.allclose(initial, model.particle_q.numpy()[ids], atol=5e-8, rtol=0):
            raise ValueError("backing rest bindings do not match the Newton mesh")
        if particle_owners is None and len(self.mount_poses) == 1:
            particle_owners = np.zeros(model.particle_count, dtype=int)
        particle_owners = np.asarray(particle_owners)
        if (particle_owners.shape != (model.particle_count,) or not np.issubdtype(particle_owners.dtype, np.integer)
                or particle_owners.min() < 0 or particle_owners.max() >= len(self.mount_poses)
                or not np.array_equal(particle_owners[ids], owners)):
            raise ValueError("every particle must identify its backing")
        tet_owners = particle_owners[model.tet_indices.numpy()]
        if not (tet_owners == tet_owners[:, :1]).all():
            raise ValueError("a tetrahedron cannot span separately prescribed backings")
        super().__init__(model, contact, **kwargs)
        self.seed_subdivision_depth = seed_subdivision_depth
        self._validate_seed_subdivision_depth()
        self._pin_ids = wp.array(ids, dtype=int, device=model.device)
        self._pin_owners = wp.array(owners, dtype=int, device=model.device)
        self._pin_local = wp.array(local, dtype=wp.vec3d, device=model.device)
        self._mount_device = wp.array(self.mount_poses, dtype=wp.mat44d, device=model.device)
        self._particle_owners = wp.array(particle_owners, dtype=int, device=model.device)
        self._mount_delta = wp.array(np.tile(np.eye(4), (len(self.mount_poses), 1, 1)), dtype=wp.mat44d, device=model.device)
        self._mount_sphere = wp.zeros(2, dtype=wp.vec3d, device=model.device)
        self._mount_requested = self.mount_poses.copy()

    def _seed_boundary(self):
        # Move the INITIAL GUESS with each rigid backing and relative sphere
        # motion. The world-space inertia predictor is deliberately untouched.
        # This avoids inverting skinny CAD tets by moving only their fixed end.
        self._seed_face_weights.zero_()
        args = [self._particle_owners, self._mount_delta, self._mount_sphere,
                self.energy.offset, self.contact.activation_distance_m,
                self.contact.min_gap_m, self._seed_face_weights]
        wp.launch(mount_face_seed, len(self.surface_triangles), [self.old, self.surface_triangles, *args], device=self.model.device)
        wp.launch(mount_particle_seed, self.model.particle_count,
                  [self.old, self.model.particle_mass, *args, self.positions], device=self.model.device)
        wp.launch(prescribe_backing, len(self._pin_ids),
                  [self._pin_ids, self._pin_owners, self._pin_local, self._mount_device, self.positions],
                  device=self.model.device)

    def _advance(self, state_in, state_out, dt, c0, c1, depth):
        return self._advance_mount(state_in, state_out, dt, c0, c1, depth,
                                   self.mount_poses, self._mount_requested)

    def _validate_seed_subdivision_depth(self):
        value = self.seed_subdivision_depth
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**31-1):
            raise ValueError("seed subdivision depth must be None or a nonnegative integer")

    def _validate_runtime_parameters(self):
        super()._validate_runtime_parameters()
        self._validate_seed_subdivision_depth()

    def _may_refine_seed(self, depth, attempt_mark):
        """Use the existing final GPU report; no extra geometry readback.

        A slightly infeasible initial guess is not a proof of an infeasible
        commanded endpoint. Reserve extra time refinement for that case only.
        Linear/nonlinear convergence, CCD and backing failures keep their
        ordinary budget, and every smaller attempt must pass all original gates.
        """
        limit = self.seed_subdivision_depth
        if (not self.device_control or limit is None or depth >= limit
                or len(self._attempt_work) <= attempt_mark):
            return False
        last = self._attempt_work[-1]
        path = last.get("path_diagnostics") or {}
        jacobian = path.get("min_tet_jacobian", -np.inf)
        return (last.get("code") == 1 and path.get("failed_checks") == ["tet_volume"]
                and .9*self.min_jacobian < jacobian <= self.min_jacobian)

    def _advance_mount(self, state_in, state_out, dt, c0, c1, depth, b0, b1):
        mark = len(self._trace)
        attempt_mark = len(self._attempt_work)
        self._mount_device.assign(b1)
        self._mount_delta.assign(b1 @ np.linalg.inv(b0))
        self._mount_sphere.assign(np.array([c0, c1]))
        try:
            return self._attempt(state_in, state_out, dt, c0, c1)
        except IPCStepRejected as initial:
            del self._trace[mark:]
            if not initial.retryable or (depth >= self.max_depth and not self._may_refine_seed(depth, attempt_mark)):
                raise
            middle = self.state()
            target = ((c0+c1)*.5).astype(np.float32).astype(float)
            bm = midpoint(b0, b1)
            try:
                first = self._advance_mount(state_in, middle, dt*.5, c0, target, depth+1, b0, bm)
                wp.copy(middle.particle_f, state_in.particle_f)
                second = self._advance_mount(middle, state_out, dt*.5, target, c1, depth+1, bm, b1)
            except IPCStepRejected as retry:
                del self._trace[mark:]
                if not retry.retryable:
                    raise
                raise IPCStepRejected(initial.reason, retry_failure=retry.retry_failure or retry.reason) from retry
            return ImplicitStepDiagnostics(
                first.internal_substeps+second.internal_substeps,
                first.iterations+second.iterations,
                max(first.max_residual_N, second.max_residual_N),
                min(first.min_swept_gap_m, second.min_swept_gap_m),
                min(first.min_tet_jacobian, second.min_tet_jacobian),
                max(first.max_local_energy_increase_J, second.max_local_energy_increase_J),
                first.converged and second.converged,
                min(first.min_surface_gap_m, second.min_surface_gap_m),
                first.self_contact_checked and second.self_contact_checked,
                first.self_contact_candidate_pairs+second.self_contact_candidate_pairs)

    def _step_impl(self, *args, mount_target=None, **kwargs):
        target = self.mount_poses if mount_target is None else rigid_poses(mount_target)
        if target.shape != self.mount_poses.shape:
            raise ValueError("backing count cannot change during a run")
        self._mount_requested = target.copy()
        try:
            result = super()._step_impl(*args, **kwargs)
        except Exception:
            self._mount_device.assign(self.mount_poses)
            self._mount_requested = self.mount_poses.copy()
            raise
        self.mount_poses = target.copy()
        return result
