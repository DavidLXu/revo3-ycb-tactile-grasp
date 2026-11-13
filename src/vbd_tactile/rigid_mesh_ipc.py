"""Prescribed closed rigid meshes coupled to soft FEM in one GPU IPC solve.

Rigid collision vertices are Dirichlet unknowns, not elastic particles. Their
reactions are retained; no independent rigid dynamics or ABD is implied.
"""
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from .ipc import IPCSphereContact
from .ipc_precision import position_kernel
from .ipc_surface import boundary_triangles
from .ipc_self_audit import SurfaceMotionAudit, _bounds
from .ipc_vbd import ImplicitStepDiagnostics
from .mounted_ipc import SolverMountedIPCGPU, rigid_poses, midpoint, prescribe_backing
from . import rigid_mesh_gpu as k


def combine(a, b):
    return ImplicitStepDiagnostics(a.internal_substeps+b.internal_substeps,
        a.iterations+b.iterations, max(a.max_residual_N,b.max_residual_N),
        min(a.min_swept_gap_m,b.min_swept_gap_m), min(a.min_tet_jacobian,b.min_tet_jacobian),
        max(a.max_local_energy_increase_J,b.max_local_energy_increase_J),
        a.converged and b.converged, min(a.min_surface_gap_m,b.min_surface_gap_m),
        a.self_contact_checked and b.self_contact_checked,
        a.self_contact_candidate_pairs+b.self_contact_candidate_pairs)


class SolverRigidMeshIPCGPU(SolverMountedIPCGPU):
    def __init__(self, model, *, meshes, rigid_ranges, pin_ids, pin_owners, pin_local,
                 mount_poses, particle_owners, curved_path=False, **kwargs):
        if not isinstance(curved_path,bool):
            raise ValueError("curved_path must be a bool")
        self.curved_path = curved_path
        self.meshes, self.rigid_ranges = tuple(meshes), tuple(rigid_ranges)
        if not self.meshes or len(self.meshes) != len(self.rigid_ranges):
            raise ValueError("provide meshes and their appended particle ranges")
        mount_poses = rigid_poses(mount_poses)
        self.soft_mount_count = len(mount_poses)-len(meshes)
        if self.soft_mount_count < 1:
            raise ValueError("soft backings must precede rigid object poses")
        groups = np.full(model.particle_count, -1, np.int32)
        faces, ids = [], []
        for i, (mesh, bounds) in enumerate(zip(meshes, rigid_ranges)):
            if (bounds.step not in (None, 1) or bounds.start is None or bounds.stop is None
                    or bounds.start < 0 or bounds.stop > model.particle_count
                    or bounds.stop-bounds.start != len(mesh.vertices)
                    or np.any(groups[bounds] >= 0)):
                raise ValueError("invalid or overlapping rigid particle ranges")
            groups[bounds] = i
            ids.extend(range(bounds.start, bounds.stop))
            faces.append(mesh.triangles+bounds.start)
            if not np.all(np.asarray(particle_owners)[bounds] == self.soft_mount_count+i):
                raise ValueError("rigid vertices must be bound to their object pose")
        rigid_faces = np.concatenate(faces)
        if np.any(groups[model.tet_indices.numpy()] >= 0):
            raise ValueError("rigid collision vertices cannot belong to elastic tetrahedra")
        covered = np.unique(np.concatenate((model.tet_indices.numpy().ravel(), np.asarray(ids))))
        if len(covered) != model.particle_count:
            raise ValueError("every particle must belong to FEM or a rigid mesh")
        collision = np.concatenate((boundary_triangles(model.tet_indices.numpy()), rigid_faces))
        if kwargs.pop('sphere_enabled', False):
            raise ValueError("mesh solver does not enable the analytic sphere")
        if kwargs.get('device_control', True) is not True:
            raise ValueError("mesh IPC currently requires device_control=True")
        # The shared solver retains sphere configuration fields for compatibility;
        # sphere batches, initialization, validation and publication are disabled.
        contact = IPCSphereContact(.001, activation_distance_m=.0001,
                                   particle_radius_m=model.particle_radius.numpy())
        super().__init__(model, contact, pin_ids=pin_ids, pin_owners=pin_owners,
            pin_local=pin_local, mount_poses=mount_poses, particle_owners=particle_owners,
            sphere_enabled=False, collision_triangles=collision, rigid_groups=groups, **kwargs)
        self._rigid_faces = wp.array(rigid_faces, dtype=int, device=model.device)
        self._rigid_ids = wp.array(ids, dtype=int, device=model.device)
        self._rigid_offsets = wp.array(np.cumsum([0]+[len(f) for f in faces]), dtype=int, device=model.device)
        self._start_mount_device = wp.array(mount_poses, dtype=wp.mat44d, device=model.device)
        self._rigid_bounds = [wp.zeros(len(rigid_faces), dtype=wp.vec3, device=model.device) for _ in range(2)]
        self._rigid_invalid = wp.zeros(1, dtype=int, device=model.device)
        self._rigid_bvh = wp.Bvh(*self._rigid_bounds)
        self._seed_skin = wp.zeros(1, dtype=wp.float64, device=model.device)
        self._arc_errors = wp.zeros(len(mount_poses), dtype=wp.float64, device=model.device)
        from .rigid_curved_path import CurvedPathAudit
        self._curved_path_audit = CurvedPathAudit(self)
        self._mount_radii = np.array([np.linalg.norm(np.asarray(pin_local)[np.asarray(pin_owners)==i],axis=1).max(initial=0.) for i in range(len(mount_poses))])
        self.physical_path_gap = self.self_motion.min_gap
        self._physical_path_gap_device=wp.array([self.physical_path_gap],dtype=wp.float64,device=model.device)
        initial = self.state()
        self._refit_rigid(initial.particle_q, rebuild=True)
        admission = wp.zeros(1,dtype=int,device=model.device)
        wp.launch(k.rigid_surface_intersections,len(self._rigid_faces),
            [initial.particle_q,self.energy.mesh.groups,self._rigid_faces,self._rigid_faces,
             self._rigid_bvh.id,True,admission],device=model.device)
        if admission.numpy()[0]:
            raise ValueError("rigid mesh has intersecting nonincident triangles")
        # Do not hide malformed input behind same-body collision filtering.
        for mesh_faces in faces:
            audit = SurfaceMotionAudit(mesh_faces, model.particle_count, device=model.device, min_gap_m=1.e-12)
            if not audit.check(initial.particle_q, initial.particle_q)['certified']:
                raise ValueError("rigid mesh has intersecting or unresolved nonincident features")

    def state(self):
        state = super().state()
        if hasattr(self, '_pin_ids'):
            poses = wp.array(self.mount_poses, dtype=wp.mat44d, device=self.model.device)
            wp.launch(prescribe_backing, len(self._pin_ids),
                      [self._pin_ids,self._pin_owners,self._pin_local,poses,state.particle_q], device=self.model.device)
        return state

    def _refit_rigid(self, positions, *, rebuild=False):
        self._rigid_invalid.zero_()
        wp.launch(position_kernel(_bounds, positions), len(self._rigid_faces), [positions, positions, self._rigid_faces, 3, 0.,
                  *self._rigid_bounds, self._rigid_invalid], device=self.model.device)
        if rebuild:
            self._rigid_bvh.rebuild()
        else:
            self._rigid_bvh.refit()

    def _seed_boundary(self):
        self._refit_rigid(self.old)
        wp.launch(k.mesh_seed, self.model.particle_count, [self.old,self.model.particle_mass,
            self._particle_owners,self._mount_delta,self._rigid_faces,self._rigid_bvh.id,
            self._seed_skin,self.positions], device=self.model.device)
        wp.launch(prescribe_backing, len(self._pin_ids),
            [self._pin_ids,self._pin_owners,self._pin_local,self._mount_device,self.positions], device=self.model.device)

    def _extra_initial_geometry(self, positions, graph):
        graph.launch(k.rigid_surface_intersections,len(self._rigid_faces),
            [positions,self.energy.mesh.groups,self._rigid_faces,self.self_motion.faces,
             self.self_motion.face_bvh.id,False,graph.status])
        graph.launch(k.inside_closed_rigid, (self.model.particle_count,len(self.meshes)),
            [positions,self.energy.mesh.groups,self._rigid_faces,self._rigid_offsets,graph.status])
        graph.launch(k.rigid_inside_tets, len(self._rigid_ids),
            [positions,self._rigid_ids,self.self_volume.tets,self.self_volume.bvh.id,graph.status])

    def _extra_initial_bindings(self, positions, graph):
        graph.launch(k.check_bindings,len(self._pin_ids),
            [positions,self._pin_ids,self._pin_owners,self._pin_local,self._start_mount_device,graph.status])

    def _extra_physical_path(self, graph):
        graph.launch(k.curved_tet_paths,self.model.tet_count,
            [self.old,self.positions,self.model.tet_indices,self.model.tet_poses,
             self.model.particle_mass,self._particle_owners,self._arc_errors,self.diagnostic])

    def _check_physical_path(self, graph, c0, c1):
        if self.curved_path:
            self._curved_path_audit.check(graph)
        else:
            graph.paths(self.old,self.positions,c0,c1)
            self._extra_physical_path(graph)
            self.self_motion.check(self.old,self.positions,readback=False,
                                   gap_override=self._physical_path_gap_device)

    def _validate_runtime_parameters(self):
        super()._validate_runtime_parameters()
        if self.device_control is not True or self.sphere_enabled is not False:
            raise ValueError("prescribed mesh IPC requires GPU device control with analytic sphere disabled")
        if not isinstance(self.curved_path,bool):
            raise ValueError("curved_path must be a bool")
        checks = self._curved_path_audit.max_checks
        if isinstance(checks,bool) or not isinstance(checks,int) or not 1 <= checks <= 65536:
            raise ValueError("curved path certificate budget must be an integer in [1,65536]")

    def _advance_mount(self, state_in, state_out, dt, c0, c1, depth, b0, b1):
        rotations = Rotation.from_matrix(b1[:,:3,:3] @ b0[:,:3,:3].transpose(0,2,1))
        angles = rotations.magnitude()
        if self.curved_path:
            # Refine geometry only inside the GPU certificate. No fixed angle
            # threshold forces a different physical integration time grid.
            self._curved_path_audit.rotvec.assign(rotations.as_rotvec())
        # A rotated point deviates from its chord by at most R*(1-cos(theta/2)).
        # Inflating the physical CCD clearance by twice the maximum bound also
        # certifies the prescribed SLERP path, not merely its endpoint chord.
        sagitta = self._mount_radii*2*np.sin(angles/4)**2
        arc_guard = 2*float(sagitta.max(initial=0.))
        if not self.curved_path and arc_guard > self.self_motion.min_gap*.25*4096**2:
            raise ValueError("rotation needs over 4096 geometric intervals; split the command")
        if not self.curved_path and arc_guard > self.self_motion.min_gap*.25:
            mark = len(self._trace)
            middle = self.state()
            bm = midpoint(b0,b1)
            try:
                first = self._advance_mount(state_in,middle,dt*.5,c0,c1,depth,b0,bm)
                wp.copy(middle.particle_f,state_in.particle_f)
                second = self._advance_mount(middle,state_out,dt*.5,c0,c1,depth,bm,b1)
                return combine(first,second)
            except Exception:
                del self._trace[mark:]
                raise
        self.physical_path_gap = (float(np.nextafter(self.self_motion.min_gap+arc_guard,np.inf))
                                  if arc_guard and not self.curved_path else self.self_motion.min_gap)
        self._physical_path_gap_device.assign(np.array([self.physical_path_gap]))
        motion = np.linalg.norm(b1[:,:3,3]-b0[:,:3,3],axis=1)+2*self._mount_radii*np.sin(angles/2)
        self._seed_skin.assign(np.array([4*motion.max(initial=0.)+2*self.energy.mesh.dhat+self.energy.mesh.dmin]))
        self._start_mount_device.assign(b0)
        self._arc_errors.assign(sagitta)
        return super()._advance_mount(state_in,state_out,dt,c0,c1,depth,b0,b1)

    def step(self, state_in, state_out, control, contacts, dt, *, mount_target=None):
        # Reuse transactional publication/adaptive retry without a fictitious body.
        return super().step(state_in,state_out,control,contacts,dt,
                            sphere_center=np.zeros(3),sphere_target=np.zeros(3),mount_target=mount_target)
