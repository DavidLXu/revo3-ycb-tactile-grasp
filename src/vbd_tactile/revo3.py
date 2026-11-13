"""Actual hand_v11 fingertip volumes mounted to their five DIP links.

All selected gels share one implicit GPU IPC system, including soft/soft IPC.
The articulation supplies prescribed link poses. External contact supports
the analytic sphere, prescribed closed triangle meshes, and optional free
six-DOF mesh bodies through the partitioned implicit coupling driver.
"""
import json
from pathlib import Path
from time import perf_counter

import newton
import numpy as np
import warp as wp

from .ipc import IPCSphereContact, positive_scalar
from .mounted_ipc import SolverMountedIPCGPU, rigid_poses
from .curved_markers import CurvedMarkerLayer, lift_to_coating
from .markers import PinholeCamera
from .revo3_material import DEFAULT_STIFFNESS_SCALE

FINGERS = ("thumb", "index", "middle", "ring", "little")


@wp.kernel
def contact_resultants(owners: wp.array(dtype=int), sphere: wp.array(dtype=wp.vec3),
                       soft: wp.array(dtype=wp.vec3), result: wp.array2d(dtype=wp.vec3d)):
    i = wp.tid()
    wp.atomic_add(result, 0, owners[i], wp.vec3d(sphere[i]))
    wp.atomic_add(result, 1, owners[i], wp.vec3d(soft[i]))


class Revo3TactileHand:
    rigid_meshes = ()
    rigid_dynamics = None

    def __init__(self, assets, *, device="cuda:0", fingers=FINGERS,
                 probe_finger="index", image_size=256, gravity=(0, 0, -9.81),
                 probe_radius_m=.006, probe_xy_m=(0., 0.), volume_barrier_activation=.1,
                 contact_candidate_skin_m=0.0, self_path_reuse_skin_m=0.0,
                 seed_subdivision_depth=8, rigid_meshes=None, rigid_mesh_poses=None,
                 friction_iterations=0, dynamic_rigid_properties=None, dynamic_rigid_ids=None,
                 canonicalize_rigid_facets=False,rigid_acceleration=False,
                 rigid_solver='partitioned',abd_stiffness_pa=1.e8,abd_translation_axes=None,
                 abd_friction_policy='strict',abd_friction_passes=1,abd_force_tolerance_N=None,
                 abd_accuracy_profile=None,abd_velocity_tolerance_m_s=None,
                 abd_friction_velocity_epsilon_m_s=None,gel_poisson_ratio=None,
                 gel_stiffness_scale=DEFAULT_STIFFNESS_SCALE,
                 gel_material_model='stable_neo_hookean',
                 abd_final_path_policy='physical_segment'):
        if gel_material_model not in ('stable_neo_hookean','log_neo_hookean'):
            raise ValueError('unsupported gel_material_model')
        self.gel_material_model = gel_material_model
        # User-selected default: twice the original elastic stiffness.
        # Scale both Lame constants at fixed Poisson ratio; mass/damping stay
        # unchanged. Scale 1 with no Poisson override restores the original
        # material exactly. These are material choices, not calibration.
        gel_mu, gel_lambda = 43006.99300699301, 264185.8141858142
        from numbers import Real
        if (isinstance(gel_stiffness_scale,(bool,np.bool_))
                or not isinstance(gel_stiffness_scale,Real)
                or not np.isfinite(gel_stiffness_scale) or gel_stiffness_scale <= 0):
            raise ValueError('gel_stiffness_scale must be a positive finite scalar')
        if gel_poisson_ratio is not None:
            if (isinstance(gel_poisson_ratio,(bool,np.bool_))
                    or not isinstance(gel_poisson_ratio,Real)
                    or not np.isfinite(gel_poisson_ratio)
                    or not 0 <= gel_poisson_ratio < .5):
                raise ValueError('gel_poisson_ratio must be a finite scalar in [0, 0.5)')
            gel_lambda = 2*gel_mu*float(gel_poisson_ratio)/(1-2*float(gel_poisson_ratio))
        gel_mu *= float(gel_stiffness_scale)
        gel_lambda *= float(gel_stiffness_scale)
        if not np.isfinite([gel_mu,gel_lambda,gel_mu+gel_lambda]).all():
            raise ValueError('gel_stiffness_scale produces nonfinite material constants')
        gel_nu = gel_lambda/(2*(gel_lambda+gel_mu))
        self.gel_material_parameters = dict(mu_pa=gel_mu,lambda_pa=gel_lambda,
            poisson_ratio=gel_nu,young_modulus_pa=2*gel_mu*(1+gel_nu),
            bulk_modulus_pa=gel_lambda+2*gel_mu/3,density_kg_m3=1010.,damping=2.,
            stiffness_scale=float(gel_stiffness_scale))
        fingers = tuple(fingers)
        if not fingers or len(set(fingers)) != len(fingers) or any(f not in FINGERS for f in fingers):
            raise ValueError("select distinct Revo3 fingers")
        if probe_finger not in fingers:
            raise ValueError("probe finger must be enabled")
        radius = float(probe_radius_m)
        xy = np.asarray(probe_xy_m, dtype=float)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("probe radius must be finite and positive")
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("probe XY must contain two finite coordinates")
        self.rigid_meshes = tuple(rigid_meshes or ())
        if abd_friction_velocity_epsilon_m_s is not None:
            from numbers import Real
            value=abd_friction_velocity_epsilon_m_s
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<=0:
                raise ValueError('ABD friction smoothing velocity must be a positive finite scalar')
            if rigid_solver!='abd' or dynamic_rigid_properties is None:
                raise ValueError('ABD friction smoothing requires the dynamic ABD backend')
        from .abd_path_policy import validate_final_path_policy
        validate_final_path_policy(abd_final_path_policy)
        if abd_final_path_policy!='physical_segment' and (rigid_solver!='abd' or dynamic_rigid_properties is None):
            raise ValueError('incremental path policy requires the dynamic ABD backend')
        if abd_accuracy_profile not in (None,'tacex_ordinary','taccel_ordinary'):
            raise ValueError('unknown ABD accuracy profile')
        if abd_velocity_tolerance_m_s is not None and abd_accuracy_profile is None:
            raise ValueError('velocity tolerance requires an ordinary Newton profile')
        if abd_accuracy_profile is not None:
            if rigid_solver!='abd' or dynamic_rigid_properties is None:
                raise ValueError('ABD accuracy profile requires the dynamic ABD backend')
            if abd_force_tolerance_N is not None or abd_friction_passes!=1 or abd_friction_policy not in ('strict','lagged'):
                raise ValueError('ordinary Newton selects single lagged friction and velocity convergence; omit conflicting force/pass options')
            abd_friction_policy='lagged'
        self.abd_accuracy_profile=abd_accuracy_profile
        if abd_force_tolerance_N is not None and (rigid_solver!='abd' or dynamic_rigid_properties is None):
            raise ValueError('ABD force tolerance requires the dynamic ABD backend')
        if (abd_friction_policy!='strict' or abd_friction_passes!=1) and (rigid_solver!='abd' or dynamic_rigid_properties is None):
            raise ValueError('ABD friction policies require the dynamic ABD backend')
        if abd_translation_axes is not None and rigid_solver!='abd':
            raise ValueError('translation guides require the ABD backend')
        if not isinstance(canonicalize_rigid_facets,bool):raise ValueError('canonicalize_rigid_facets must be a bool')
        if canonicalize_rigid_facets:
            self.rigid_meshes=tuple(m.canonicalized_planar() for m in self.rigid_meshes)
        if dynamic_rigid_properties is not None and not self.rigid_meshes:
            raise ValueError('dynamic rigid bodies require mesh mode')
        if dynamic_rigid_ids is not None and dynamic_rigid_properties is None:
            raise ValueError('dynamic IDs require mass properties')
        if not isinstance(rigid_acceleration,bool):raise ValueError('rigid_acceleration must be a bool')
        if rigid_solver not in ('partitioned','abd'):raise ValueError('rigid_solver must be partitioned or abd')
        if rigid_solver=='abd' and (dynamic_rigid_properties is None or rigid_acceleration):
            raise ValueError('ABD requires dynamic rigid properties and owns its joint solve; disable partitioned acceleration')
        if rigid_acceleration and dynamic_rigid_properties is None:
            raise ValueError('rigid acceleration requires dynamic mass properties')
        if self.rigid_meshes:
            object_poses = rigid_poses(rigid_mesh_poses)
            if len(object_poses) != len(self.rigid_meshes):
                raise ValueError("provide one world pose per rigid mesh")
        elif rigid_mesh_poses is not None:
            raise ValueError("rigid poses require rigid meshes")
        self.assets, self.fingers = Path(assets), fingers
        self.data = {f: dict(np.load(self.assets/f"{f}.npz", allow_pickle=False)) for f in fingers}
        self.metadata = {f: json.loads(str(d["metadata_json"])) for f, d in self.data.items()}
        self.link_poses = np.array([d["link_to_world"] for d in self.data.values()])
        self.mount_offsets = np.array([d["sensor_to_link"] for d in self.data.values()])
        mounts = self.link_poses @ self.mount_offsets
        self.probe_finger = probe_finger
        self.probe_radius = radius
        self.dt = 1/240.
        self.ranges = {}
        builder = newton.ModelBuilder(gravity=gravity)
        pins, owners, pin_local = [], [], []
        rest_world, boundary = [], []
        for owner, (finger, d) in enumerate(self.data.items()):
            start = len(builder.particle_mass)
            tet_start = len(builder.tet_indices)
            q = d["rest_m"] @ mounts[owner, :3, :3].T + mounts[owner, :3, 3]
            builder.add_soft_mesh(pos=wp.vec3(0), rot=wp.quat_identity(), scale=1., vel=wp.vec3(0),
                vertices=q.tolist(), indices=d["tetrahedra"].ravel().tolist(), density=1010.,
                k_mu=gel_mu, k_lambda=gel_lambda, k_damp=2.,
                tri_ke=0., tri_ka=0., tri_kd=0., add_surface_mesh_edges=False,
                particle_radius=2.5e-5, label=f"/{finger}/gel")
            end = len(builder.particle_mass)
            if end-start != len(q):
                raise RuntimeError("Newton changed the mesh vertex ordering")
            if len(builder.tet_indices)-tet_start != len(d["tetrahedra"]):
                raise ValueError(f"Newton rejected an inverted {finger} tetrahedron; regenerate a better mesh")
            self.ranges[finger] = slice(start, end)
            rest_world.append(q)
            boundary.append(d["triangles"]+start)
            local_pins = d["pinned"].astype(int)
            for i in local_pins:
                builder.particle_mass[start+i] = 0.
            pins.extend(start+local_pins)
            owners.extend([owner]*len(local_pins))
            pin_local.extend(d["rest_m"][local_pins])
        self.rest = np.concatenate(rest_world)
        self.triangles = np.concatenate(boundary)
        particle_owners = np.concatenate([np.full(len(d["rest_m"]), i, int) for i, d in enumerate(self.data.values())])
        self.rigid_ranges = ()
        if self.rigid_meshes:
            from .rigid_mesh import append_rigid_meshes
            self.rigid_ranges, _ = append_rigid_meshes(builder,self.rigid_meshes,object_poses,particle_radius_m=2.5e-5)
            for i,(mesh,bounds) in enumerate(zip(self.rigid_meshes,self.rigid_ranges)):
                pins.extend(range(bounds.start,bounds.stop))
                owners.extend([len(fingers)+i]*len(mesh.vertices))
                pin_local.extend(mesh.vertices)
            particle_owners = np.concatenate([particle_owners]+[np.full(len(mesh.vertices),len(fingers)+i,int) for i,mesh in enumerate(self.rigid_meshes)])
            mounts = np.concatenate((mounts,object_poses))
            self.probe = np.zeros(3)
            self.probe_rest_local = self.probe_surface_local = None
        else:
            d = self.data[probe_finger]
            self.probe_surface_local = lift_to_coating(d["rest_m"], d["coating_triangles"], xy[None])[0]
            # A height sample plus sphere radius is unsafe on a sloping/curved pad:
            # a neighbouring face may contact the SIDE of the sphere first.
            # Offline placement uses exact closest points on every surface triangle.
            from scipy.optimize import brentq
            from trimesh.triangles import closest_point
            tri = d["rest_m"][d["triangles"]]*1000.
            def clearance(z):
                centers = np.tile([xy[0]*1000., xy[1]*1000., z*1000.], (len(tri), 1))
                nearest = closest_point(tri, centers)
                return np.linalg.norm(nearest-centers, axis=1).min()*.001-self.probe_radius-2.5e-5-.0003
            height = brentq(clearance, self.probe_surface_local[2],
                            d["rest_m"][:, 2].max()+self.probe_radius+.001, xtol=1e-13)
            self.probe_rest_local = np.array([*xy, height])
            self.probe = self.probe_world(self.probe_rest_local, mounts)
            shape = newton.ModelBuilder.ShapeConfig()
            shape.ke = shape.kd = shape.mu = 0.
            shape.has_particle_collision = False
            body = builder.add_body(xform=wp.transform(wp.vec3(*self.probe), wp.quat_identity()),
                                    is_kinematic=True, label="/ipc_probe")
            builder.add_shape_sphere(body, radius=self.probe_radius, cfg=shape)
        builder.color()
        self.model = builder.finalize(device=device)
        contact = IPCSphereContact(sphere_radius_m=self.probe_radius, activation_distance_m=.0001,
                                   barrier_stiffness=.1, min_gap_m=1e-6,
                                   particle_radius_m=self.model.particle_radius.numpy())
        solver_class, solver_args, solver_extra = SolverMountedIPCGPU, (contact,), {}
        if self.rigid_meshes:
            from .rigid_mesh_ipc import SolverRigidMeshIPCGPU
            solver_class, solver_args = SolverRigidMeshIPCGPU, ()
            solver_extra = dict(meshes=self.rigid_meshes,rigid_ranges=self.rigid_ranges)
        self.solver = solver_class(self.model, *solver_args, **solver_extra,
            pin_ids=np.asarray(pins, int), pin_owners=np.asarray(owners, int),
            pin_local=np.asarray(pin_local), mount_poses=mounts,
            particle_owners=particle_owners,
            iterations=128, force_tolerance_N=1e-6, min_tet_jacobian=.05,
            max_subdivision_depth=4, self_contact_gap_m=1e-6 if self.rigid_meshes else 1e-7,
            seed_subdivision_depth=seed_subdivision_depth,
            friction_iterations=friction_iterations,
            # A stopping-policy profile must not silently recalibrate friction.
            friction_coefficient=.5, friction_velocity_epsilon_m_s=.001 if abd_friction_velocity_epsilon_m_s is None else abd_friction_velocity_epsilon_m_s,
            self_activation_distance_m=5e-5, self_barrier_stiffness=1000.,
            self_friction_coefficient=.5, linear_tolerance=.01, state_precision="float64",
            volume_barrier_activation=volume_barrier_activation,
            contact_candidate_skin_m=contact_candidate_skin_m,
            self_path_reuse_skin_m=self_path_reuse_skin_m)
        self.solver.energy.material_model = gel_material_model
        self.state, self.scratch = self.solver.state(), self.solver.state()
        self.control = self.model.control()
        self.step_index = 0
        self.time_s = 0.
        self.renderers = {}
        self.image_size = image_size
        self._contact_resultants = wp.zeros((2, len(fingers)+len(self.rigid_meshes)), dtype=wp.vec3d, device=device)
        if dynamic_rigid_properties is not None:
            if rigid_solver=='abd':
                from .abd_dynamics import SolverABDFEMIPCGPU
                self.rigid_dynamics=SolverABDFEMIPCGPU(self.solver,dynamic_rigid_properties,
                    dynamic_ids=dynamic_rigid_ids,gravity=gravity,stiffness_pa=abd_stiffness_pa,translation_axes=abd_translation_axes,
                    friction_policy=abd_friction_policy,friction_passes=abd_friction_passes,
                    force_tolerance_N=abd_force_tolerance_N,
                    final_path_policy=abd_final_path_policy,
                    newton_convergence='velocity' if abd_accuracy_profile is not None else 'force',
                    velocity_tolerance_m_s=((.01 if abd_accuracy_profile=='taccel_ordinary' else .05)
                        if abd_velocity_tolerance_m_s is None else abd_velocity_tolerance_m_s))
            else:
                from .rigid_dynamics import SolverCoupledRigidIPCGPU
                self.rigid_dynamics=SolverCoupledRigidIPCGPU(self.solver,dynamic_rigid_properties,
                    dynamic_ids=dynamic_rigid_ids,gravity=gravity,warm_start=rigid_acceleration,
                    inexact_fem=rigid_acceleration)
            self.rigid_state=self.rigid_dynamics.state()
            self.rigid_scratch=self.rigid_dynamics.state()

    def probe_world(self, local, mounts=None):
        mounts = self.solver.mount_poses if mounts is None else mounts
        frame = mounts[self.fingers.index(self.probe_finger)]
        return (frame[:3, :3] @ np.asarray(local) + frame[:3, 3]).astype(np.float32).astype(float)

    def _record_command_failure(self, started, error):
        """A host validation failure reports zero work in both solver layers."""
        self.solver._record_step_work(started, error=error)
        if self.rigid_dynamics is not None:
            self.rigid_dynamics.last_step_work = dict(
                accepted=False, error=error, wall_ms=1000*(perf_counter()-started),
                converged=False,accepted_without_convergence=False,budget_accepted_intervals=0,
                input_rejected=True, joint_solves=0, fem_evaluations=0,
                newton_iterations=0, pcg_iterations=0, capture_ms=0.,
                published_substeps=[], failed_attempts=[], attempts=[])

    def step(self, probe_target=None, *, link_poses=None, dt=None, rigid_targets=None,
             external_rigid_wrenches=None):
        """Advance transactionally; update link poses only on a successful solve."""
        started = perf_counter()
        try:
            poses = self.link_poses if link_poses is None else rigid_poses(link_poses)
            if poses.shape != self.link_poses.shape:
                raise ValueError("link poses must match the selected finger order")
            if external_rigid_wrenches is not None and self.rigid_dynamics is None:
                raise ValueError('external rigid wrenches require dynamic rigid bodies')
            if self.rigid_meshes:
                if probe_target is not None:
                    raise ValueError("mesh mode takes rigid_targets, not a sphere center")
                target = self.solver.mount_poses[len(self.fingers):] if rigid_targets is None else rigid_poses(rigid_targets)
                if len(target) != len(self.rigid_meshes):
                    raise ValueError("rigid target count cannot change")
            else:
                if rigid_targets is not None:
                    raise ValueError("rigid targets require mesh mode")
                target = np.asarray(probe_target, float).astype(np.float32).astype(float)
            step_dt = positive_scalar(self.dt if dt is None else dt, "dt")
        except Exception as exc:
            # A rejected hand command must not expose the preceding solver
            # call's successful work report. No GPU work has run for this call.
            self._record_command_failure(started, str(exc) or type(exc).__name__)
            raise
        if self.rigid_meshes:
            mounts=np.concatenate((poses @ self.mount_offsets,target))
            if self.rigid_dynamics is None:
                result = self.solver.step(self.state,self.scratch,self.control,None,step_dt,mount_target=mounts)
            else:
                result=self.rigid_dynamics.step(self.state,self.scratch,self.rigid_state,self.rigid_scratch,
                    step_dt,mount_target=mounts,external_wrenches=external_rigid_wrenches,control=self.control)
                self.rigid_state,self.rigid_scratch=self.rigid_scratch,self.rigid_state
        else:
            result = self.solver.step(self.state, self.scratch, self.control, None,
                step_dt, sphere_center=self.probe, sphere_target=target,
                mount_target=poses @ self.mount_offsets)
            self.probe = target.copy()
        self.state, self.scratch = self.scratch, self.state
        self.link_poses = poses.copy()
        self.step_index += 1
        self.time_s += step_dt
        return result

    def snapshot(self, *, optical=True, optical_coverage=True):
        """Accepted geometry and instantaneous contact resultants in world axes.

        The total includes sphere and soft/soft contact (normal and friction).
        It is not a backing reaction or a time average over adaptive substeps.
        optical_coverage=False omits only the returned coverage array; marker
        occlusion and RGB still use the complete coating rasterization.
        """
        from .perspective_optical_gpu import CUDAPerspectiveTactileRenderer
        # Submit every camera before the first host download. Geometry and
        # marker interpolation read the accepted device state directly.
        optical_frames = {}
        if optical:
            try:
                for index, finger in enumerate(self.fingers):
                    d = self.data[finger]
                    if finger not in self.renderers:
                        layer = CurvedMarkerLayer(d["rest_m"], d["coating_triangles"])
                        size = self.image_size
                        camera = PinholeCamera(size, size, size*1.25, size*1.25,
                                               (size-1)*.5, (size-1)*.5, np.eye(3), np.array([0, 0, .035]))
                        self.renderers[finger] = CUDAPerspectiveTactileRenderer(
                            d["rest_m"], d["coating_triangles"], layer, camera,
                            supersample=2, outward_winding=True, device=self.model.device)
                    self.renderers[finger].enqueue(self.state.particle_q,
                        offset=self.ranges[finger].start, pose=self.solver.mount_poses[index])
                for finger in self.fingers:
                    self.renderers[finger].copy_outputs(coverage=optical_coverage)
                wp.synchronize_device(self.model.device)
                for finger in self.fingers:
                    optical_frames[finger] = self.renderers[finger].collect()
            finally:
                for renderer in self.renderers.values():
                    renderer.discard_pending()
        world = self.state.particle_q.numpy()
        self._contact_resultants.zero_()
        wp.launch(contact_resultants, self.model.particle_count,
                  [self.solver._particle_owners, self.solver.contact_forces,
                   self.solver.self_contact_forces, self._contact_resultants], device=self.model.device)
        # Download two vectors per finger instead of both per-node force arrays.
        forces = self._contact_resultants.numpy()
        result = dict(positions_m=world, probe_m=self.probe.copy(), step=self.step_index,
                      gel_material_model=self.solver.energy.material_model,
                      gel_material=self.gel_material_parameters.copy() if hasattr(self, 'gel_material_parameters') else None,
                      time_s=self.time_s, link_poses=self.link_poses.copy(),
                      mount_poses=self.solver.mount_poses.copy(),
                      probe_finger=self.probe_finger, fingers={})
        for index, finger in enumerate(self.fingers):
            d, bounds = self.data[finger], self.ranges[finger]
            frame = result['mount_poses'][index]
            local = (world[bounds]-frame[:3, 3]) @ frame[:3, :3]
            sphere, soft = forces[:, index]
            info = dict(local_positions_m=local, force_world_N=sphere+soft,
                        sphere_force_world_N=sphere.copy(), soft_contact_force_world_N=soft.copy())
            if optical:
                info["optical"] = optical_frames[finger]
            result["fingers"][finger] = info
        if self.rigid_meshes:
            result['probe_m'] = None
            result['rigid_meshes'] = [dict(name=mesh.name,pose=self.solver.mount_poses[len(self.fingers)+i].copy(),
                positions_m=world[bounds].copy(),triangles=mesh.triangles.copy(),
                force_world_N=forces[:,len(self.fingers)+i].sum(axis=0))
                for i,(mesh,bounds) in enumerate(zip(self.rigid_meshes,self.rigid_ranges))]
            for info in result['fingers'].values():
                info['surface_contact_force_world_N'] = info.pop('soft_contact_force_world_N')
            if self.rigid_dynamics is not None:
                velocities=self.rigid_state.velocities.numpy()
                for b,i in enumerate(self.rigid_dynamics.ids):
                    result['rigid_meshes'][i].update(dynamic=True,com_velocity_world_m_s=velocities[b,:3].copy(),
                        angular_velocity_world_rad_s=velocities[b,3:].copy())
                    if hasattr(self.rigid_state,'q'):
                        affine=result['rigid_meshes'][i]['pose'][:3,:3]
                        result['rigid_meshes'][i].update(dynamics_model='abd',
                            orthogonality_error=float(np.linalg.norm(affine.T@affine-np.eye(3))),
                            affine_determinant=float(np.linalg.det(affine)))
                    elif getattr(self.rigid_dynamics,'display_name',None)=='MuJoCo Warp':
                        result['rigid_meshes'][i]['dynamics_model']='mujoco_warp'
        return result
