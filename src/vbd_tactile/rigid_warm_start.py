"""GPU-only guarded reuse of FEM iterates within a rigid interface solve.

The previous solution is advected to the new rigid trial. It is only an initial
guess: old state, inertia, friction displacement and acceptance stay unchanged.
An infeasible or higher-energy guess is discarded entirely on device.
"""
import warp as wp
from .ipc_gpu_graph import IPCDeviceGraph
from . import ipc_gpu_control as ctl
from . import rigid_mesh_gpu as mesh_k
from .mounted_ipc import prescribe_backing
from .rigid_dynamics_gpu import rotation,translation

wp.set_module_options({'enable_backward':False})


@wp.kernel
def motion(old: wp.array(dtype=wp.mat44d), new: wp.array(dtype=wp.mat44d),
           radii: wp.array(dtype=wp.float64), delta: wp.array(dtype=wp.mat44d),
           skin: wp.array(dtype=wp.float64), base_skin: wp.float64):
    i=wp.tid();r=rotation(new[i])*wp.transpose(rotation(old[i]))
    p=translation(new[i])-r*translation(old[i])
    delta[i]=wp.mat44d(r[0,0],r[0,1],r[0,2],p[0],r[1,0],r[1,1],r[1,2],p[1],
                      r[2,0],r[2,1],r[2,2],p[2],0.,0.,0.,1.)
    norm=wp.float64(0.)
    for j in range(3):
        for k in range(3):norm+=(r[j,k]-wp.float64(j==k))*(r[j,k]-wp.float64(j==k))
    displacement=wp.length(translation(new[i])-translation(old[i]))+radii[i]*wp.sqrt(norm)
    wp.atomic_max(skin,0,wp.float64(4.)*displacement+base_skin)


@wp.kernel
def valid_positions(x: wp.array(dtype=wp.vec3d), eligible: wp.array(dtype=int)):
    if eligible[0]==0:return
    p=x[wp.tid()]
    if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
        wp.atomic_min(eligible,0,0)


@wp.kernel
def choose(feasible: wp.array(dtype=int), stats: wp.array(dtype=wp.float64),
           original: wp.array(dtype=wp.float64), restore: wp.array(dtype=int),
           accepted: wp.array(dtype=int)):
    keep=feasible[0]!=0 and stats[2]==wp.float64(0.)
    keep=keep and wp.isfinite(stats[0]) and wp.isfinite(stats[1]) and stats[0]<original[0]
    restore[0]=int(not keep);accepted[0]=int(keep)


@wp.kernel
def report(accepted: wp.array(dtype=int), active: wp.array(dtype=int),
           rows: wp.array2d(dtype=wp.float64)):
    rows[1,16]=wp.float64(accepted[0]);rows[1,17]=wp.float64(active[0])


class CoupledWarmStartGraph(IPCDeviceGraph):
    has_warm_seed=True

    def __init__(self,solver,*,warm_friction=False):
        super().__init__(solver)
        self.warm_friction=warm_friction
        self.warm_positions=wp.empty_like(solver.positions)
        self.warm_mounts=wp.empty_like(solver._mount_device)
        self.warm_delta=wp.empty_like(solver._mount_device)
        self.warm_radii=wp.array(solver._mount_radii,dtype=wp.float64,device=self.device)
        self.warm_skin=wp.zeros(1,dtype=wp.float64,device=self.device)
        self.warm_active=wp.zeros(1,dtype=int,device=self.device)
        self.warm_eligible=wp.zeros(1,dtype=int,device=self.device)
        self.seed=wp.empty_like(solver.positions)
        self.seed_stats=wp.empty_like(self.e.stats)
        self.seed_feasible=wp.zeros(1,dtype=int,device=self.device)
        self.seed_restore=wp.zeros(1,dtype=int,device=self.device)
        self.seed_accepted=wp.zeros(1,dtype=int,device=self.device)

    def prepare(self,state):
        if self.s.friction_iterations<2:
            raise ValueError('coupled warm start requires iterated friction and its extended GPU report')
        wp.load_module(module=__name__,device=self.device)
        super().prepare(state)

    def record(self,state,output,control,dt,c0,c1):
        self.first_value=True;self.seed_accepted.zero_()
        super().record(state,output,control,dt,c0,c1)
        self.launch(report,1,[self.seed_accepted,self.warm_active,self.report])

    def value(self,x,old,dt,c0,c1):
        super().value(x,old,dt,c0,c1)
        if not self.first_value:return
        self.first_value=False
        base_value=super().value
        s=self.s
        def candidate():
            wp.copy(self.seed,x);wp.copy(self.seed_stats,self.e.stats)
            self.warm_skin.zero_()
            self.launch(motion,len(s.mount_poses),[self.warm_mounts,s._mount_device,self.warm_radii,
                self.warm_delta,self.warm_skin,2*self.e.mesh.dhat+self.e.mesh.dmin])
            s._refit_rigid(self.warm_positions)
            self.launch(mesh_k.mesh_seed,self.e.n,[self.warm_positions,s.model.particle_mass,
                s._particle_owners,self.warm_delta,s._rigid_faces,s._rigid_bvh.id,self.warm_skin,x])
            self.launch(prescribe_backing,len(s._pin_ids),
                [s._pin_ids,s._pin_owners,s._pin_local,s._mount_device,x])
            self.paths(old,x,c0,c1)
            s.self_motion.check(old,x,readback=False)
            self.launch(ctl.trial_feasible,1,[s.diagnostic,s.self_motion.stats,s.contact.min_gap_m,
                s.min_jacobian,self.seed_feasible])
            wp.capture_if(self.seed_feasible,lambda:base_value(x,old,dt,c0,c1))
            self.launch(choose,1,[self.seed_feasible,self.e.stats,self.seed_stats,self.seed_restore,self.seed_accepted])
            def restore():
                wp.copy(x,self.seed)
                wp.capture_if(self.seed_feasible,lambda:base_value(x,old,dt,c0,c1))
            wp.capture_if(self.seed_restore,restore)
        wp.copy(self.warm_eligible,self.warm_active)
        self.launch(valid_positions,self.e.n,[self.warm_positions,self.warm_eligible])
        wp.capture_if(self.warm_eligible,candidate)
        if self.warm_friction:
            # A warm position paired with the old friction frame solves an
            # intentionally stale problem before recovering the current frame.
            # Initialize the fixed point from the accepted warm geometry. Its
            # slip is STILL x-old, and the original final force-change and FEM
            # residual checks still decide convergence. A rejected/NaN warm
            # guess cannot update the frame. Never copy candidate-cache data
            # into the lagged frame without recomputing normals and loads.
            def refresh():
                self.freeze(x,dt,c1,initial=False)
                self._value_from_current_contacts(x,old,dt,c0,c1)
            wp.capture_if(self.seed_accepted,refresh)
