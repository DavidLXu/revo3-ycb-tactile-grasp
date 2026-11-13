"""One grasp session: the textured Revo3 hand's five FEM fingertip gels against one fixed YCB object.

Physics is the vendored prescribed-motion solver (`vbd_tactile.revo3.Revo3TactileHand`, rigid-mesh
mode): complete fingertip gels (stable Neo-Hookean, 2x stiffness), GPU IPC contact with friction,
implicit time steps that are only accepted when the force residual is <= 1e-6 N and every element and
path check passes. Finger joints are *prescribed* (each gel stays pinned to its DIP link, whose pose
comes from the hand's forward kinematics); the object pose is *prescribed and constant* - it has no
mass and is never lifted.

A finger's pose is one curl value s in [0, 1] (MCP/PIP/DIP coupled by the synergy stored in the
object's plan). The grip mapping used by the UI:

    grip 0 .. TOUCH         open hand -> curl at which the rigid gel first touches the object
    grip TOUCH .. 1         0 .. MAX_CLOSURE_MM of extra rigid closure past first touch

"Closure past touch" is how far the undeformed gel would overlap the object; the solver turns it
into real gel deformation and contact force. It is a commanded displacement, not a force.
"""
import json
import time

import numpy as np
import warp as wp

from vbd_tactile.ipc_vbd import IPCStepRejected
from vbd_tactile.revo3 import Revo3TactileHand
from vbd_tactile.rigid_mesh import RigidTriangleMesh

from .hand_fk import HandFK, proper
from .paths import FINGERTIPS, YCB
from .synergy import FINGERS, hand_angles

TOUCH = 0.7
MAX_CLOSURE_MM = 1.5


class GraspSession:
    def __init__(self, object_id, *, device='cuda:0', dt=1 / 30, fk=None):
        self.object_id, self.dt = object_id, dt
        folder = YCB / object_id
        self.plan = json.loads((folder / 'plan.json').read_text())
        self.thumb = tuple(self.plan['thumb'])
        self.profiles = {f: r['profile'] for f, r in self.plan['fingers'].items()}
        depth = self.plan['depth_mm']
        self.finger = {}
        for f, r in self.plan['fingers'].items():
            if r['contact']:
                rate = (r['s'] - r['s_contact']) / depth          # curl per mm of closure past touch
                self.finger[f] = dict(contact=True, s_touch=r['s_contact'], rate=rate)
            else:  # never reaches the object without another link hitting it first: stop at the safe curl
                self.finger[f] = dict(contact=False, s_touch=r['s'], rate=0.)
        self.fk = fk or HandFK()
        self.object_to_world = np.array(self.plan['object_to_world'])
        mesh = RigidTriangleMesh.load(folder / 'physics.ply', scale_to_m=1.0)
        started = time.perf_counter()
        self.hand = Revo3TactileHand(FINGERTIPS, fingers=FINGERS, rigid_meshes=[mesh],
                                     rigid_mesh_poses=[self.object_to_world], device=device,
                                     contact_candidate_skin_m=.0003, image_size=128)
        zero = np.array([proper(self.fk.poses()[f'right_{f}_DIP_link']) for f in FINGERS])
        if abs(zero - self.hand.link_poses).max() > 1e-6:
            raise RuntimeError('hand forward kinematics does not match the physics fingertip mounts')
        self.curls = {f: 0. for f in FINGERS}
        self.ranges = {f: self.hand.ranges[f] for f in FINGERS}
        self.surface = np.asarray(self.hand.triangles).reshape(-1, 3)
        self.step_scale = 1.
        # Move the thumb from its rest pose into the planned opposition pose (no contact yet).
        self._step(dict(self.curls))
        wp.synchronize()
        self.setup_s = time.perf_counter() - started
        self.last = dict(status='ready', forces_N=np.zeros(5), wall_ms=0.)
        self._update_snapshot()

    # ---- mapping -------------------------------------------------------------------------
    def curl_for_grip(self, f, g):
        p = self.finger[f]
        g = float(np.clip(g, 0., 1.))
        if g <= TOUCH or not p['contact']:
            return min(g / TOUCH, 1.) * p['s_touch']
        return min(p['s_touch'] + p['rate'] * (g - TOUCH) / (1 - TOUCH) * MAX_CLOSURE_MM, 1.)

    def closure_mm(self, f, s=None):
        """Commanded rigid closure past first touch (mm); negative = still open; None if it never touches."""
        p, s = self.finger[f], self.curls[f] if s is None else s
        if not p['contact']:
            return None
        return (s - p['s_touch']) / p['rate'] if s >= p['s_touch'] else -1. if s < p['s_touch'] - 1e-9 else 0.

    # ---- stepping ------------------------------------------------------------------------
    def link_poses(self, curls=None):
        return self.fk.poses(hand_angles(curls or self.curls, self.thumb, self.profiles))

    def _step(self, curls):
        P = self.link_poses(curls)
        dip = np.array([proper(P[f'right_{f}_DIP_link']) for f in FINGERS])
        return self.hand.step(link_poses=dip, rigid_targets=[self.object_to_world], dt=self.dt)

    def step_curls(self, curls):
        """One 1/30 s physics step to exactly these curls. Raises IPCStepRejected (state unchanged)."""
        wp.synchronize()
        t0 = time.perf_counter()
        diag = self._step(curls)
        wp.synchronize()
        self.curls = dict(curls)
        self._update_snapshot(wall_ms=(time.perf_counter() - t0) * 1e3, diag=diag)
        return self.last

    def step_toward(self, targets, free_cap=.02, contact_cap_mm=.15):
        """Rate-limited step toward target curls. Near or past touch, a finger moves at most
        contact_cap_mm of closure per step (scaled down after a rejected step)."""
        nxt = {}
        for f in FINGERS:
            s, t, p = self.curls[f], targets[f], self.finger[f]
            near = p['contact'] and max(s, t) > p['s_touch'] - 0.5 * free_cap
            cap = p['rate'] * contact_cap_mm * self.step_scale if near else free_cap
            nxt[f] = s + float(np.clip(t - s, -cap, cap))
        try:
            out = self.step_curls(nxt)
            self.step_scale = min(1., self.step_scale * 1.5)
            return out
        except IPCStepRejected as exc:
            self.step_scale = max(self.step_scale / 2, 1 / 64)
            self.last = dict(self.last, status=f'step rejected, retrying smaller ({str(exc)[:80]})')
            return self.last

    def at(self, targets, tol=1e-6):
        return all(abs(self.curls[f] - targets[f]) <= tol for f in FINGERS)

    def _update_snapshot(self, wall_ms=0., diag=None):
        snap = self.hand.snapshot(optical=False)
        forces = np.array([snap['fingers'][f]['force_world_N'] for f in FINGERS])
        d = {}
        if diag is not None:
            for k in ('iterations', 'max_residual_N', 'min_tet_jacobian', 'internal_substeps', 'converged'):
                d[k] = getattr(diag, k, None)
        self.positions = snap['positions_m']
        self.last = dict(status='ok', forces_N=np.linalg.norm(forces, axis=1), force_vectors_N=forces,
                         object_force_N=snap['rigid_meshes'][0]['force_world_N'], wall_ms=wall_ms, **d)

    def gel_surface(self, f):
        """(vertices, faces) of finger f's simulated gel surface, local indices."""
        r = self.ranges[f]
        tri = self.surface[(self.surface >= r.start).all(1) & (self.surface < r.stop).all(1)]
        used = np.unique(tri)
        remap = np.full(r.stop, -1)
        remap[used] = np.arange(len(used))
        return used, remap[tri]
