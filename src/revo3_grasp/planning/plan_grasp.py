"""Kinematic grasp planner for a fixed object and the textured Revo3 right hand (planning only, CPU).

Each finger closes along one curl parameter s in [0,1] (MCP:PIP:DIP coupled). The thumb first takes
an opposition pose (CMP, CMR) and then curls. For each candidate object pose the planner finds, per
finger, the curl where its rest-shape gel first touches the closed physics mesh, rejecting poses
where any other hand link (all visuals except the hidden DIP rubber) comes closer than a clearance
to the object first. The final curl adds a fixed rigid-gel interpenetration depth; the soft solver
then turns that commanded closure into actual deformation and force.
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation
from ..hand_fk import HandFK
from ..paths import FINGERTIPS, HAND

from ..synergy import FINGERS, CHAIN, PROFILES, THUMB_PROFILES, S, curl_angles, hand_angles  # noqa: F401


class Planner:
    def __init__(self, tips, mesh, clearance=.0008, rng=np.random.default_rng(0)):
        self.fk = HandFK(); d = np.load(HAND / 'links.npz')
        self.tips = {f: dict(np.load(Path(tips) / f'{f}.npz')) for f in FINGERS}
        self.mesh = mesh; self.clearance = clearance
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))))
        # Gel boundary in sensor frame; other link visuals in link frames (subsampled for planning).
        self.gel = {}
        for f, t in self.tips.items():
            b = np.unique(t['triangles']); g = t['rest_m'][b]
            # Contact-facing half of the gel (above the mount) carries all planning contacts.
            g = g[g[:, 2] > np.percentile(g[:, 2], 40)]
            self.gel[f] = g[rng.choice(len(g), min(len(g), 250), replace=False)]
        self.links = {}
        for n in self.fk.rest:
            if f'verts/{n}' not in d or 'DIP_rubber' in n:
                continue
            v = d[f'verts/{n}']
            self.links[n] = v[rng.choice(len(v), min(len(v), 120), replace=False)]
        self.owner = {n: next((f for f in FINGERS if f'_{f}_' in n), None) for n in self.links}

    def sdf(self, p):
        return self.scene.compute_signed_distance(o3d.core.Tensor(p.astype(np.float32))).numpy()

    def gel_world(self, f, poses):
        T = poses[f'right_{f}_DIP_link'] @ self.tips[f]['sensor_to_link']
        return self.gel[f] @ T[:3, :3].T + T[:3, 3]

    def link_world(self, n, poses):
        T = poses[n]; return self.links[n] @ T[:3, :3].T + T[:3, 3]

    def finger_points(self, f, s, thumb, profile):
        P = self.fk.poses(hand_angles({f: s}, thumb, {f: profile}))
        return self.gel_world(f, P), np.concatenate([self.link_world(n, P) for n, o in self.owner.items() if o == f])

    def sweep(self, thumb, grid=S):
        """Hand-frame point sets per finger, profile and curl value (independent of object pose)."""
        out = {}
        for f in FINGERS:
            for prof in PROFILES:
                pts = [self.finger_points(f, s, thumb, prof) for s in grid]
                out[f, prof] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
        P = self.fk.poses(hand_angles({}, thumb))
        out['palm'] = np.concatenate([self.link_world(n, P) for n, o in self.owner.items() if o is None])
        return out

    def finger(self, gel, other, grid, to_obj, depth):
        g = self.sdf(to_obj(gel.reshape(-1, 3))).reshape(len(grid), -1)
        o = self.sdf(to_obj(other.reshape(-1, 3))).reshape(len(grid), -1).min(1)
        if g[0].min() < self.clearance or o[0] < self.clearance:
            return 'overlap'
        clear = np.minimum.accumulate(o) >= self.clearance
        reach = np.flatnonzero((-g.min(1) >= depth) & clear)
        if len(reach) == 0:
            free = np.flatnonzero(clear & (g.min(1) > 0))
            return dict(contact=False, s=float(grid[free[-1]]))
        k = reach[0]; i = np.argmin(g[k])
        return dict(contact=True, s_contact=float(grid[np.flatnonzero(g.min(1) < 0)[0]]), s=float(grid[k]),
                    rigid_depth_m=float(-g[k].min()), link_clearance_m=float(o[k]), pad_index=int(i))

    def evaluate(self, sweep, Tobj, depth, grid=S):
        """Object pose Tobj (object->world): best profile per finger, or None if the hand overlaps the object."""
        inv = np.linalg.inv(Tobj)
        to_obj = lambda p: p @ inv[:3, :3].T + inv[:3, 3]
        if self.sdf(to_obj(sweep['palm'])).min() < 2 * self.clearance:
            return None
        res = {}
        for f in FINGERS:
            best = None
            for prof in PROFILES:
                r = self.finger(*sweep[f, prof], grid, to_obj, depth)
                if r == 'overlap':
                    return None
                r['profile'] = prof
                if r['contact']:
                    r['pad_point_sensor_m'] = self.gel[f][r.pop('pad_index')].tolist()
                    r['pad_offset_m'] = float(np.hypot(*r['pad_point_sensor_m'][:2]))
                key = (r['contact'], -r.get('pad_offset_m', 0), -r['s'])
                if best is None or key > best[0]:
                    best = (key, r)
            res[f] = best[1]
        return res

    def refine(self, Tobj, coarse, depth, thumb):
        """Re-solve each finger on a fine curl grid with all link points."""
        out = {}
        fine = np.linspace(0., 1., 801)
        inv = np.linalg.inv(Tobj); to_obj = lambda p: p @ inv[:3, :3].T + inv[:3, 3]
        for f, r in coarse.items():
            pts = [self.finger_points(f, s, thumb, r['profile']) for s in fine]
            x = self.finger(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), fine, to_obj, depth)
            x['profile'] = r['profile']
            if x['contact']:
                x['pad_point_sensor_m'] = self.gel[f][x.pop('pad_index')].tolist()
                x['pad_offset_m'] = float(np.hypot(*x['pad_point_sensor_m'][:2]))
            out[f] = x
        return out


def score(res):
    if res is None:
        return -np.inf
    if not res['thumb']['contact']:
        return -1
    n = sum(r['contact'] for r in res.values())
    off = sum(r['pad_offset_m'] for r in res.values() if r['contact'])
    return n - 20 * off


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', required=True); p.add_argument('--tips', default=str(FINGERTIPS)); p.add_argument('--output', required=True)
    p.add_argument('--depth-mm', type=float, default=1.0); p.add_argument('--step-mm', type=float, default=8)
    p.add_argument('--orientations', default='identity', help='identity or all')
    a = p.parse_args()
    mesh = trimesh.load(a.mesh, force='mesh'); centroid = mesh.centroid.copy()
    mesh.apply_translation(-centroid)   # plan in a centroid-origin object frame
    pl = Planner(a.tips, mesh)
    rots = [np.eye(3)] if a.orientations == 'identity' else [
        Rotation.from_euler('xyz', e, degrees=True).as_matrix()
        for e in [(0, 0, 0), (90, 0, 0), (0, 90, 0), (90, 0, 90), (0, 90, 90), (90, 90, 0)]]
    step = a.step_mm / 1000
    grid = list(itertools.product(np.arange(.02, .12, step), np.arange(-.03, .035, step), np.arange(.05, .17, step)))
    best = (-np.inf, None)
    for thumb in [(70., 20.), (90., 20.), (90., 40.), (110., 40.), (110., 60.)]:
        pl.thumb = thumb; sw = pl.sweep(thumb)
        for ri, R in enumerate(rots):
            for c in grid:
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = c
                res = pl.evaluate(sw, T, a.depth_mm / 1000)
                sc = score(res)
                if sc > best[0]:
                    best = (sc, dict(thumb=thumb, rotation_index=ri, T=T.tolist(), fingers=res))
                    print(f'score {sc:.3f} thumb {thumb} rot {ri} c {np.round(c, 3)} contacts',
                          [f for f in FINGERS if res[f]['contact']], flush=True)
    sc, plan = best
    # Local refinement of the position around the best candidate.
    pl.thumb = tuple(plan['thumb']); sw = pl.sweep(pl.thumb)
    T0 = np.array(plan['T'])
    for d in itertools.product(*[np.linspace(-step, step, 5)] * 3):
        T = T0.copy(); T[:3, 3] += d
        res = pl.evaluate(sw, T, a.depth_mm / 1000); s2 = score(res)
        if s2 > sc:
            sc, plan = s2, dict(plan, T=T.tolist(), fingers=res)
    plan['fingers'] = pl.refine(np.array(plan['T']), plan['fingers'], a.depth_mm / 1000, pl.thumb)
    sc = score(plan['fingers'])
    Tw = np.array(plan['T']); C = np.eye(4); C[:3, 3] = -centroid
    plan.update(score=sc, depth_mm=a.depth_mm, centroid_m=centroid.tolist(),
                object_to_world=(Tw @ C).tolist(), profiles=PROFILES, thumb_profiles=THUMB_PROFILES, mesh=a.mesh)
    Path(a.output).write_text(json.dumps(plan, indent=1))
    print('BEST', sc, json.dumps({f: {k: v for k, v in r.items() if k in ('contact', 's')} for f, r in plan['fingers'].items()}))
