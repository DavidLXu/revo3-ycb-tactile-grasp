"""Build a closed, decimated physics surface from a YCB google_16k model (meters, YCB frame unchanged).

The textured OBJ is kept for rendering. The physics mesh comes from nontextured.ply, merged, quadric
decimated with Open3D, and must be watertight, winding-consistent, single-shell and free of
self-intersections; otherwise the target face count is raised and retried.
"""
import argparse, json
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh

p = argparse.ArgumentParser()
p.add_argument('model_dir'); p.add_argument('output'); p.add_argument('--faces', type=int, default=2500)
a = p.parse_args()
src = trimesh.load(Path(a.model_dir) / 'nontextured.ply', force='mesh', process=True)
src.merge_vertices()
report = dict(source_vertices=len(src.vertices), source_faces=len(src.faces), source_watertight=bool(src.is_watertight))


def self_intersections(m):
    o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    return len(np.asarray(o.get_self_intersecting_triangles()))


def remesh(m, pitch=.0004):
    # Fallback for scans with non-manifold edges: solid voxelization + marching cubes (closed surface).
    v = m.voxelized(pitch).fill()
    out = v.marching_cubes
    out.apply_transform(v.transform)
    return out


target = a.faces
if not src.is_watertight:
    report['source_repair'] = 'largest component + 0.4 mm voxel remesh (source has non-manifold edges)'
    src = max(src.split(only_watertight=False), key=lambda c: len(c.faces))
    src = remesh(src)
    report['remeshed_faces'] = len(src.faces)
for attempt in range(8):
    o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(src.vertices), o3d.utility.Vector3iVector(src.faces))
    o = o.simplify_quadric_decimation(target)
    o.remove_duplicated_vertices(); o.remove_degenerate_triangles(); o.remove_duplicated_triangles(); o.remove_unreferenced_vertices()
    m = trimesh.Trimesh(np.asarray(o.vertices), np.asarray(o.triangles), process=True)
    trimesh.repair.fix_normals(m)
    ok = (m.is_watertight and m.is_winding_consistent and len(m.split(only_watertight=False)) == 1
          and m.volume > 0 and self_intersections(m) == 0)
    report.setdefault('attempts', []).append(dict(target=target, faces=len(m.faces), watertight=bool(m.is_watertight),
                                                  ok=bool(ok)))
    if ok:
        break
    target = int(target * 1.4)
else:
    raise SystemExit('could not produce a valid closed physics mesh: ' + json.dumps(report))
orig = trimesh.load(Path(a.model_dir) / 'nontextured.ply', force='mesh', process=True)
d = trimesh.proximity.closest_point(m, orig.vertices[::7])[1]
report.update(faces=len(m.faces), vertices=len(m.vertices), volume_m3=float(m.volume), extents_m=m.extents.tolist(),
              source_to_physics_max_m=float(d.max()), source_to_physics_p95_m=float(np.percentile(d, 95)))
m.export(a.output)
Path(a.output).with_suffix('.json').write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != 'attempts'}))
