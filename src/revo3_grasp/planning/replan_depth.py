"""Usage: python -m revo3_grasp.planning.replan_depth <object_id> <depth_mm>

Re-solve the final finger curls of an existing plan for a different rigid closure depth (same object pose)."""
import json, sys
import numpy as np, trimesh
from .plan_grasp import Planner, score
from ..paths import FINGERTIPS, YCB
obj, depth_mm = sys.argv[1], float(sys.argv[2])
p = json.load(open(YCB / obj / 'plan.json'))
mesh = trimesh.load(YCB / obj / 'physics.ply', force='mesh'); mesh.apply_translation(-np.array(p['centroid_m']))
pl = Planner(FINGERTIPS, mesh)
p['fingers'] = pl.refine(np.array(p['T']), p['fingers'], depth_mm / 1000, tuple(p['thumb']))
p.update(score=score(p['fingers']), depth_mm=depth_mm, replanned_from_depth_mm=json.load(open(YCB / obj / 'plan.json'))['depth_mm'])
open(YCB / obj / 'plan.json', 'w').write(json.dumps(p, indent=1))
print({f: (r['contact'], round(r['s'], 4), r.get('pad_point_sensor_m')) for f, r in p['fingers'].items()})
