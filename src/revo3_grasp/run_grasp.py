"""Scripted grasp (approach, squeeze, hold, release) for one object; saves states for rendering.

    python -m revo3_grasp.run_grasp --object 013_apple --output outputs/013_apple

Writes states.npz (gel positions per 1/30 s step, curls, forces) and report.json.
"""
import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from .paths import OBJECTS
from .sim import GraspSession
from .synergy import FINGERS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--object', required=True, choices=OBJECTS)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-release', action='store_true')
    p.add_argument('--dt', type=float, default=1 / 30, help='physics step and saved-frame interval (s), e.g. 1/60 for 60 fps video')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    session = GraspSession(a.object, device=a.device, dt=a.dt)
    plan = session.plan['fingers']
    fin = {f: r['s'] for f, r in plan.items()}
    pre = {f: (r['s_contact'] - .25 * (r['s'] - r['s_contact'])) if r['contact'] else .85 * r['s']
           for f, r in plan.items()}
    ease = lambda u: .5 - .5 * np.cos(np.pi * np.clip(u, 0, 1))
    phases = [('approach', .8), ('squeeze', .6), ('hold', .6)] + ([] if a.no_release else [('release', .5), ('rest', .2)])
    steps = []
    for name, dur in phases:
        n = int(round(dur / session.dt))
        for k in range(1, n + 1):
            u = ease(k / n)
            if name == 'approach':
                c = {f: u * pre[f] for f in FINGERS}
            elif name == 'squeeze':
                c = {f: pre[f] + u * (fin[f] - pre[f]) for f in FINGERS}
            elif name == 'hold':
                c = dict(fin)
            elif name == 'release':
                c = {f: fin[f] + u * (pre[f] - fin[f]) for f in FINGERS}
            else:
                c = dict(pre)
            steps.append((name, c))
    rows, states, forces, curls, error = [], [], [], [], None
    n_gel = session.ranges['little'].stop
    try:
        for i, (name, c) in enumerate(steps, 1):
            r = session.step_curls(c)
            rows.append(dict(step=i, phase=name, time_s=i * session.dt, curls=c, wall_ms=r['wall_ms'],
                             finger_force_N=r['force_vectors_N'].tolist(),
                             object_force_N=np.asarray(r['object_force_N']).tolist(),
                             **{k: r.get(k) for k in ('iterations', 'max_residual_N', 'min_tet_jacobian',
                                                      'internal_substeps', 'converged')}))
            states.append(session.positions[:n_gel].astype(np.float32))
            forces.append(r['force_vectors_N'])
            curls.append([c[f] for f in FINGERS])
            print(i, name, f"{r['wall_ms']:.0f} ms", np.round(r['forces_N'], 2).tolist(), flush=True)
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        np.savez_compressed(out / 'states.npz', positions=np.asarray(states), finger_forces=np.asarray(forces),
                            curls=np.asarray(curls), initial=session.hand.rest.astype(np.float32),
                            ranges=np.array([[session.ranges[f].start, session.ranges[f].stop] for f in FINGERS]),
                            gel_triangles=session.surface, object_to_world=session.object_to_world,
                            thumb=np.array(session.thumb), profiles=np.array([session.profiles[f] for f in FINGERS]))
        report = dict(complete=error is None and len(rows) == len(steps), error=error, steps=len(rows),
                      planned_steps=len(steps), object=a.object, dt_s=session.dt, setup_s=session.setup_s,
                      plan=session.plan, rows=rows)
        (out / 'report.json').write_text(json.dumps(report, indent=1, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)))
        print('COMPLETE', report['complete'], flush=True)
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
