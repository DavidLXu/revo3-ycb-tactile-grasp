"""Headless check of the interactive app: drives the sliders from code, then switches object.

    python scripts/smoke_app.py --port 8091
Prints the contact forces after closing the grip, and exits non-zero if no finger touches.
"""
import argparse
import threading
import time

import numpy as np
import viser

from revo3_grasp.app import App
from revo3_grasp.paths import OBJECT_LABELS
from revo3_grasp.synergy import FINGERS

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, default=8091)
p.add_argument('--device', default='cuda:0')
a = p.parse_args()
app = App(viser.ViserServer(host='127.0.0.1', port=a.port, verbose=False), a.device, '013_apple')
threading.Thread(target=app.run, daemon=True).start()


def wait(cond, timeout, what):
    t = time.time()
    while not cond():
        if time.time() - t > timeout:
            raise SystemExit(f'timeout waiting for {what}')
        time.sleep(.5)


def settled(grip):
    s = app.session
    return s is not None and s.at({f: s.curl_for_grip(f, grip) for f in FINGERS})


ok = True
for oid, grips in (('013_apple', (.95, .8)), ('025_mug', (.9,))):
    if app.session is None or app.session.object_id != oid:
        app.obj.value = OBJECT_LABELS[oid]           # same path as a user picking the dropdown
        wait(lambda: app.session is not None and app.session.object_id == oid, 3600, f'{oid} scene')
    for g in grips:
        app.master.value = g                          # 'move all fingers together' copies to every finger
        t0 = time.time()
        wait(lambda: settled(g), 1800, f'{oid} grip {g}')
        time.sleep(1.)
        F = app.session.last['forces_N']
        print(f'{oid} grip {g:.2f}: forces {np.round(F, 2).tolist()} N, {time.time() - t0:.0f} s, '
              f'status {app.session.last["status"]}', flush=True)
        ok &= bool(F.max() > .1)
print('SMOKE_OK' if ok else 'SMOKE_FAIL', flush=True)
raise SystemExit(0 if ok else 1)
