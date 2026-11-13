"""Interactive grasp demo in the browser (viser).

    python -m revo3_grasp.app --port 8080          # on the GPU machine
    ssh -L 8080:localhost:8080 <gpu-host>          # then open http://localhost:8080

Sliders move the fingers (prescribed joint motion); the fingertip gels are simulated live on the GPU
and the per-finger contact force updates every physics step. The dropdown switches YCB objects
(each switch rebuilds the scene; the very first launch also compiles the CUDA kernels, which can
take several minutes and is cached afterwards).
"""
import argparse
import threading
import time
import traceback

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation

from .hand_fk import HandFK
from .paths import HAND, OBJECT_LABELS, OBJECTS, YCB
from .sim import MAX_CLOSURE_MM, TOUCH, GraspSession
from .synergy import FINGERS

GEL_COLOR = (238, 168, 92)
HAND_COLOR = (186, 189, 196)
FORCE_SCALE_N = 15.


def wxyz(T):
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.array([w, x, y, z])


class App:
    def __init__(self, server, device, initial):
        self.server, self.device = server, device
        self.fk = HandFK()
        links = np.load(HAND / 'links.npz')
        self.lock = threading.Lock()
        self.session = None
        self.pending = initial
        self.gels = {}
        self.object_handle = None
        server.scene.set_up_direction('+z')
        self.link_handles = {}
        for n in self.fk.rest:
            if f'verts/{n}' not in links or 'DIP_rubber' in n:   # DIP rubber is replaced by the simulated gel
                continue
            self.link_handles[n] = server.scene.add_mesh_simple(
                f'/hand/{n}', links[f'verts/{n}'].astype(np.float32), links[f'faces/{n}'].astype(np.uint32),
                color=HAND_COLOR, flat_shading=False)
        self._build_gui()

        @server.on_client_connect
        def _(client):
            client.camera.position = (0.42, 0.30, 0.24)
            client.camera.look_at = (0.08, 0.0, 0.12)
            client.camera.up_direction = (0., 0., 1.)

    # ---- GUI --------------------------------------------------------------------------------
    def _build_gui(self):
        g = self.server.gui
        g.add_markdown('### Revo3 hand · YCB grasp\nFinger motion is prescribed; fingertip gels are FEM + IPC '
                       'on the GPU; the object is held fixed (no lifting).')
        self.obj = g.add_dropdown('Object', options=[OBJECT_LABELS[o] for o in OBJECTS],
                                  initial_value=OBJECT_LABELS[self.pending])
        self.together = g.add_checkbox('Move all fingers together', True)
        self.master = g.add_slider('Grip (all)', 0., 1., .01, 0.,
                                   marks=((0., 'open'), (TOUCH, 'touch'), (1., f'+{MAX_CLOSURE_MM:g} mm')))
        with g.add_folder('Per finger'):
            self.sliders = {f: g.add_slider(f.capitalize(), 0., 1., .01, 0.) for f in FINGERS}
        self.open_btn = g.add_button('Open hand')
        with g.add_folder('Contact force (N)'):
            self.bars = {f: g.add_progress_bar(0., color='orange') for f in FINGERS}
            self.force_md = g.add_markdown('')
        self.status = g.add_markdown('Loading...')

        @self.obj.on_update
        def _(_):
            oid = next(o for o in OBJECTS if OBJECT_LABELS[o] == self.obj.value)
            with self.lock:
                self.pending = oid

        @self.master.on_update
        def _(_):
            if self.together.value:
                for s in self.sliders.values():
                    s.value = self.master.value

        @self.open_btn.on_click
        def _(_):
            self.master.value = 0.
            for s in self.sliders.values():
                s.value = 0.

    # ---- scene ------------------------------------------------------------------------------
    def _load(self, oid):
        self.status.content = f'Building **{OBJECT_LABELS[oid]}** scene on {self.device} (first run compiles CUDA kernels)...'
        for h in list(self.gels.values()):
            h[0].remove()
        self.gels.clear()
        if self.object_handle is not None:
            self.object_handle.remove()
        self.session = None
        session = GraspSession(oid, device=self.device, fk=self.fk)
        mesh = trimesh.load(YCB / oid / 'textured.obj', force='mesh', process=False)
        T = session.object_to_world
        self.object_handle = self.server.scene.add_mesh_trimesh('/object', mesh, wxyz=wxyz(T), position=T[:3, 3])
        for f in FINGERS:
            used, faces = session.gel_surface(f)
            v = session.positions[used].astype(np.float32)
            h = self.server.scene.add_mesh_simple(f'/gel/{f}', v, faces.astype(np.uint32), color=GEL_COLOR)
            self.gels[f] = (h, used, faces)
        self.master.value = 0.
        for s in self.sliders.values():
            s.value = 0.
        self.session = session
        self._refresh()
        self.status.content = f'**{OBJECT_LABELS[oid]}** ready (setup {session.setup_s:.0f} s). Drag the sliders.'

    def _refresh(self):
        s = self.session
        P = s.link_poses()
        for n, h in self.link_handles.items():
            h.position, h.wxyz = P[n][:3, 3], wxyz(P[n])
        for f, (h, used, faces) in self.gels.items():
            v = s.positions[used].astype(np.float32)
            try:
                h.vertices = v
            except Exception:  # older viser: re-add the mesh under the same name
                self.gels[f] = (self.server.scene.add_mesh_simple(f'/gel/{f}', v, faces.astype(np.uint32),
                                                                   color=GEL_COLOR), used, faces)
        F = s.last['forces_N']
        rows = ['| finger | force | closure past touch |', '|---|---|---|']
        for k, f in enumerate(FINGERS):
            self.bars[f].value = float(min(F[k] / FORCE_SCALE_N, 1.) * 100)
            c = s.closure_mm(f)
            rows.append(f'| {f} | {F[k]:5.2f} N | ' + ('never touches' if c is None else
                        'open' if c < 0 else f'{c:4.2f} mm') + ' |')
        rows.append(f'\nsum {F.sum():.2f} N · last step {s.last["wall_ms"]:.0f} ms · {s.last["status"]}')
        self.force_md.content = '\n'.join(rows)

    # ---- loop -------------------------------------------------------------------------------
    def run(self):
        settle = 0
        while True:
            with self.lock:
                oid, self.pending = self.pending, None
            if oid is not None:
                try:
                    self._load(oid)
                except Exception:
                    traceback.print_exc()
                    self.status.content = f'Failed to build {oid}; see server log.'
                settle = 0
                continue
            s = self.session
            if s is None:
                time.sleep(.1)
                continue
            targets = {f: s.curl_for_grip(f, self.sliders[f].value) for f in FINGERS}
            if s.at(targets):
                if settle >= 3:            # a few extra steps so the gel settles at the new pose
                    time.sleep(.03)
                    continue
                settle += 1
                try:
                    s.step_curls(dict(s.curls))
                except Exception as exc:  # a rejected step leaves the state unchanged
                    s.last = dict(s.last, status=f'settle step rejected: {str(exc)[:60]}')
                    settle = 3
            else:
                settle = 0
                s.step_toward(targets)
            self._refresh()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--object', default='013_apple', choices=OBJECTS)
    a = p.parse_args()
    App(viser.ViserServer(host=a.host, port=a.port), a.device, a.object).run()


if __name__ == '__main__':
    main()
