"""Forward kinematics of the textured Revo3 right hand from its USD joints (numpy, no USD needed)."""
import json
from pathlib import Path
import numpy as np

from .paths import HAND


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def frame(pos, rot):
    T = np.eye(4)
    T[:3, :3] = quat_matrix(rot if rot is not None else [1, 0, 0, 0])
    T[:3, 3] = pos if pos else 0
    return T


def axis_rotation(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    i = 'XYZ'.index(axis)
    T = np.eye(4)
    j, k = (i+1) % 3, (i+2) % 3
    T[j, j], T[j, k], T[k, j], T[k, k] = c, -s, s, c
    return T


class HandFK:
    def __init__(self, joints=HAND/'joints.json', links=HAND/'links.npz'):
        d = np.load(links)
        self.rest = {str(n): d[f'world/{n}'] for n in d['names']}
        self.root = d['root_world']
        self.joints = {}
        for e in json.loads(Path(joints).read_text()):
            b0 = e['body0'][0].split('/')[-1] if e['body0'] else 'world'
            b1 = e['body1'][0].split('/')[-1]
            self.joints[e['path'].split('/')[-1]] = dict(
                type=e['type'], parent=b0, child=b1, J0=frame(e['pos0'], e['rot0']), J1=frame(e['pos1'], e['rot1']),
                axis=e['attrs'].get('physics:axis', 'X'),
                limits=(e['attrs'].get('physics:lowerLimit'), e['attrs'].get('physics:upperLimit')))
        self.revolute = [k for k, v in self.joints.items() if v['type'] == 'PhysicsRevoluteJoint']
        self.by_child = {v['child']: k for k, v in self.joints.items()}

    def poses(self, angles_deg=None):
        """World 4x4 of every link; angles_deg maps revolute joint name -> degrees (default 0)."""
        angles_deg = angles_deg or {}
        out = {}

        def solve(link):
            if link in out:
                return out[link]
            name = self.by_child.get(link)
            if name is None:
                out[link] = self.rest[link]
                return out[link]
            j = self.joints[name]
            if j['parent'] not in self.rest:   # world anchor
                out[link] = self.rest[link]
                return out[link]
            P = solve(j['parent'])
            R = axis_rotation(j['axis'], angles_deg.get(name, 0.)) if j['type'] == 'PhysicsRevoluteJoint' else np.eye(4)
            out[link] = P @ j['J0'] @ R @ np.linalg.inv(j['J1'])
            return out[link]
        for n in self.rest:
            solve(n)
        return out


def proper(T):
    """Nearest proper rigid transform (USD stores float32 quaternions, |q|-1 ~ 4e-8)."""
    T = np.array(T, float)
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = U @ Vt
    T[3] = [0, 0, 0, 1]
    return T


if __name__ == '__main__':
    fk = HandFK()
    zero = fk.poses()
    err = max(abs(zero[n] - fk.rest[n]).max() for n in fk.rest)
    print('revolute', len(fk.revolute), 'rest reproduction max error', err)
    unresolved = [n for n in fk.rest if n not in fk.by_child]
    print('links without a parent joint', unresolved)
