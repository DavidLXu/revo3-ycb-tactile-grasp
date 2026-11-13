"""Finger curl synergies shared by the planner, physics driver and renderer."""
import numpy as np
FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
CHAIN = {f: (f'right_{f}_MCP_joint', f'right_{f}_PIP_joint', f'right_{f}_DIP_joint') for f in FINGERS}
# Curl synergies (MCP, PIP, DIP degrees at s=1); each finger independently picks the best one.
PROFILES = {'power': (90., 90., 63.), 'mid': (90., 55., 40.), 'precision': (90., 25., 18.)}
THUMB_PROFILES = {'power': (70., 80., 70.), 'mid': (70., 45., 40.), 'precision': (70., 15., 15.)}
S = np.linspace(0., 1., 61)


def curl_angles(f, s, thumb, profile='power'):
    ratio = (THUMB_PROFILES if f == 'thumb' else PROFILES)[profile]
    out = {j: s * r for j, r in zip(CHAIN[f], ratio)}
    if f == 'thumb':
        out.update({'right_thumb_CMP_joint': thumb[0], 'right_thumb_CMR_joint': thumb[1]})
    return out


def hand_angles(curls, thumb, profiles=None):
    out = {}
    for f, s in curls.items():
        out.update(curl_angles(f, s, thumb, (profiles or {}).get(f, 'power')))
    out.setdefault('right_thumb_CMP_joint', thumb[0]); out.setdefault('right_thumb_CMR_joint', thumb[1])
    return out
