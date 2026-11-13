"""Observe the physical chord independently of incremental IPC acceptance.

The incremental policy certifies Newton segments and the final configuration.
It does NOT certify straight interpolation between physical endpoint states.
Keep the old-to-new chord audit as telemetry, including its first failing pair.
All fields share the solver's existing report readback.
"""
import warp as wp
from .ipc_gpu_control import safe_path

wp.set_module_options({'enable_backward': False})


def validate_final_path_policy(policy):
    if policy not in ('physical_segment', 'incremental'):
        raise ValueError('ABD final path policy must be physical_segment or incremental')


@wp.kernel
def reset_observation(report: wp.array2d(dtype=wp.float64)):
    for i in range(23, 36):
        report[0, i] = wp.float64(0.)
    for i in range(31, 36):
        report[0, i] = wp.float64(-1.)


@wp.kernel
def observe_segment(d: wp.array(dtype=wp.float64), motion: wp.array(dtype=int),
                    witness: wp.array2d(dtype=int), faces: wp.array2d(dtype=int),
                    edges: wp.array2d(dtype=int), gap: wp.float64, jac: wp.float64,
                    report: wp.array2d(dtype=wp.float64)):
    report[0, 23] = wp.float64(1.)
    report[0, 24] = wp.float64(int(safe_path(d, gap, jac) and motion[1] == 0 and motion[2] == 0))
    report[0, 25] = d[3]
    report[0, 26] = wp.float64(motion[1])
    report[0, 27] = d[0]
    report[0, 28] = wp.float64(motion[2])
    report[0, 29] = d[2]
    report[0, 30] = d[5]
    if motion[1] > 0 and motion[2] == 0:
        kind, a, b = witness[0, 0], witness[0, 1], witness[0, 2]
        report[0, 31] = wp.float64(kind)
        if kind == 0:
            report[0, 32] = wp.float64(a)
            for j in range(3):
                report[0, 33 + j] = wp.float64(faces[b, j])
        else:
            for j in range(2):
                report[0, 32 + j] = wp.float64(edges[a, j])
                report[0, 34 + j] = wp.float64(edges[b, j])
