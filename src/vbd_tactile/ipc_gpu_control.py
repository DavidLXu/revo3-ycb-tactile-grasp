"""Device decisions for the transactional IPC graph (no host convergence loop).

Status slots: 0 error, 5 accepted Newton iterations, 6 current line retries,
7 required contact capacity, 8 accepted trial, 9 total rejected trials,
10 executed Hessians, 11 cumulative PCG iterations, 12 seed repair iterations,
13 whether the last seed projection changed any stored position,
14 whether a capacity overflow occurred while freezing initial contacts.
Value slots: 0 energy, 1 residual, 2 alpha, 3..5 certified gap/J/surface gap,
7 contact count, 8 physical CCD pair count, 9..11 PCG residual/tolerance/count,
12..15 self force max and sum. Report row zero is packed by ``report``;
later rows hold accepted-iteration history and are bounded by the Newton budget.
History row 1 stores final path diagnostics in columns 9..13, the capacity
phase in column 14, and seed repairs in 15.
"""
from typing import Any
import warp as wp
from .ipc_precision import position_overloads, pose_value

wp.set_module_options({"enable_backward": False})


@wp.kernel
def update_pose(c0: wp.vec3d, c1: wp.vec3d,
                old: wp.array(dtype=wp.vec3d), target: wp.array(dtype=wp.vec3d),
                delta: wp.array(dtype=wp.vec3d)):
    old[0] = c0
    target[0] = c1
    delta[0] = c1-c0


@wp.kernel
def publish_sphere(target: wp.array(dtype=wp.vec3d), delta: wp.array(dtype=wp.vec3d), dt: wp.float64,
                   pose: wp.array(dtype=wp.transform), velocity: wp.array(dtype=wp.spatial_vector)):
    pose[0] = wp.transform(wp.vec3(target[0]), wp.transform_get_rotation(pose[0]))
    velocity[0] = wp.spatial_vector(wp.vec3(delta[0]/dt), wp.vec3(0.0))


@wp.kernel
def reset_path(d: wp.array(dtype=wp.float64)):
    for i in range(6):
        d[i] = wp.float64(0.0)
    d[2] = wp.float64(wp.inf)
    d[3] = wp.float64(wp.inf)
    d[5] = wp.float64(wp.inf)


@wp.kernel
def initialize(s: wp.array(dtype=int), v: wp.array(dtype=wp.float64)):
    for i in range(16):
        s[i] = 0
        v[i] = wp.float64(0.0)
    v[3] = wp.float64(wp.inf)
    v[4] = wp.float64(wp.inf)
    v[5] = wp.float64(wp.inf)


@wp.kernel
def gate(s: wp.array(dtype=int), out: wp.array(dtype=int)):
    out[0] = int(s[0] == 0)


@wp.kernel
def flag_error(flag: wp.array(dtype=int), code: int, s: wp.array(dtype=int)):
    if flag[0] != 0 and s[0] == 0:
        s[0] = code


@wp.kernel
def capacity(count: wp.array(dtype=int), invalid: wp.array(dtype=int), maximum: int,
             initial: bool, s: wp.array(dtype=int)):
    if s[0] == 0:
        if invalid[0] != 0:
            s[0] = 3
        elif count[0] > maximum:
            s[0] = 2
            s[7] = count[0]
            s[14] = int(initial)


@wp.func
def safe_path(d: wp.array(dtype=wp.float64), gap: wp.float64, jac: wp.float64):
    return (d[0] == wp.float64(0.0) and d[2] > gap and d[3] > jac and d[5] > gap)


@wp.kernel
def check_path(d: wp.array(dtype=wp.float64), motion: wp.array(dtype=int),
               gap: wp.float64, jac: wp.float64, code: int,
               s: wp.array(dtype=int), v: wp.array(dtype=wp.float64)):
    if s[0] == 0:
        if not safe_path(d, gap, jac) or motion[1] != 0 or motion[2] != 0:
            s[0] = code
        else:
            v[3] = wp.min(v[3], d[2])
            v[4] = wp.min(v[4], d[3])
            v[5] = wp.min(v[5], d[5])
            v[8] = wp.float64(motion[0])


@wp.kernel
def seed_needed(d: wp.array(dtype=wp.float64), gap: wp.float64, out: wp.array(dtype=int)):
    out[0] = int(d[0] != wp.float64(0.0) or d[5] <= gap)


@wp.kernel
def objective(stats: wp.array(dtype=wp.float64), count: wp.array(dtype=int),
              s: wp.array(dtype=int), v: wp.array(dtype=wp.float64)):
    if stats[2] != wp.float64(0.0) or not wp.isfinite(stats[0]) or not wp.isfinite(stats[1]):
        s[0] = 3
    else:
        v[0] = stats[0]
        v[1] = stats[1]
        v[7] = wp.float64(count[0])


@wp.kernel
def newton_condition(s: wp.array(dtype=int), v: wp.array(dtype=wp.float64),
                     tolerance: wp.float64, maximum: int, out: wp.array(dtype=int)):
    out[0] = 0
    if s[0] == 0:
        if not wp.isfinite(tolerance) or tolerance <= wp.float64(0.0) or maximum < 1:
            s[0] = 29
        elif not wp.isfinite(v[1]) or v[1] < wp.float64(0.0):
            s[0] = 3
        elif v[1] > tolerance:
            if s[5] >= maximum:
                s[0] = 7
            else:
                out[0] = 1


@wp.kernel
def preconditioner(stats: wp.array(dtype=wp.float64), s: wp.array(dtype=int)):
    s[10] += 1
    if stats[2] != wp.float64(0.0):
        s[0] = 4


@wp.kernel
def linear(stats: wp.array(dtype=wp.float64), tol: wp.float64,
           s: wp.array(dtype=int), v: wp.array(dtype=wp.float64)):
    actual = wp.sqrt(wp.max(stats[0], wp.float64(0.0)))
    threshold = wp.max(wp.float64(1e-10), tol * wp.sqrt(wp.max(stats[1], wp.float64(0.0))))
    good = stats[2] < wp.float64(0.0) and actual <= threshold * wp.float64(1.05)
    for i in range(5):
        good = good and wp.isfinite(stats[i])
    if not good:
        s[0] = 5
    v[9] = actual
    v[10] = threshold
    v[11] = stats[3]
    s[11] += int(stats[3])
    s[6] = 0
    s[8] = 0
    v[2] = wp.float64(1.0)


@position_overloads("x", "out")
@wp.kernel
def trial(x: wp.array(dtype=Any), direction: wp.array(dtype=wp.vec3d),
          gradient: wp.array(dtype=wp.vec3d), mass: wp.array(dtype=float),
          v: wp.array(dtype=wp.float64), out: wp.array(dtype=Any), stats: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = x[i]
    if mass[i] > 0.0:
        out[i] = type(x[i])(wp.vec3d(x[i]) + v[2] * direction[i])
    dx = wp.vec3d(out[i]) - wp.vec3d(x[i])
    wp.atomic_add(stats, 0, wp.dot(gradient[i], dx))
    wp.atomic_add(stats, 1, wp.dot(dx, dx))
    for j in range(3):
        if not wp.isfinite(wp.float64(out[i][j])) or not wp.isfinite(direction[i][j]):
            wp.atomic_add(stats, 2, wp.float64(1.0))


@wp.kernel
def trial_direction(stats: wp.array(dtype=wp.float64), s: wp.array(dtype=int)):
    if stats[2] != wp.float64(0.0) or not wp.isfinite(stats[0]) or not wp.isfinite(stats[1]):
        s[0] = 6
    elif stats[0] >= wp.float64(0.0) or stats[1] == wp.float64(0.0):
        s[0] = 6


@wp.kernel
def trial_feasible(d: wp.array(dtype=wp.float64), motion: wp.array(dtype=int),
                   gap: wp.float64, jac: wp.float64, out: wp.array(dtype=int)):
    out[0] = int(safe_path(d, gap, jac) and motion[1] == 0 and motion[2] == 0)


@wp.func
def save_trial(accepted: bool, stats: wp.array(dtype=wp.float64),
               d: wp.array(dtype=wp.float64), count: wp.array(dtype=int),
               s: wp.array(dtype=int), v: wp.array(dtype=wp.float64),
               history: wp.array2d(dtype=wp.float64)):
    good = stats[2] == wp.float64(0.0) and wp.isfinite(stats[0]) and wp.isfinite(stats[1])
    if good and accepted:
        row = s[5] + 1
        history[row, 0] = v[0]
        history[row, 1] = stats[0]
        history[row, 2] = v[2]
        history[row, 3] = v[1]
        history[row, 4] = v[7]
        history[row, 5] = d[3]
        history[row, 6] = v[11]
        history[row, 7] = v[9]
        history[row, 8] = v[10]
        v[0] = stats[0]
        v[1] = stats[1]
        v[7] = wp.float64(count[0])
        v[3] = wp.min(v[3], d[2])
        v[4] = wp.min(v[4], d[3])
        v[5] = wp.min(v[5], d[5])
        s[5] += 1
        s[8] = 1


@wp.kernel
def accept_trial(stats: wp.array(dtype=wp.float64), line: wp.array(dtype=wp.float64),
                 d: wp.array(dtype=wp.float64), count: wp.array(dtype=int),
                 s: wp.array(dtype=int), v: wp.array(dtype=wp.float64),
                 history: wp.array2d(dtype=wp.float64)):
    save_trial(stats[0] <= v[0] + wp.float64(1e-4)*line[0], stats, d, count, s, v, history)


@wp.kernel
def accept_trial_increment(stats: wp.array(dtype=wp.float64), line: wp.array(dtype=wp.float64),
                           d: wp.array(dtype=wp.float64), count: wp.array(dtype=int),
                           s: wp.array(dtype=int), v: wp.array(dtype=wp.float64),
                           history: wp.array2d(dtype=wp.float64),
                           increment: wp.array(dtype=wp.float64), disabled: wp.array(dtype=int)):
    accepted = stats[0] <= v[0] + wp.float64(1e-4)*line[0]
    if disabled[0] == 0:
        accepted = wp.isfinite(increment[0]) and increment[0] <= wp.float64(1e-4)*line[0]
    save_trial(accepted, stats, d, count, s, v, history)


@wp.kernel
def line_condition(s: wp.array(dtype=int), v: wp.array(dtype=wp.float64),
                   advance: bool, out: wp.array(dtype=int)):
    out[0] = 0
    if s[0] == 0 and s[8] == 0:
        if advance:
            s[6] += 1
            s[9] += 1
            v[2] *= wp.float64(0.5)
        if s[6] >= 60:
            s[0] = 6
        else:
            out[0] = 1


@wp.kernel
def forces(a: wp.array(dtype=wp.float64), b: wp.array(dtype=wp.float64),
           s: wp.array(dtype=int), v: wp.array(dtype=wp.float64)):
    if a[2] != wp.float64(0.0) or b[2] != wp.float64(0.0):
        s[0] = 9
    v[12] = b[1]
    v[13] = b[3]
    v[14] = b[4]
    v[15] = b[5]


@wp.kernel
def report(s: wp.array(dtype=int), v: wp.array(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    out[0, 0] = wp.float64(s[0])
    out[0, 1] = wp.float64(s[5])
    out[0, 2] = v[1]
    out[0, 3] = v[7]
    out[0, 4] = v[3]
    out[0, 5] = v[4]
    out[0, 6] = v[5]
    out[0, 7] = v[8]
    for i in range(4):
        out[0, 8+i] = v[12+i]
    out[0, 12] = wp.float64(s[7])
    out[0, 13] = wp.float64(s[9])
    out[0, 14] = wp.float64(s[10])
    out[0, 15] = wp.float64(s[11])
    out[1, 14] = wp.float64(s[14])
    out[1, 15] = wp.float64(s[12])


@wp.kernel
def path_report(d: wp.array(dtype=wp.float64), motion: wp.array(dtype=int),
                out: wp.array2d(dtype=wp.float64)):
    out[1, 9] = d[3]
    out[1, 10] = d[2]
    out[1, 11] = d[5]
    out[1, 12] = d[0]
    out[1, 13] = wp.float64(int(motion[1] != 0) + 2 * int(motion[2] != 0))


@wp.kernel
def select_contact_cg(normal: wp.array(dtype=int), friction: wp.array(dtype=int),
                      full: wp.array(dtype=int), fixed: wp.array(dtype=int)):
    dynamic = normal[0] > 0 or friction[0] > 0
    full[0] = int(dynamic)
    fixed[0] = int(not dynamic)


@wp.kernel
def cg_condition(it: wp.array(dtype=int), cond: wp.array(dtype=int),
                 residual: wp.array(dtype=wp.float64), tolerance: wp.array(dtype=wp.float64), maximum: int):
    it[0] += 1
    cond[0] = int(it[0] < maximum and residual[0] > tolerance[0] and wp.isfinite(residual[0]))


@position_overloads(poses=('center',))
@wp.kernel
def validate_body(pose: wp.array(dtype=wp.transform), velocity: wp.array(dtype=wp.spatial_vector),
                  center: Any, s: wp.array(dtype=int)):
    _center = pose_value(center)
    p, vel = pose[0], velocity[0]
    for j in range(7):
        if not wp.isfinite(p[j]):
            s[0] = 20
    for j in range(6):
        if not wp.isfinite(vel[j]):
            s[0] = 20
    norm = wp.float64(0.0)
    for j in range(4):
        norm += wp.float64(p[j+3]) * wp.float64(p[j+3])
    if wp.abs(norm - wp.float64(1.0)) > wp.float64(1.1e-5):
        s[0] = 21
    for j in range(3):
        if vel[j+3] != 0.0:
            s[0] = 22
        if wp.float64(p[j]) != _center[j]:
            s[0] = 23


@wp.kernel
def volume_status(invalid: wp.array(dtype=int), stats: wp.array(dtype=int), s: wp.array(dtype=int)):
    if invalid[0] != 0 or stats[1] != 0:
        s[0] = 27
