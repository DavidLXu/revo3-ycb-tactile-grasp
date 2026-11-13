"""GPU ordinary Newton termination in the units used by TacEx/libuipc.

The direction is unscaled by line search. Physical vertex corrections include
the affine mapping, while affine shape coordinates are divided by body radius.
Convergence requires the full unscaled path to be feasible, independently of
energy backtracking. The optional iteration-limit exit is explicitly reported
as unconverged and makes no force-equilibrium claim.
"""
import warp as wp

wp.set_module_options({'enable_backward': False})


@wp.kernel
def vertex_correction(direction: wp.array(dtype=wp.vec3d), owner: wp.array(dtype=int),
                      weights: wp.array(dtype=wp.vec4d), mass: wp.array(dtype=float),
                      n: int, metrics: wp.array(dtype=wp.float64)):
    i = wp.tid()
    delta = wp.vec3d(0.)
    if owner[i] >= 0:
        for j in range(4):
            delta += weights[i][j] * direction[n + 4 * owner[i] + j]
    elif mass[i] > 0.:
        delta = direction[i]
    for j in range(3):
        if not wp.isfinite(delta[j]):
            wp.atomic_add(metrics, 2, wp.float64(1.))
        else:
            wp.atomic_max(metrics, 0, wp.abs(delta[j]))


@wp.kernel
def affine_correction(direction: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64),
                      n: int, metrics: wp.array(dtype=wp.float64)):
    b = wp.tid()
    for col in range(1, 4):
        for axis in range(3):
            delta = direction[n + 4 * b + col][axis] / radius[b]
            if not wp.isfinite(delta):
                wp.atomic_add(metrics, 2, wp.float64(1.))
            else:
                wp.atomic_max(metrics, 1, wp.abs(delta))


@wp.kernel
def condition(status: wp.array(dtype=int), values: wp.array(dtype=wp.float64),
              done: wp.array(dtype=int), maximum: int, out: wp.array(dtype=int)):
    out[0] = 0
    if status[0] == 0:
        if not wp.isfinite(values[1]) or values[1] < wp.float64(0.):
            status[0] = 3
        elif done[0] == 0:
            # Exactly zero gradient needs no linear solve/descent direction.
            # This is NOT the former 1e-6 N acceptance shortcut.
            if values[1] == wp.float64(0.):
                done[0] = 2
            elif status[5] >= maximum:
                status[0] = 7
            else:
                out[0] = 1


@wp.kernel
def accept_iteration_limit(status: wp.array(dtype=int), values: wp.array(dtype=wp.float64),
                           done: wp.array(dtype=int), maximum: int):
    # Only a completed Newton loop's budget exit is eligible. All its accepted
    # updates already passed the original linear solve, CCD and Armijo tests.
    # Final geometry, shape, force and finite-state checks still run afterward.
    # Reason 5 is deliberately NOT convergence; never clear other failures.
    if status[0] == 7 and done[0] == 0 and maximum > 0 and status[5] == maximum:
        if wp.isfinite(values[0]) and wp.isfinite(values[1]) and values[1] >= wp.float64(0.):
            status[0] = 0
            done[0] = 5
        else:
            status[0] = 3


@wp.kernel
def full_path(values: wp.array(dtype=wp.float64), feasible: wp.array(dtype=int),
              metrics: wp.array(dtype=wp.float64)):
    # Only a successful audit of the entire Newton direction certifies the
    # CCD fraction = 1 condition. A short backtracking trial cannot set it.
    if values[2] == wp.float64(1.) and feasible[0] != 0:
        metrics[3] = wp.float64(1.)


@wp.kernel
def check(status: wp.array(dtype=int), values: wp.array(dtype=wp.float64),
          metrics: wp.array(dtype=wp.float64), dt: wp.float64,
          velocity_tol: wp.float64, transform_tol: wp.float64, done: wp.array(dtype=int)):
    metrics[4] = values[2]
    if status[0] == 0:
        if metrics[2] != wp.float64(0.):
            status[0] = 3
        elif (status[8] == 1 and metrics[3] == wp.float64(1.)
              and metrics[0] <= velocity_tol * dt
              and metrics[1] <= transform_tol * dt):
            done[0] = 1


@wp.kernel
def coordinate_resolution(base: wp.array(dtype=wp.vec3d), direction: wp.array(dtype=wp.vec3d),
                          mass: wp.array(dtype=float), metrics: wp.array(dtype=wp.float64)):
    # Normwise backward error of each free 3-vector, in float64 coordinate
    # units. Zero coordinates introduce no absolute length/force floor.
    i=wp.tid()
    if mass[i]>0.:
        size=wp.float64(0.); change=wp.float64(0.)
        for j in range(3):
            size=wp.max(size,wp.abs(base[i][j])); change=wp.max(change,wp.abs(direction[i][j]))
        ratio=wp.float64(0.)
        if change>wp.float64(0.):
            ratio=wp.float64(wp.inf)
            if size>wp.float64(0.):ratio=(change/size)/wp.float64(2.220446049250313e-16)
        for j in range(3):
            if not wp.isfinite(base[i][j]) or not wp.isfinite(direction[i][j]):ratio=wp.float64(wp.inf)
        wp.atomic_max(metrics,5,ratio)


@wp.kernel
def stationary_candidate(status: wp.array(dtype=int), values: wp.array(dtype=wp.float64),
                         metrics: wp.array(dtype=wp.float64), guard: wp.array(dtype=wp.float64),
                         use_guard: bool, dt: wp.float64, velocity_tol: wp.float64,
                         transform_tol: wp.float64, out: wp.array(dtype=int)):
    # The unscaled Newton direction is a stopping test at the CURRENT iterate.
    # A verified linear solve precedes this test. There is no absolute force
    # shortcut and no acceptance of a small line-search alpha.
    out[0] = 0
    # Advance at least once for resolvable motion. At machine coordinate
    # resolution, requiring an update or a strictly decreasing force norm can
    # loop on rounding noise. This exception is relative to coordinates, NOT
    # a small-force threshold or a line-search-alpha test.
    resolution=wp.isfinite(metrics[5]) and metrics[5]<=wp.float64(1.)
    if (status[0] == 0 and (status[5] > 0 or resolution) and metrics[2] == wp.float64(0.)
            and wp.isfinite(metrics[0]) and wp.isfinite(metrics[1])
            and metrics[0] <= velocity_tol * dt and metrics[1] <= transform_tol * dt
            and wp.isfinite(values[1]) and values[1] >= wp.float64(0.)):
        if not use_guard or (wp.isfinite(guard[0]) and (values[1] <= guard[0] or resolution)):
            out[0] = 1


@wp.kernel
def stationary_accept(status: wp.array(dtype=int), feasible: wp.array(dtype=int),
                      trial_stats: wp.array(dtype=wp.float64), metrics: wp.array(dtype=wp.float64),
                      done: wp.array(dtype=int)):
    # Certify the FULL proposed correction before keeping the current iterate.
    # It need not be representably different: no update is applied here.
    if (status[0] == 0 and feasible[0] != 0 and trial_stats[2] == wp.float64(0.)
            and wp.isfinite(trial_stats[0]) and wp.isfinite(trial_stats[1])):
        done[0] = 3
        if metrics[5]<=wp.float64(1.):done[0]=4
        metrics[3] = wp.float64(1.)
        metrics[4] = wp.float64(0.)


@wp.kernel
def unfinished(status: wp.array(dtype=int), done: wp.array(dtype=int), out: wp.array(dtype=int)):
    out[0] = int(status[0] == 0 and done[0] == 0)


@wp.kernel
def report(metrics: wp.array(dtype=wp.float64), done: wp.array(dtype=int), dt: wp.float64,
           out: wp.array2d(dtype=wp.float64)):
    out[0, 18] = metrics[0] / dt
    out[0, 19] = metrics[1] / dt
    out[0, 20] = metrics[4]
    out[0, 21] = wp.float64(done[0])
    out[0, 22] = metrics[3]


@wp.kernel
def start_residual_guard(values: wp.array(dtype=wp.float64), guard: wp.array(dtype=wp.float64)):
    guard[0] = values[1]
    guard[1] = wp.float64(0.)


@wp.kernel
def residual_guard(status: wp.array(dtype=int), values: wp.array(dtype=wp.float64),
                   guard: wp.array(dtype=wp.float64), done: wp.array(dtype=int)):
    # A small correction at x does not certify the gradient at x+p. In
    # particular, a tiny newly active corner can approach the barrier much
    # faster than the absolute displacement tolerance resolves. Do not stop
    # with a residual worse than one already reached in this SAME objective.
    # This is an extra stopping guard, not a monotone line search: accept the
    # energy-decreasing iterate and continue Newton. No force is clipped and
    # no absolute force tolerance is introduced.
    if status[0] == 0:
        r = values[1]
        if not wp.isfinite(r) or r < wp.float64(0.) or not wp.isfinite(guard[0]):
            status[0] = 3
        elif r > guard[0]:
            if done[0] == 1:
                done[0] = 0
                guard[1] += wp.float64(1.)
        else:
            guard[0] = r


@wp.kernel
def report_residual_guard(guard: wp.array(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    out[0, 36] = guard[0]
    out[0, 37] = guard[1]
