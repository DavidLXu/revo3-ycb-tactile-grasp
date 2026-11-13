"""Fixed-point IPC friction updates, entirely inside the captured GPU solve.

The normal load and closest-point frame are refreshed at the current iterate;
relative displacement always remains relative to the start of the substep.
Both free-node equilibrium and the change of ALL nodal contact forces must
meet the existing absolute force tolerance. Fixed reactions are checked too.
"""
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def save_forces(a: wp.array(dtype=wp.vec3d), b: wp.array(dtype=wp.vec3d),
                previous: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    previous[i] = a[i]+b[i]


@wp.kernel
def force_change(a: wp.array(dtype=wp.vec3d), b: wp.array(dtype=wp.vec3d),
                 previous: wp.array(dtype=wp.vec3d), change: wp.array(dtype=wp.float64)):
    i = wp.tid()
    error = wp.length(a[i]+b[i]-previous[i])
    if not wp.isfinite(error):
        error = wp.float64(wp.inf)
    wp.atomic_max(change, 0, error)


@wp.kernel
def start(status: wp.array(dtype=int), condition: wp.array(dtype=int)):
    condition[0] = int(status[0] == 0)


@wp.kernel
def check(status: wp.array(dtype=int), values: wp.array(dtype=wp.float64),
          change: wp.array(dtype=wp.float64), count: wp.array(dtype=int),
          condition: wp.array(dtype=int), maximum: int, tolerance: wp.float64):
    count[0] += 1
    condition[0] = 0
    if status[0] == 0:
        if not wp.isfinite(change[0]):
            status[0] = 3
        elif change[0] > tolerance or values[1] > tolerance:
            if count[0] >= maximum:
                status[0] = 12
            else:
                condition[0] = 1


@wp.kernel
def report(count: wp.array(dtype=int), change: wp.array(dtype=wp.float64),
           output: wp.array2d(dtype=wp.float64)):
    output[0, 16] = wp.float64(count[0])
    output[0, 17] = change[0]
