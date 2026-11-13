"""Optional backward-Euler kinematic defect indicator, evaluated on CUDA.

BE uses h*v1 as its displacement; trapezoidal quadrature gives h*(v0+v1)/2.
Their difference is a cheap local indicator, not a bound on global/contact
error. Physical states are never extrapolated. Rejected trials are retried
by the existing coupled subdivision mechanism, with all IPC checks retained.
"""
import warp as wp

wp.set_module_options({'enable_backward': False})


@wp.kernel
def reset(report: wp.array2d(dtype=wp.float64)):
    report[0,38]=wp.float64(0.)


@wp.kernel
def measure(v0: wp.array(dtype=wp.vec3d), v1: wp.array(dtype=wp.vec3d),
            mobile: wp.array(dtype=float), dt: wp.float64,
            report: wp.array2d(dtype=wp.float64), status: wp.array(dtype=int)):
    i=wp.tid()
    if mobile[i]>0.:
        defect=wp.float64(.5)*dt*wp.length(v1[i]-v0[i])
        if not wp.isfinite(defect):
            wp.atomic_max(status,0,10)
        else:
            wp.atomic_max(report,0,38,defect)


@wp.kernel
def measure_affine(v0: wp.array(dtype=wp.vec3d), v1: wp.array(dtype=wp.vec3d),
                   owner: wp.array(dtype=int), weights: wp.array(dtype=wp.vec4d),
                   dt: wp.float64, report: wp.array2d(dtype=wp.float64),
                   status: wp.array(dtype=int)):
    i=wp.tid();b=owner[i]
    if b>=0:
        dv=wp.vec3d(0.)
        for j in range(4):dv+=weights[i][j]*(v1[4*b+j]-v0[4*b+j])
        defect=wp.float64(.5)*dt*wp.length(dv)
        if not wp.isfinite(defect):wp.atomic_max(status,0,10)
        else:wp.atomic_max(report,0,38,defect)


@wp.kernel
def check(report: wp.array2d(dtype=wp.float64), status: wp.array(dtype=int),
          tolerance: wp.float64):
    if status[0]==0 and report[0,38]>tolerance:
        status[0]=15
