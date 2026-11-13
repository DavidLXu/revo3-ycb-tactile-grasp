"""Rotation-aware trial endpoints for detached, unconstrained affine bodies.

For dA, W = skew(dA A^-1), use exp(alpha W)(A+alpha(dA-WA)).
Its derivative at zero is the ORIGINAL checked Newton direction. For a pure
rotation it preserves A^T A exactly. This changes trial endpoints only, not
energy, inertia, constitutive parameters, PCG or convergence tolerances.
Each accepted update is still a straight affine segment audited by CCD. A
linear CCD fraction must NOT be reused along this nonlinear family of endpoints.
"""
import warp as wp
from . import abd_gpu as ak

wp.set_module_options({'enable_backward':False})

@wp.func
def endpoint(a:wp.mat33d,d:wp.mat33d,alpha:wp.float64):
    v=d*wp.inverse(a);w=(v-wp.transpose(v))*wp.float64(.5)
    angle=alpha*wp.length(wp.vec3d(w[2,1],w[0,2],w[1,0]))
    z=alpha*w;s=wp.float64(1.);c=wp.float64(.5)
    if wp.abs(angle)<wp.float64(1e-4):
        aa=angle*angle
        s=wp.float64(1.)-aa/wp.float64(6.)+aa*aa/wp.float64(120.)
        c=wp.float64(.5)-aa/wp.float64(24.)+aa*aa/wp.float64(720.)
    else:
        s=wp.sin(angle)/angle;c=(wp.float64(1.)-wp.cos(angle))/(angle*angle)
    return (wp.identity(n=3,dtype=wp.float64)+s*z+c*z*z)*(a+alpha*(d-w*a))

@wp.kernel
def contact_bodies(ids:wp.array(dtype=wp.vec4i),count:wp.array(dtype=int),
                   owner:wp.array(dtype=int),active:wp.array(dtype=int)):
    i=wp.tid()
    if i<count[0]:
        for j in range(4):
            k=ids[i][j]
            if k>=0:
                b=owner[k]
                if b>=0:wp.atomic_max(active,b,1)

@wp.kernel
def eligibility(direction:wp.array(dtype=wp.vec3d),radius:wp.array(dtype=wp.float64),
                axes:wp.array(dtype=wp.vec3d),active:wp.array(dtype=int),n:int,
                eligible:wp.array(dtype=int),linear_ccd:wp.array(dtype=int)):
    b=wp.tid();i=n+4*b
    d=wp.matrix_from_cols(direction[i+1],direction[i+2],direction[i+3])/radius[b]
    # Small corrections retain the original Newton path and stopping certificate.
    eligible[b]=int(active[b]==0 and wp.dot(axes[b],axes[b])==wp.float64(0.)
                    and wp.ddot(d,d)>wp.float64(1e-6))
    if eligible[b]!=0:wp.atomic_min(linear_ccd,0,0)

@wp.kernel
def change(base:wp.array(dtype=wp.vec3d),direction:wp.array(dtype=wp.vec3d),
           gradient:wp.array(dtype=wp.vec3d),radius:wp.array(dtype=wp.float64),
           eligible:wp.array(dtype=int),values:wp.array(dtype=wp.float64),n:int,
           out:wp.array(dtype=wp.vec3d),changed:wp.array(dtype=int)):
    b=wp.tid();i=n+4*b;r=radius[b]
    if eligible[b]!=0:
        a=ak.affine(base,i,r)
        d=wp.matrix_from_cols(direction[i+1],direction[i+2],direction[i+3])/r
        trial=endpoint(a,d,values[2]);work=wp.float64(0.);linear=wp.float64(0.)
        for j in range(3):
            work+=wp.dot(gradient[i+j+1],r*wp.vec3d(trial[0,j]-a[0,j],trial[1,j]-a[1,j],trial[2,j]-a[2,j]))
            linear+=wp.dot(gradient[i+j+1],values[2]*direction[i+j+1])
        # Keep a useful local descent model. The COMPLETE gradient-work test,
        # objective (including frozen friction), Armijo and CCD run afterward.
        if wp.isfinite(work) and work<=wp.max(linear,wp.float64(0.)):
            for j in range(3):out[i+j+1]=r*wp.vec3d(trial[0,j],trial[1,j],trial[2,j])
            wp.atomic_max(changed,0,1)

@wp.kernel
def trial_stats(base:wp.array(dtype=wp.vec3d),out:wp.array(dtype=wp.vec3d),
                direction:wp.array(dtype=wp.vec3d),gradient:wp.array(dtype=wp.vec3d),stats:wp.array(dtype=wp.float64)):
    i=wp.tid();d=out[i]-base[i]
    wp.atomic_add(stats,0,wp.dot(gradient[i],d));wp.atomic_add(stats,1,wp.dot(d,d))
    for j in range(3):
        if not wp.isfinite(out[i][j]) or not wp.isfinite(direction[i][j]):wp.atomic_add(stats,2,wp.float64(1.))

@wp.kernel
def observe(changed:wp.array(dtype=int),status:wp.array(dtype=int),
            metrics:wp.array(dtype=wp.float64),counts:wp.array(dtype=int)):
    if changed[0]!=0:
        counts[0]+=1
        if status[8]==1:counts[1]+=1
        # A nonlinear endpoint does not certify the full LINEAR Newton path.
        # Require another iteration with the ordinary small-correction test.
        metrics[3]=wp.float64(0.)

@wp.kernel
def report(counts:wp.array(dtype=int),out:wp.array2d(dtype=wp.float64)):
    out[0,46]=wp.float64(counts[0]);out[0,47]=wp.float64(counts[1])
