"""Separate physical FEM and affine residual blocks without a host readback."""
import warp as wp
wp.set_module_options({'enable_backward':False})


@wp.kernel
def measure(gradient:wp.array(dtype=wp.vec3d),mass:wp.array(dtype=float),n:int,
            out:wp.array2d(dtype=wp.float64)):
    i=wp.tid()
    if i<n:
        if mass[i]>0.:
            wp.atomic_max(out,0,42,wp.length(gradient[i]))
    else:
        value=wp.length(gradient[i])
        wp.atomic_max(out,0,43,value)
        if (i-n)%4==0:wp.atomic_max(out,0,44,value)
        else:wp.atomic_max(out,0,45,value)


@wp.kernel
def reset(out:wp.array2d(dtype=wp.float64)):
    for i in range(4):out[0,42+i]=wp.float64(0.)
