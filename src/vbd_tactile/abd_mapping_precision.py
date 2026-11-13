"""Compensate world-vertex rounding when comparing tiny affine energy changes.

The optimizer differentiates x=B*u, while materializing world vertices rounds
that map. Near equilibrium, this roundoff work can exceed the entire Newton
decrease. Recover the mapping error with compensated products/sums and apply
the first-order energy correction at BOTH endpoints. The omitted term is
second order in floating-point mapping error, not in the Newton displacement.
This does not alter positions, contact forces, residual tolerances or CCD.
"""
import warp as wp

# TwoSum must observe each rounded operation. Automatic multiply/add fusion
# invalidates its error terms on CUDA; the product residual below uses an
# explicit fma with a deliberately rounded product instead.
wp.set_module_options({'enable_backward':False,'fuse_fp':False})


@wp.func_native('return ::fma(a, b, c);')
def fused_multiply_add(a:wp.float64,b:wp.float64,c:wp.float64)->wp.float64:
    ...


@wp.func
def mapping_error(u:wp.array(dtype=wp.vec3d),base:int,w:wp.vec4d,x:wp.vec3d):
    error=wp.vec3d(0)
    for k in range(3):
        hi=wp.float64(0);lo=wp.float64(0)
        for j in range(4):
            product=w[j]*u[base+j][k]
            product_low=fused_multiply_add(w[j],u[base+j][k],-product)
            total=hi+product
            part=total-hi
            lo+=(hi-(total-part))+(product-part)+product_low
            hi=total
        error[k]=(hi-x[k])+lo
    return error


@wp.kernel
def energy_correction(a:wp.array(dtype=wp.vec3d),b:wp.array(dtype=wp.vec3d),
                      x0:wp.array(dtype=wp.vec3d),x1:wp.array(dtype=wp.vec3d),
                      g0:wp.array(dtype=wp.vec3d),g1:wp.array(dtype=wp.vec3d),
                      owner:wp.array(dtype=int),weights:wp.array(dtype=wp.vec4d),n:int,
                      increment:wp.array(dtype=wp.float64)):
    i=wp.tid();body=owner[i]
    if body>=0:
        base=n+4*body
        e0=mapping_error(a,base,weights[i],x0[i])
        e1=mapping_error(b,base,weights[i],x1[i])
        wp.atomic_add(increment,0,wp.dot(g1[i],e1)-wp.dot(g0[i],e0))
