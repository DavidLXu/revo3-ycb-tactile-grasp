"""Joint PCG products without per-iteration full-vector scatter/gather passes."""
import warp as wp
from .ipc_gpu_math import mat12
wp.set_module_options({'enable_backward':False})


@wp.kernel
def base_product(x:wp.array(dtype=wp.vec3d),offsets:wp.array(dtype=int),counts:wp.array(dtype=int),
                 columns:wp.array(dtype=int),values:wp.array3d(dtype=wp.float64),
                 body:wp.array(dtype=mat12),n:int,out:wp.array2d(dtype=wp.float64)):
    row,k=wp.tid();value=wp.float64(0)
    if row<n:
        for slot in range(offsets[row],offsets[row]+counts[row]):
            v=x[columns[slot]]
            value+=values[slot,k,0]*v[0]+values[slot,k,1]*v[1]+values[slot,k,2]*v[2]
    else:
        b=(row-n)//4;local=(row-n)%4
        for j in range(4):
            for l in range(3):value+=body[b][3*local+k,3*j+l]*x[n+4*b+j][l]
    out[row,k]=value


@wp.kernel
def contact_product(full:wp.array(dtype=wp.vec3d),ids:wp.array(dtype=wp.vec4i),
                    indices:wp.array(dtype=int),count:wp.array(dtype=int),h:wp.array3d(dtype=wp.float64),
                    mobile:wp.array(dtype=float),owner:wp.array(dtype=int),weights:wp.array(dtype=wp.vec4d),
                    n:int,lanes:int,out:wp.array2d(dtype=wp.float64)):
    lane,local=wp.tid()
    for slot in range(lane,count[0],lanes):
        i=indices[slot];ind=ids[i];row=ind[local//3]
        if row>=0:
            if mobile[row]>0.:
                value=wp.float64(0)
                for j in range(4):
                    col=ind[j]
                    if col>=0:
                        if mobile[col]>0.:
                            for l in range(3):value+=h[i,local,3*j+l]*full[col][l]
                b=owner[row]
                if b>=0:
                    for j in range(4):wp.atomic_add(out,n+4*b+j,local%3,weights[row][j]*value)
                else:wp.atomic_add(out,row,local%3,value)
