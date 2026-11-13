"""One CUDA warp cooperates on each 12x12 cyclic Jacobi projection.

The sweep count, pivot order, thresholds and reconstruction match psd12.
Rows of each pivot rotation and entries of reconstruction run in parallel.
The wrapper MUST launch whole 32-thread blocks; all lanes follow the same
queue entry and barriers. No contact or eigenmode is omitted.
"""
import warp as wp
from .ipc_gpu_math import mat12
wp.set_module_options({'enable_backward':False})


@wp.func_native(r'''
#ifdef __CUDA_ARCH__
    const int lane = threadIdx.x;
    __shared__ double a[144];
    __shared__ double vectors[144];
    __shared__ double scale, cosine, sine;
    __shared__ int converged, rotate;
    const int index = pending.data[slot];
    for (int j = lane; j < 144; j += 32) {
        int r = j / 12, c = j % 12;
        a[j] = (hessians.data[index].data[r][c] + hessians.data[index].data[c][r]) * .5;
        vectors[j] = (r == c) ? 1. : 0.;
    }
    __syncwarp();
    for (int sweep = 0; sweep < 16; ++sweep) {
        if (lane == 0) {
            scale = 0.; double off = 0.;
            for (int i = 0; i < 12; ++i) {
                scale = wp::max(scale, ::fabs(a[12*i+i]));
                for (int j = i+1; j < 12; ++j) off = wp::max(off, ::fabs(a[12*i+j]));
            }
            converged = off <= 1.e-13 * wp::max(scale, 1.e-30);
        }
        __syncwarp();
        if (converged) break;
        for (int p = 0; p < 11; ++p) {
            for (int q = p+1; q < 12; ++q) {
                if (lane == 0) {
                    const double apq = a[12*p+q];
                    rotate = ::fabs(apq) > 1.e-15 * wp::max(scale, 1.e-30);
                    if (rotate) {
                        const double tau = (a[12*q+q]-a[12*p+p])/(2.*apq);
                        const double sign = tau < 0. ? -1. : 1.;
                        const double t = sign/(::fabs(tau)+::sqrt(1.+tau*tau));
                        cosine = 1./::sqrt(1.+t*t); sine = t*cosine;
                        a[12*p+p] = a[12*p+p]-t*apq;
                        a[12*q+q] = a[12*q+q]+t*apq;
                        a[12*p+q] = 0.; a[12*q+p] = 0.;
                    }
                }
                __syncwarp();
                if (rotate && lane < 12) {
                    const int k = lane;
                    const double c = cosine, s = sine;
                    if (k != p && k != q) {
                        const double kp = a[12*k+p], kq = a[12*k+q];
                        a[12*k+p] = c*kp-s*kq; a[12*p+k] = a[12*k+p];
                        a[12*k+q] = s*kp+c*kq; a[12*q+k] = a[12*k+q];
                    }
                    const double vp = vectors[12*k+p], vq = vectors[12*k+q];
                    vectors[12*k+p] = c*vp-s*vq;
                    vectors[12*k+q] = s*vp+c*vq;
                }
                __syncwarp();
            }
        }
    }
    for (int j = lane; j < 144; j += 32) {
        const int r = j/12, c = j%12;
        double value = 0.;
        for (int k = 0; k < 12; ++k)
            value += wp::max(a[12*k+k],0.)*vectors[12*r+k]*vectors[12*c+k];
        hessians.data[index].data[r][c] = value;
    }
    __syncwarp();
#endif
    return 0;
''')
def project(hessians:wp.array(dtype=mat12),pending:wp.array(dtype=int),slot:int)->int:
    ...


@wp.kernel
def queued(hessians:wp.array(dtype=mat12),pending:wp.array(dtype=int),count:wp.array(dtype=int),lanes:int):
    block=wp.tid()//32
    for slot in range(block,count[0],lanes):
        project(hessians,pending,slot)


def project_queue(hessians,pending,count,device):
    device=wp.get_device(device)
    if not device.is_cuda:raise ValueError('warp-cooperative projection requires CUDA')
    if (hessians.dtype!=mat12 or pending.dtype!=wp.int32 or count.dtype!=wp.int32
            or hessians.ndim!=1 or pending.shape!=hessians.shape or count.shape!=(1,)
            or not all(a.is_contiguous and a.device==device for a in (hessians,pending,count))):
        raise ValueError('projection requires matching contiguous device arrays')
    lanes=min(len(pending),256)
    if lanes:
        wp.launch(queued,lanes*32,[hessians,pending,count,lanes],block_dim=32,device=device)
