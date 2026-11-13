"""GPU-only on-demand elastic projection; retry at the SAME Newton iterate.

Only tangent failures (preconditioner or linear direction) permit a retry.
Collision, energy, geometry and nonlinear acceptance errors are never cleared.
The projected retry still passes the original complete linear/descent checks.
"""
import warp as wp
wp.set_module_options({'enable_backward':False})


@wp.kernel
def begin(state:wp.array(dtype=int),counts:wp.array(dtype=int)):
    state[0]=0
    counts[0]+=1


@wp.kernel
def recover(status:wp.array(dtype=int),stats:wp.array(dtype=wp.float64),kind:int,
            state:wp.array(dtype=int),flag:wp.array(dtype=int),counts:wp.array(dtype=int)):
    flag[0]=0
    if (kind==4 or kind==5) and state[0]==0 and status[0]==kind:
        status[0]=0
        stats[2]=wp.float64(0.)
        state[0]=1
        flag[0]=1
        counts[kind-3]+=1


@wp.kernel
def report(counts:wp.array(dtype=int),out:wp.array2d(dtype=wp.float64)):
    for i in range(3):out[0,39+i]=wp.float64(counts[i])
