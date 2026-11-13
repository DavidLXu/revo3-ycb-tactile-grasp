"""Device controls for distinct friction update policies.

Strict and per-Newton policies retain the refreshed residual/force-change
check. Finite lagged passes solve their frozen objective to strict equilibrium
but deliberately do not assert refreshed fixed-point consistency.
"""
import warp as wp
wp.set_module_options({'enable_backward':False})


@wp.kernel
def backtrack_budget(status:wp.array(dtype=int),condition:wp.array(dtype=int),maximum:int):
    # Abort, never accept, an expensive attempt. The normal transactional
    # retry uses a smaller physical timestep and retains all acceptance tests.
    if status[0]==0 and status[9]>=maximum:
        status[0]=14
        condition[0]=0


@wp.kernel
def update_tolerance(change:wp.array(dtype=wp.float64),strict:wp.float64,out:wp.array(dtype=wp.float64)):
    value=wp.max(strict,wp.min(wp.float64(100)*strict,wp.float64(.1)*change[0]))
    if not wp.isfinite(change[0]):value=strict
    out[0]=value


@wp.kernel
def relative_tolerance(error:wp.array(dtype=wp.float64),index:int,strict:wp.float64,
                       ratio:wp.float64,out:wp.array(dtype=wp.float64)):
    # An inexact frozen-friction subproblem may track the current outer
    # error. Publication still checks the unchanged strict residual AND
    # all nodal force changes in ipc_gpu_friction_iteration.check.
    value=strict
    if wp.isfinite(error[index]) and error[index]>=wp.float64(0.):
        value=wp.max(strict,ratio*error[index])
    out[0]=value


@wp.kernel
def condition(s:wp.array(dtype=int),v:wp.array(dtype=wp.float64),tolerance:wp.array(dtype=wp.float64),
              strict:wp.float64,maximum:int,out:wp.array(dtype=int)):
    out[0]=0
    if s[0]==0:
        if not wp.isfinite(tolerance[0]) or not wp.isfinite(strict) or strict<=wp.float64(0) or tolerance[0]<strict or maximum<1:
            s[0]=29
        elif not wp.isfinite(v[1]) or v[1]<wp.float64(0):s[0]=3
        elif v[1]>tolerance[0]:
            if s[5]>=maximum:s[0]=7
            else:out[0]=1


@wp.kernel
def finite_pass_condition(status:wp.array(dtype=int),count:wp.array(dtype=int),
                          passes:int,out:wp.array(dtype=int)):
    # count is refreshes AFTER the initial solve. Never refresh after the
    # last solve: that would publish forces from an unsolved objective.
    out[0]=int(status[0]==0 and count[0]<passes-1)


@wp.kernel
def record_refresh(status:wp.array(dtype=int),change:wp.array(dtype=wp.float64),
                   count:wp.array(dtype=int)):
    count[0]+=1
    if status[0]==0 and not wp.isfinite(change[0]):status[0]=3


@wp.kernel
def newton_frame_start(status:wp.array(dtype=int),mark:wp.array(dtype=int)):
    mark[0]=status[5]


@wp.kernel
def one_newton_condition(status:wp.array(dtype=int),values:wp.array(dtype=wp.float64),
                         mark:wp.array(dtype=int),strict:wp.float64,maximum:int,
                         out:wp.array(dtype=int)):
    out[0]=0
    if status[0]==0:
        if not wp.isfinite(values[1]) or values[1]<wp.float64(0.):status[0]=3
        elif values[1]>strict and status[5]==mark[0]:
            if status[5]>=maximum:status[0]=7
            else:out[0]=1


@wp.kernel
def final_pass_tolerance(count:wp.array(dtype=int),passes:int,strict:wp.float64,
                         tolerance:wp.array(dtype=wp.float64)):
    # A finite policy has no later fixed-point check to protect publication.
    # Therefore its LAST frozen solve must always use the strict target.
    if count[0]>=passes-1:tolerance[0]=strict
