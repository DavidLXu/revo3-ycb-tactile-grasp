"""Device-only rejection of unevaluable surface-contact line-search trials.

Only trial objectives may be treated as +infinity. Initial/current accepted
states still propagate every capacity, domain and quadrature failure.
"""
import warp as wp
wp.set_module_options({'enable_backward':False})

@wp.kernel
def reject_trial(status:wp.array(dtype=int),stats:wp.array(dtype=wp.float64),counter:wp.array(dtype=int)):
    code=status[0]
    if code==53 or code==54 or code==55 or code==56:
        stats[0]=wp.float64(wp.inf)
        stats[2]=wp.float64(1)
        status[0]=0
        counter[0]+=1
