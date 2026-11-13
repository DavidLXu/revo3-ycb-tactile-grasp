# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 newton-tactile-simulation contributors
# SPDX-License-Identifier: Apache-2.0
"""Warp 1.16 CG updates with a persistent device mask for grouped graph loops.

Numerical updates match Warp's kernels. The extra mask freezes x/r/p after
convergence, a nonfinite residual or the EXACT iteration budget, even when
the captured group has unused tail iterations. No oracle iteration counts.
"""
from typing import Any
import warp as wp
from warp._src.optim.linear import _find_entry_batch
wp.set_module_options({'enable_backward':False})


@wp.kernel
def update_xr(tol:wp.array(dtype=Any),resid:wp.array(dtype=Any),rz_old:wp.array(dtype=Any),
        p_Ap:wp.array(dtype=Any),x:wp.array(dtype=Any),r:wp.array(dtype=Any),p:wp.array(dtype=Any),
        Ap:wp.array(dtype=Any),batch_offsets:wp.array(dtype=int),dofs_per_entry:int,active:wp.array(dtype=int)):
    if active[0]==0:return
    i=wp.tid();bid=_find_entry_batch(i,batch_offsets,dofs_per_entry)
    if bid>=0:
        alpha=wp.where(resid[bid]>tol[bid],rz_old[bid]/p_Ap[bid],rz_old.dtype(0.0))
        x[i]=x[i]+alpha*p[i]
        r[i]=r[i]-alpha*Ap[i]


@wp.kernel
def update_p(tol:wp.array(dtype=Any),resid_new:wp.array(dtype=Any),rz_old:wp.array(dtype=Any),
        rz_new:wp.array(dtype=Any),z:wp.array(dtype=Any),p:wp.array(dtype=Any),
        batch_offsets:wp.array(dtype=int),dofs_per_entry:int,active:wp.array(dtype=int)):
    if active[0]==0:return
    i=wp.tid();bid=_find_entry_batch(i,batch_offsets,dofs_per_entry)
    if bid>=0:
        beta=wp.where(resid_new[bid]>tol[bid],rz_new[bid]/rz_old[bid],rz_old.dtype(0.0))
        p[i]=z[i]+beta*p[i]


@wp.kernel
def next_iteration(it:wp.array(dtype=int),active:wp.array(dtype=int),residual:wp.array(dtype=wp.float64),
        tolerance:wp.array(dtype=wp.float64),maximum:int):
    if active[0]!=0:
        it[0]+=1
        active[0]=int(it[0]<maximum and residual[0]>tolerance[0] and wp.isfinite(residual[0]))


@wp.kernel
def checked_update_xr(tol:wp.array(dtype=wp.float64),resid:wp.array(dtype=wp.float64),
        rz_old:wp.array(dtype=wp.float64),p_Ap:wp.array(dtype=wp.float64),
        x:wp.array(dtype=wp.vec3d),r:wp.array(dtype=wp.vec3d),p:wp.array(dtype=wp.vec3d),
        Ap:wp.array(dtype=wp.vec3d),active:wp.array(dtype=int),failure:wp.array(dtype=int)):
    # Single-system ABD path. Do not divide by nonpositive curvature, nor
    # introduce an absolute cutoff for a small but valid positive system.
    if active[0]==0 or failure[0]!=0:return
    i=wp.tid()
    if resid[0]>tol[0]:
        if (not wp.isfinite(p_Ap[0]) or p_Ap[0]<=wp.float64(0.) or
                not wp.isfinite(rz_old[0]) or rz_old[0]<=wp.float64(0.)):
            wp.atomic_max(failure,0,1)
        else:
            alpha=rz_old[0]/p_Ap[0]
            x[i]+=alpha*p[i]
            r[i]-=alpha*Ap[i]


@wp.kernel
def checked_next_iteration(it:wp.array(dtype=int),active:wp.array(dtype=int),
        residual:wp.array(dtype=wp.float64),tolerance:wp.array(dtype=wp.float64),
        maximum:int,failure:wp.array(dtype=int)):
    if active[0]!=0:
        it[0]+=1
        active[0]=int(failure[0]==0 and it[0]<maximum and residual[0]>tolerance[0]
                      and wp.isfinite(residual[0]))
