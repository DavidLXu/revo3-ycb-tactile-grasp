# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 newton-tactile-simulation contributors
# SPDX-License-Identifier: Apache-2.0
"""Warp CG with preallocated loop controls for nested conditional graphs.

The numerical update is from warp/_src/optim/linear.py (Apache-2.0).
Loop controls are preallocated for the outer Newton graph. The optional ABD
on-demand path also detects nonpositive curvature before applying an update.
Positive-curvature CG updates and the relative residual criterion are unchanged.
Version compatibility is checked by SolverIPCVBD before importing this module.
"""
import warp as wp
from warp._src.optim.linear import CG, _initialize_absolute_tolerance, _cg_kernel_1, _cg_kernel_2
from .ipc_gpu_control import cg_condition
from . import ipc_gpu_grouped_cg as grouped


class NestedCG(CG):
    def _allocate(self):
        # Warp 1.17 added this constructor option. Our guarded/grouped loop
        # does not implement restart cycles; silently accepting it is wrong.
        if getattr(self, "_restart", None) is not None:
            raise ValueError("NestedCG does not support restart; use Warp CG for restart experiments")
        super()._allocate()
        # Retain the established allocations. Sharing GA's base-class buffer
        # only saves setup allocations, and the trajectory comparison has not
        # cleared that experimental change (see the GA reconciliation report).
        self._iterations = wp.zeros(1, dtype=int, device=self._device)
        self._condition = wp.zeros(1, dtype=int, device=self._device)
        self.curvature_failure = wp.zeros(1, dtype=int, device=self._device)

    def _run(self, A, b, x, M):
        group_size=getattr(self,'group_size',1)
        if isinstance(group_size,bool) or not isinstance(group_size,int) or group_size not in (1,2,4,8):
            raise ValueError('CG group size must be 1, 2, 4 or 8')
        device = self._device
        batch_offsets = A.batch_offsets
        checked=getattr(self,'check_positive_curvature',False)
        fused_product=getattr(self,'fused_product_dot',None)
        fused_preconditioner=getattr(self,'fused_precondition_dot',None)
        if fused_preconditioner is not None and (M is None or batch_offsets is not None or x.dtype!=wp.vec3d):
            raise ValueError('fused CG preconditioner requires a preconditioned single vec3d system')
        if fused_product is not None and (batch_offsets is not None or x.dtype!=wp.vec3d):
            raise ValueError('fused CG product requires a single vec3d system')
        if checked and (batch_offsets is not None or x.dtype!=wp.vec3d):
            raise ValueError('checked CG requires a single vec3d system')
        self.curvature_failure.zero_()
        dofs_per_entry = self._dofs_per_entry
        tiled_dot = self._tiled_dot
        p_and_Ap = self._p_and_Ap
        r_and_z_buf = self._r_and_z_buf
        r_repeated = self._r_repeated

        if M is None:
            # without preconditioner r == z
            r_and_z = r_repeated
            rz_new = tiled_dot.col(0)
        else:
            r_and_z = r_and_z_buf
            rz_new = tiled_dot.col(1)

        r, z = r_and_z[0], r_and_z[1]
        r_norm_sq = tiled_dot.col(0)

        p, Ap = p_and_Ap[0], p_and_Ap[1]
        rz_old, atol_sq = self._residuals[0], self._residuals[1]

        # Not strictly necessary, but makes it more robust to user-provided LinearOperators
        Ap.zero_()
        z.zero_()

        # Initialize tolerance from right-hand-side norm
        _initialize_absolute_tolerance(b, self._tol, self._atol, tiled_dot, atol_sq)
        # Initialize residual
        # The IPC caller explicitly zeros x before each solve. In that case
        # r=b exactly: do not launch the entire sparse/contact operator on 0.
        # Generic callers retain the original nonzero-initial-guess behavior.
        if getattr(self, "zero_initial_guess", False):
            r.assign(b)
        else:
            A.matvec(x, b, r, alpha=-1.0, beta=1.0)

        def update_rr_rz():
            # z = M r
            if M is None:
                tiled_dot.compute(r, r)
            elif fused_preconditioner is not None:
                fused_preconditioner(r,z,tiled_dot)
            else:
                M.matvec(r, z, z, alpha=1.0, beta=0.0)
                tiled_dot.compute(r_repeated, r_and_z_buf)

        update_rr_rz()
        p.assign(z)

        def do_iteration():
            rz_old.assign(rz_new)

            # Ap = A * p;
            p_Ap = tiled_dot.col(1)
            if fused_product is None:
                A.matvec(p, Ap, Ap, alpha=1, beta=0)
                tiled_dot.compute(p, Ap, col_offset=1)
            else:
                fused_product(p,Ap,p_Ap)

            if checked:
                wp.launch(grouped.checked_update_xr,x.shape[0],
                    [atol_sq,r_norm_sq,rz_old,p_Ap,x,r,p,Ap,self._condition,self.curvature_failure],device=device)
            else:wp.launch(
                kernel=_cg_kernel_1 if group_size==1 else grouped.update_xr,
                dim=x.shape[0],
                device=device,
                inputs=[atol_sq, r_norm_sq, rz_old, p_Ap, x, r, p, Ap, batch_offsets, dofs_per_entry]+([] if group_size==1 else [self._condition]),
            )

            update_rr_rz()

            wp.launch(
                kernel=_cg_kernel_2 if group_size==1 else grouped.update_p,
                dim=z.shape[0],
                device=device,
                inputs=[atol_sq, r_norm_sq, rz_old, rz_new, z, p, batch_offsets, dofs_per_entry]+([] if group_size==1 else [self._condition]),
            )

        self._iterations.fill_(-1)
        def condition():
            wp.launch(cg_condition, 1,
                [self._iterations, self._condition, r_norm_sq, atol_sq, self._maxiter], device=device)
        condition()
        def cycle():
            for _ in range(group_size):
                do_iteration()
                if checked:
                    wp.launch(grouped.checked_next_iteration,1,[self._iterations,self._condition,
                        r_norm_sq,atol_sq,self._maxiter,self.curvature_failure],device=device)
                elif group_size==1:condition()
                else:
                    wp.launch(grouped.next_iteration,1,[self._iterations,self._condition,
                        r_norm_sq,atol_sq,self._maxiter],device=device)
        wp.capture_while(self._condition, cycle)
        return self._iterations, r_norm_sq, atol_sq
