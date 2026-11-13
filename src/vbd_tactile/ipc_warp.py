"""Reusable Warp sphere contact, CCD, and tactile reduction buffers.

Particle coordinates stay on the device. Distance calculations use float64 on
float32 Newton coordinates, including rechecking the *rounded* accepted path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing import Any
from .ipc_precision import position_overloads
import warp as wp
from .ipc_precision import position_kernel

from .ipc import points_and_center
from .types import TactileFrame


@wp.func
def _finite3(x: wp.vec3d):
    return wp.isfinite(x[0]) and wp.isfinite(x[1]) and wp.isfinite(x[2])


@wp.func
def _swept_gap(
    x0: wp.vec3d, x1: wp.vec3d, c0: wp.vec3d, c1: wp.vec3d, radius: wp.float64
):
    r0 = x0 - c0
    v = (x1 - c1) - r0
    den = wp.dot(v, v)
    t = wp.float64(0.0)
    if den > wp.float64(0.0):
        t = wp.clamp(-wp.dot(r0, v) / den, wp.float64(0.0), wp.float64(1.0))
    return wp.length(r0 + t * v) - radius


@position_overloads("p")
@wp.kernel
def _evaluate(
    p: wp.array(dtype=Any),
    radii: wp.array(dtype=wp.float64),
    c: wp.vec3d,
    radius: wp.float64,
    d: wp.float64,
    k: wp.float64,
    min_gap: wp.float64,
    forces: wp.array(dtype=wp.vec3),
    gaps: wp.array(dtype=wp.float64),
    normals: wp.array(dtype=wp.vec3),
    diag: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    delta = wp.vec3d(p[i]) - c
    length = wp.length(delta)
    gap = length - radius - radii[i]
    gaps[i] = gap
    forces[i] = wp.vec3(0.0)
    normals[i] = wp.vec3(0.0)
    if not _finite3(delta) or not wp.isfinite(gap) or gap <= min_gap:
        wp.atomic_add(diag, 3, wp.float64(1.0))
    else:
        wp.atomic_min(diag, 0, gap)
        normal = delta / length
        normals[i] = wp.vec3(normal)
        if gap < d:
            a = gap - d
            lg = wp.log(gap / d)
            first = -wp.float64(2.0) * a * lg - a * a / gap
            force = wp.vec3(-k * first * normal)
            if not _finite3(wp.vec3d(force)):
                wp.atomic_add(diag, 3, wp.float64(1.0))
            else:
                forces[i] = force
                wp.atomic_max(diag, 1, wp.length(wp.vec3d(force)))
                wp.atomic_add(diag, 2, -k * a * a * lg)


@wp.kernel
def _ccd_limits(
    old: wp.array(dtype=wp.vec3),
    new: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float64),
    c0: wp.vec3d,
    c1: wp.vec3d,
    radius: wp.float64,
    min_gap: wp.float64,
    safety: wp.float64,
    iterations: int,
    alpha: wp.array(dtype=wp.float64),
    diag: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    x0, x1 = wp.vec3d(old[i]), wp.vec3d(new[i])
    scale = wp.max(wp.length(x0), wp.length(x1))
    scale = wp.max(scale, wp.max(wp.length(c0), wp.length(c1)))
    scale = wp.max(scale, radius + radii[i])
    guard = wp.float64(8.0 * 1.1920928955078125e-7) * scale
    r = radius + radii[i]
    old_gap = wp.length(x0 - c0) - r
    threshold = wp.min(min_gap + guard, min_gap + wp.float64(0.5) * (old_gap - min_gap))
    if not _finite3(x0) or not _finite3(x1) or old_gap <= min_gap:
        wp.atomic_add(diag, 3, wp.float64(1.0))
    elif _swept_gap(x0, x1, c0, c1, r) <= threshold:
        lo, hi = wp.float64(0.0), wp.float64(1.0)
        for it in range(iterations):
            mid = (lo + hi) * wp.float64(0.5)
            if (
                _swept_gap(x0, x0 + mid * (x1 - x0), c0, c0 + mid * (c1 - c0), r)
                > threshold
            ):
                lo = mid
            else:
                hi = mid
        wp.atomic_min(alpha, 0, safety * lo)


@wp.kernel
def _accept(
    old: wp.array(dtype=wp.vec3),
    new: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float64),
    c0: wp.vec3d,
    accepted_c: wp.vec3d,
    radius: wp.float64,
    alpha: wp.float64,
    accepted: wp.array(dtype=wp.vec3),
    diag: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    x0, x1 = wp.vec3d(old[i]), wp.vec3d(new[i])
    rounded = wp.vec3(x0 + alpha * (x1 - x0))
    accepted[i] = rounded
    r = radius + radii[i]
    wp.atomic_min(diag, 0, wp.length(wp.vec3d(rounded) - accepted_c) - r)
    wp.atomic_min(diag, 4, _swept_gap(x0, wp.vec3d(rounded), c0, accepted_c, r))


@wp.kernel
def _velocity(
    old: wp.array(dtype=wp.vec3),
    accepted: wp.array(dtype=wp.vec3),
    dt: float,
    velocity: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    velocity[i] = (accepted[i] - old[i]) / dt


@dataclass(frozen=True)
class WarpStepResult:
    positions: object
    sphere_center: np.ndarray
    alpha: float
    min_gap_m: float
    endpoint_min_gap_m: float


class WarpSphereContact:
    """Workspace for one contact model and a fixed number of particles.

    Output arrays are reused by subsequent calls. All public calls validate
    their compact diagnostics before returning to the caller.
    """

    def __init__(self, contact, count, device="cuda:0"):
        self.contact = contact
        self.count = int(count)
        self.device = wp.get_device(device)
        self.radii = wp.array(
            contact.radii(count).copy(), dtype=wp.float64, device=self.device
        )
        self.forces = wp.empty(count, dtype=wp.vec3, device=self.device)
        self.gaps = wp.empty(count, dtype=wp.float64, device=self.device)
        self.normals = wp.empty(count, dtype=wp.vec3, device=self.device)
        self.accepted = wp.empty(count, dtype=wp.vec3, device=self.device)
        self.alpha = wp.ones(1, dtype=wp.float64, device=self.device)
        self.diag = wp.empty(5, dtype=wp.float64, device=self.device)

    def _check(self, positions):
        if (
            positions.dtype != wp.vec3
            or positions.shape != (self.count,)
            or positions.device != self.device
        ):
            raise ValueError(
                "positions must be a vec3 array matching workspace count and device"
            )

    def _reset(self):
        self.diag.assign(np.array([np.inf, 0.0, 0.0, 0.0, np.inf], dtype=np.float64))

    def evaluate(self, positions, center):
        if (
            positions.dtype not in (wp.vec3, wp.vec3d)
            or positions.shape != (self.count,)
            or positions.device != self.device
        ):
            raise ValueError(
                "positions must be vec3/vec3d with matching count and device"
            )
        _, center = points_and_center(np.empty((0, 3)), center)
        self._reset()
        c = self.contact
        if self.count:
            wp.launch(
                position_kernel(_evaluate, positions),
                self.count,
                inputs=[
                    positions,
                    self.radii,
                    wp.vec3d(*center),
                    c.sphere_radius_m,
                    c.activation_distance_m,
                    c.barrier_stiffness,
                    c.min_gap_m,
                    self.forces,
                    self.gaps,
                    self.normals,
                    self.diag,
                ],
                device=self.device,
            )
        diagnostics = self.diag.numpy()
        if diagnostics[3]:
            raise ValueError("IPC GPU state is invalid or at/beyond the minimum gap")
        return {
            "min_gap_m": float(diagnostics[0]),
            "max_force_N": float(diagnostics[1]),
            "energy_J": float(diagnostics[2]),
        }

    def conservative_step(
        self, old, new, old_center, new_center, *, max_backtracks=60, safety=0.9
    ):
        self._check(old)
        self._check(new)
        _, c0 = points_and_center(np.empty((0, 3)), old_center)
        _, c1 = points_and_center(np.empty((0, 3)), new_center)
        if (
            isinstance(max_backtracks, bool)
            or int(max_backtracks) != max_backtracks
            or max_backtracks < 1
        ):
            raise ValueError("max_backtracks must be a positive integer")
        if not np.isfinite(safety) or not 0 < safety < 1:
            raise ValueError("safety must be in (0,1)")
        self._reset()
        self.alpha.fill_(1.0)
        c = self.contact
        if self.count:
            wp.launch(
                _ccd_limits,
                self.count,
                inputs=[
                    old,
                    new,
                    self.radii,
                    wp.vec3d(*c0),
                    wp.vec3d(*c1),
                    c.sphere_radius_m,
                    c.min_gap_m,
                    safety,
                    int(max_backtracks),
                    self.alpha,
                    self.diag,
                ],
                device=self.device,
            )
        if self.diag.numpy()[3]:
            raise ValueError(
                "IPC CCD inputs are non-finite or have insufficient precision margin"
            )
        alpha = float(self.alpha.numpy()[0])
        center = (c0 + alpha * (c1 - c0)).astype(np.float32).astype(np.float64)
        if self.count:
            wp.launch(
                _accept,
                self.count,
                inputs=[
                    old,
                    new,
                    self.radii,
                    wp.vec3d(*c0),
                    wp.vec3d(*center),
                    c.sphere_radius_m,
                    alpha,
                    self.accepted,
                    self.diag,
                ],
                device=self.device,
            )
        diagnostics = self.diag.numpy()
        if diagnostics[0] <= c.min_gap_m or diagnostics[4] <= c.min_gap_m:
            raise RuntimeError(
                "IPC CCD rounded accepted state/path violates minimum gap"
            )
        return WarpStepResult(
            self.accepted, center, alpha, float(diagnostics[4]), float(diagnostics[0])
        )

    def assign_velocity(self, old, dt, output):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        wp.launch(
            _velocity,
            self.count,
            inputs=[old, self.accepted, dt, output],
            device=self.device,
        )


@wp.func
def _round_even(value: float):
    lo = int(wp.floor(value))
    frac = value - float(lo)
    if frac > 0.5 or (frac == 0.5 and lo % 2 != 0):
        lo += 1
    return lo


@position_overloads("positions")
@wp.kernel
def _project(
    positions: wp.array(dtype=Any),
    forces: wp.array(dtype=wp.vec3),
    origin: wp.vec3,
    u: wp.vec3,
    v: wp.vec3,
    normal: wp.vec3,
    width: float,
    height: float,
    rows: int,
    cols: int,
    sigma: wp.float64,
    radius: int,
    grid: wp.array(dtype=wp.vec3),
    sums: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    p = wp.vec3(positions[i]) - origin
    x, y = wp.dot(p, u), wp.dot(p, v)
    f = forces[i]
    if not _finite3(wp.vec3d(p)) or not _finite3(wp.vec3d(f)):
        wp.atomic_add(sums, 5, wp.float64(1.0))
        return
    load = wp.vec3(wp.dot(f, u), wp.dot(f, v), wp.max(0.0, wp.dot(f, normal)))
    if (
        (f[0] != 0.0 or f[1] != 0.0 or f[2] != 0.0)
        and x >= 0.0
        and x <= width
        and y >= 0.0
        and y <= height
    ):
        du, dv = width / float(cols), height / float(rows)
        cx, cy = x / du - 0.5, y / dv - 0.5
        c0, r0 = _round_even(cx), _round_even(cy)
        nearest_c, nearest_r = wp.clamp(c0, 0, cols - 1), wp.clamp(r0, 0, rows - 1)
        nearest_dx = (wp.float64(nearest_c) - wp.float64(cx)) * wp.float64(du)
        nearest_dy = (wp.float64(nearest_r) - wp.float64(cy)) * wp.float64(dv)
        nearest = nearest_dx * nearest_dx + nearest_dy * nearest_dy
        total = float(0.0)  # noqa: UP018 -- Warp needs a dynamic loop accumulator
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                row, col = r0 + dr, c0 + dc
                if row >= 0 and row < rows and col >= 0 and col < cols:
                    dx = (wp.float64(col) - wp.float64(cx)) * wp.float64(du)
                    dy = (wp.float64(row) - wp.float64(cy)) * wp.float64(dv)
                    distance = wp.max(dx * dx + dy * dy - nearest, wp.float64(0.0))
                    total += float(
                        wp.exp(-wp.float64(0.5) * (distance / sigma) / sigma)
                    )
        if total > 0.0:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    row, col = r0 + dr, c0 + dc
                    if row >= 0 and row < rows and col >= 0 and col < cols:
                        dx = (wp.float64(col) - wp.float64(cx)) * wp.float64(du)
                        dy = (wp.float64(row) - wp.float64(cy)) * wp.float64(dv)
                        distance = wp.max(dx * dx + dy * dy - nearest, wp.float64(0.0))
                        weight = (
                            float(wp.exp(-wp.float64(0.5) * (distance / sigma) / sigma))
                            / total
                        )
                        wp.atomic_add(grid, row * cols + col, load * weight)
            wp.atomic_add(sums, 0, wp.float64(load[0]))
            wp.atomic_add(sums, 1, wp.float64(load[1]))
            wp.atomic_add(sums, 2, wp.float64(load[2]))
            wp.atomic_add(sums, 3, wp.float64(x) * wp.float64(load[2]))
            wp.atomic_add(sums, 4, wp.float64(y) * wp.float64(load[2]))


class WarpTactileProjector:
    """Gaussian taxel projection with only compact outputs copied to the host."""

    def __init__(self, layout, *, sigma, radius=2, device="cuda:0"):
        if (
            not np.isfinite(sigma)
            or sigma <= 0
            or isinstance(radius, bool)
            or not np.isfinite(radius)
            or radius < 1
            or int(radius) != radius
        ):
            raise ValueError(
                "GPU Gaussian projection requires positive sigma and integer radius"
            )
        self.layout, self.sigma, self.radius = layout, float(sigma), int(radius)
        self.device = wp.get_device(device)
        self.grid = wp.zeros(
            layout.rows * layout.cols, dtype=wp.vec3, device=self.device
        )
        self.sums = wp.zeros(6, dtype=wp.float64, device=self.device)

    def project(self, positions, forces, *, return_frame=True):
        if (
            positions.shape != forces.shape
            or positions.dtype not in (wp.vec3, wp.vec3d)
            or forces.dtype != wp.vec3
        ):
            raise ValueError("positions and forces must be matching vec3 arrays")
        if positions.device != self.device or forces.device != self.device:
            raise ValueError("projection arrays must match the workspace device")
        self.grid.zero_()
        self.sums.zero_()
        l = self.layout
        if len(positions):
            wp.launch(
                position_kernel(_project, positions),
                len(positions),
                inputs=[
                    positions,
                    forces,
                    wp.vec3(*l.origin),
                    wp.vec3(*l.u),
                    wp.vec3(*l.v),
                    wp.vec3(*l.normal),
                    l.width,
                    l.height,
                    l.rows,
                    l.cols,
                    self.sigma,
                    self.radius,
                    self.grid,
                    self.sums,
                ],
                device=self.device,
            )
        totals = self.sums.numpy()
        if totals[5] or not np.all(np.isfinite(totals[:5])):
            raise ValueError("tactile positions and forces must be finite")
        if not return_frame:
            return totals[:3].copy()
        grid = self.grid.numpy().reshape(l.rows, l.cols, 3)
        if not np.all(np.isfinite(grid)):
            raise ValueError("taxel accumulation overflow; rescale the forces")
        cop = np.full(3, np.nan, dtype=np.float32)
        if totals[2] > 1.0e-8:
            cop = l.local_to_world(
                np.array(
                    [totals[3] / totals[2], totals[4] / totals[2], 0.0],
                    dtype=np.float32,
                )
            )
        normal = grid[:, :, 2].copy()
        return TactileFrame(
            normal / l.cell_area,
            grid[:, :, :2].copy(),
            normal,
            grid.sum(axis=(0, 1)),
            cop,
        )
