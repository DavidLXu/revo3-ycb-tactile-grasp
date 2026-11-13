"""IPC normal contact for particles and a translating sphere.

NumPy supplies analytical derivatives and a reference CCD implementation. Optional
Warp implementations are loaded on demand; neither path is an implicit VBD solve.
All distances are in metres, forces in newtons, and barrier stiffness in N/m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def positive_scalar(value, name, *, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}"
        )
    return value


def points_and_center(positions, center):
    positions = np.asarray(positions, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or center.shape != (3,):
        raise ValueError("positions must be (N,3) and sphere_center must be (3,)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(center)):
        raise ValueError("IPC inputs must be finite")
    return positions, center


@dataclass(frozen=True)
class IPCContactEvaluation:
    gap_m: np.ndarray
    active: np.ndarray
    energy_J: float
    force_N: np.ndarray
    hessian: tuple[np.ndarray, ...]

    @property
    def min_gap_m(self):
        return float(np.min(self.gap_m)) if len(self.gap_m) else float("inf")

    @property
    def max_force_N(self):
        return (
            float(np.linalg.norm(self.force_N, axis=1).max())
            if len(self.force_N)
            else 0.0
        )


@dataclass(frozen=True)
class IPCStepResult:
    positions: np.ndarray
    alpha: float
    min_gap_m: float  # minimum over the complete accepted swept segment
    backtracks: int
    sphere_center: np.ndarray


def _barrier_terms(gap, activation_distance):
    """Return b(g), b'(g), b''(g) for b=-(g-d)^2 log(g/d)."""
    d = positive_scalar(activation_distance, "activation_distance")
    gap = np.asarray(gap, dtype=np.float64)
    if np.any(~np.isfinite(gap)) or np.any(gap <= 0.0):
        raise ValueError("IPC barrier requires strictly positive gaps")
    energy, first, second = (np.zeros_like(gap) for _ in range(3))
    active = gap < d
    g = gap[active]
    a = g - d
    log_ratio = np.log(g / d)
    energy[active] = -(a * a) * log_ratio
    first[active] = -2.0 * a * log_ratio - (a * a) / g
    second[active] = -2.0 * log_ratio - 4.0 * a / g + (a * a) / (g * g)
    return energy, first, second


class IPCSphereContact:
    """Normal barrier, exact derivatives, and continuous particle/sphere CCD.

    The exact Hessian can be indefinite in tangential directions. Request
    ``project_hessian=True`` for a PSD solver approximation; derivative tests
    must use the exact Hessian. ``compute_hessian=False`` avoids unnecessary
    allocation in diagnostics and explicit integration.
    """

    def __init__(
        self,
        sphere_radius_m,
        *,
        activation_distance_m,
        barrier_stiffness=1.0,
        particle_radius_m=0.0,
        min_gap_m=1.0e-7,
    ):
        self.sphere_radius_m = positive_scalar(sphere_radius_m, "sphere radius")
        self.activation_distance_m = positive_scalar(
            activation_distance_m, "activation distance"
        )
        self.barrier_stiffness = positive_scalar(barrier_stiffness, "barrier stiffness")
        self.min_gap_m = positive_scalar(min_gap_m, "min gap")
        if self.min_gap_m >= self.activation_distance_m:
            raise ValueError("min gap must be smaller than activation distance")
        radii = np.asarray(particle_radius_m, dtype=np.float64)
        if radii.ndim > 1 or np.any(~np.isfinite(radii)) or np.any(radii < 0):
            raise ValueError(
                "particle radii must be a finite non-negative scalar or vector"
            )
        self.particle_radius_m = radii.copy()

    def radii(self, count):
        try:
            return np.broadcast_to(self.particle_radius_m, (count,))
        except ValueError as exc:
            raise ValueError("particle radii must match the particle count") from exc

    def gaps(self, positions, sphere_center):
        p, c = points_and_center(positions, sphere_center)
        return np.linalg.norm(p - c, axis=1) - self.sphere_radius_m - self.radii(len(p))

    def evaluate(
        self, positions, sphere_center, *, compute_hessian=True, project_hessian=False
    ):
        p, c = points_and_center(positions, sphere_center)
        delta = p - c
        distance = np.linalg.norm(delta, axis=1)
        gap = distance - self.sphere_radius_m - self.radii(len(p))
        if np.any(gap <= self.min_gap_m):
            raise ValueError("IPC state is at or beyond the configured minimum gap")
        b, first, second = _barrier_terms(gap, self.activation_distance_m)
        normal = delta / distance[:, None]
        first *= self.barrier_stiffness
        second *= self.barrier_stiffness
        force = -first[:, None] * normal
        hessians = ()
        if compute_hessian:
            outer = normal[:, :, None] * normal[:, None, :]
            tangent = first / distance
            if project_hessian:
                tangent = np.maximum(tangent, 0.0)
            hessians = tuple(
                second[:, None, None] * outer
                + tangent[:, None, None] * (np.eye(3) - outer)
            )
        if not np.all(np.isfinite(force)):
            raise ValueError("IPC force overflow; rescale the model")
        return IPCContactEvaluation(
            gap,
            gap < self.activation_distance_m,
            float(self.barrier_stiffness * b.sum()),
            force,
            hessians,
        )

    def evaluate_warp(self, positions, sphere_center, *, device=None, workspace=None):
        """Return device forces and float64 gaps, rejecting invalid input states.

        Pass a reusable ``WarpSphereContact`` workspace to avoid allocations.
        Validation transfers one small diagnostic buffer, never particle arrays.
        """
        from .ipc_warp import WarpSphereContact

        backend = workspace or WarpSphereContact(
            self, len(positions), device or positions.device
        )
        if backend.contact is not self:
            raise ValueError("workspace must belong to this contact model")
        backend.evaluate(positions, sphere_center)
        return backend.forces, backend.gaps

    def swept_min_gap(self, old_positions, new_positions, old_center, new_center):
        old, c0 = points_and_center(old_positions, old_center)
        new, c1 = points_and_center(new_positions, new_center)
        if old.shape != new.shape:
            raise ValueError("old_positions and new_positions must have equal shape")
        if not len(old):
            return float("inf")
        rel0 = old - c0
        velocity = (new - c1) - rel0
        denom = np.sum(velocity * velocity, axis=1)
        alpha = np.divide(
            -np.sum(rel0 * velocity, axis=1),
            denom,
            out=np.zeros_like(denom),
            where=denom > 0,
        )
        closest = rel0 + np.clip(alpha, 0.0, 1.0)[:, None] * velocity
        return float(
            np.min(
                np.linalg.norm(closest, axis=1)
                - self.sphere_radius_m
                - self.radii(len(old))
            )
        )

    def conservative_step(
        self,
        old_positions,
        new_positions,
        old_center,
        new_center,
        *,
        max_backtracks=60,
        safety=0.9,
        position_dtype=None,
    ):
        """CCD-safe affine step with slack, optionally checked after float32 rounding.

        Unlike endpoint-only checks this catches a sphere passing right through a
        particle. Bisection uses distances only, not energy/Hessian evaluations.
        NaN proposals are rejected rather than hidden by the line search.
        """
        old, c0 = points_and_center(old_positions, old_center)
        new, c1 = points_and_center(new_positions, new_center)
        if old.shape != new.shape:
            raise ValueError("old_positions and new_positions must have equal shape")
        if (
            isinstance(max_backtracks, bool)
            or int(max_backtracks) != max_backtracks
            or max_backtracks < 1
        ):
            raise ValueError("max_backtracks must be a positive integer")
        if not np.isfinite(safety) or not 0 < safety < 1:
            raise ValueError("safety must be in (0,1)")
        dtype = np.dtype(position_dtype or np.float64)
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("position_dtype must be float32 or float64")
        old_gap = self.gaps(old, c0)
        if np.any(~np.isfinite(old_gap)) or np.any(old_gap <= self.min_gap_m):
            raise ValueError("old IPC state is not strictly separated")
        radii = self.sphere_radius_m + self.radii(len(old))
        # Match the GPU's per-particle float32 rounding bounds and acceptance
        # rule; a global coordinate bound changes the simulated trajectory.
        scale = np.maximum(np.linalg.norm(old, axis=1), np.linalg.norm(new, axis=1))
        scale = np.maximum(scale, max(np.linalg.norm(c0), np.linalg.norm(c1)))
        scale = np.maximum(scale, radii)
        threshold = np.minimum(
            self.min_gap_m + 8 * np.finfo(dtype).eps * scale,
            self.min_gap_m + 0.5 * (old_gap - self.min_gap_m),
        )
        relative = old - c0
        velocity = (new - c1) - relative
        speed2 = np.sum(velocity * velocity, axis=1)
        closest_time = np.divide(
            -np.sum(relative * velocity, axis=1),
            speed2,
            out=np.zeros_like(speed2),
            where=speed2 > 0,
        )
        closest_time = np.maximum(closest_time, 0.0)

        def segment_gaps(alpha):
            t = np.minimum(closest_time, alpha)
            return np.linalg.norm(relative + t[:, None] * velocity, axis=1) - radii

        full = segment_gaps(1.0)
        if np.any(~np.isfinite(full)):
            raise ValueError("IPC proposal overflows distance calculations")
        limited = full <= threshold
        alpha, backtracks = 1.0, 0
        if np.any(limited):
            lo, hi = np.zeros(len(old)), np.ones(len(old))
            for _ in range(int(max_backtracks)):
                mid = (lo + hi) * 0.5
                safe = segment_gaps(mid) > threshold
                lo, hi = np.where(safe, mid, lo), np.where(safe, hi, mid)
            alpha = float(np.min(np.where(limited, safety * lo, 1.0)))
            backtracks = int(max_backtracks)
        p = (old + alpha * (new - old)).astype(dtype).astype(np.float64)
        c = (c0 + alpha * (c1 - c0)).astype(dtype).astype(np.float64)
        gap = self.swept_min_gap(old, p, c0, c)
        if gap <= self.min_gap_m or (len(old) and not np.isfinite(gap)):
            raise RuntimeError("IPC CCD failed to retain a representable positive gap")
        return IPCStepResult(p, alpha, gap, backtracks, c)
