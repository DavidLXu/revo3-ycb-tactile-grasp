"""Piecewise analytical IPC primitives for supplied point/triangle and edge pairs.

These are narrow-phase building blocks, not a mesh collision pipeline. The exact
Hessian differentiates the active closest feature (including changing closest
coordinates). At feature ties the distance is only piecewise smooth; degenerate
triangles reduce to edges and parallel edges use endpoint/edge candidates.
No broad phase, self-contact exclusions, or edge-edge mollifier is implied.
"""

from dataclasses import dataclass

import numpy as np

from .ipc import _barrier_terms, positive_scalar


def _vertices(vertices):
    x = np.asarray(vertices, dtype=np.float64)
    if x.shape != (4, 3) or not np.all(np.isfinite(x)):
        raise ValueError("vertices must be finite with shape (4,3)")
    return x


def _point_edge(x, p, a, b):
    edge = x[b] - x[a]
    den = edge @ edge
    t = float(np.clip((x[p] - x[a]) @ edge / den, 0.0, 1.0)) if den > 0 else 0.0
    w = np.zeros(4)
    w[p] = 1.0
    w[a] -= 1.0 - t
    w[b] -= t
    dw = np.zeros((0, 4))
    if 0 < t < 1:
        dw = np.zeros((1, 4))
        dw[0, a] = 1.0
        dw[0, b] = -1.0
    return w, dw


def closest_feature(vertices, kind):
    """Return signed closest-point weights and their free parameter derivatives."""
    x = _vertices(vertices)
    candidates = []
    if kind == "point_triangle":
        for a, b in ((1, 2), (2, 3), (3, 1)):
            candidates.append(_point_edge(x, 0, a, b))
        e0, e1 = x[2] - x[1], x[3] - x[1]
        normal = np.cross(e0, e1)
        det = normal @ normal
        # Cross products avoid cancellation in a*c-b*b for thin faces.
        # Only exactly degenerate faces reduce to their boundary.
        if det > 0:
            rel = x[0] - x[1]
            u, v = np.cross(rel, e1) @ normal / det, np.cross(e0, rel) @ normal / det
            if u >= 0 and v >= 0 and u + v <= 1:
                candidates.append(
                    (
                        np.array([1.0, -1.0 + u + v, -u, -v]),
                        np.array([[0.0, 1.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]),
                    )
                )
    elif kind == "edge_edge":
        for p, a, b in ((0, 2, 3), (1, 2, 3)):
            candidates.append(_point_edge(x, p, a, b))
        for p, a, b in ((2, 0, 1), (3, 0, 1)):
            w, dw = _point_edge(x, p, a, b)
            candidates.append((-w, -dw))
        e0, e1 = x[1] - x[0], x[3] - x[2]
        normal = np.cross(e0, e1)
        det = normal @ normal
        if det > 0:
            rel = x[2] - x[0]
            s, t = np.cross(rel, e1) @ normal / det, np.cross(rel, e0) @ normal / det
            if 0 <= s <= 1 and 0 <= t <= 1:
                candidates.append(
                    (
                        np.array([1.0 - s, s, -1.0 + t, -t]),
                        np.array([[-1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]]),
                    )
                )
    else:
        raise ValueError("kind must be point_triangle or edge_edge")
    return min(
        candidates, key=lambda candidate: np.dot(candidate[0] @ x, candidate[0] @ x)
    )


def distance_derivatives(vertices, kind):
    """Exact distance gradient and 12x12 Hessian on the selected feature.

    Eliminate the free closest coordinates from the squared-distance Hessian
    with a Schur complement, then apply the square-root chain rule.
    """
    x = _vertices(vertices)
    w, dw = closest_feature(x, kind)
    r = w @ x
    distance = float(np.linalg.norm(r))
    if distance <= 0:
        raise ValueError("distance derivative is undefined at intersection")
    a = np.kron(w.reshape(1, 4), np.eye(3))
    gq = 2 * a.T @ r
    hq = 2 * a.T @ a
    if len(dw):
        b = x.T @ dw.T
        cross = 2 * (b.T @ a + np.stack([np.kron(d, r) for d in dw]))
        _, singular, vt = np.linalg.svd(b, full_matrices=False)
        if np.any(singular == 0):
            raise ValueError("singular active feature; Hessian is undefined")
        inverse = (vt.T / (2 * singular**2)) @ vt
        hq -= cross.T @ inverse @ cross
    gradient = gq / (2 * distance)
    hessian = hq / (2 * distance) - np.outer(gq, gq) / (4 * distance**3)
    return distance, gradient, 0.5 * (hessian + hessian.T), w


@dataclass(frozen=True)
class PrimitiveEvaluation:
    gap_m: float
    energy_J: float
    force_N: np.ndarray
    hessian: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class PrimitiveStep:
    vertices: np.ndarray
    alpha: float
    min_gap_lower_bound_m: float
    iterations: int


class IPCPrimitiveContact:
    """Barrier on distance minus thickness, with conservative advancement CCD.

    Vertex order is [point,a,b,c] or [edge0_start,edge0_end,edge1_start,edge1_end].
    CCD uses a Lipschitz bound valid for deforming primitives, not endpoint tests.
    If the iteration budget is exhausted it returns the certified prefix only.
    """

    def __init__(
        self,
        kind,
        *,
        activation_distance_m,
        barrier_stiffness=1.0,
        thickness_m=0.0,
        min_gap_m=1.0e-7,
    ):
        if kind not in ("point_triangle", "edge_edge"):
            raise ValueError("unknown primitive kind")
        self.kind = kind
        self.activation_distance_m = positive_scalar(
            activation_distance_m, "activation distance"
        )
        self.barrier_stiffness = positive_scalar(barrier_stiffness, "stiffness")
        self.thickness_m = positive_scalar(thickness_m, "thickness", allow_zero=True)
        self.min_gap_m = positive_scalar(min_gap_m, "minimum gap")
        if self.min_gap_m >= self.activation_distance_m:
            raise ValueError("minimum gap must be smaller than activation distance")

    def gap(self, vertices):
        x = _vertices(vertices)
        w, _ = closest_feature(x, self.kind)
        return float(np.linalg.norm(w @ x) - self.thickness_m)

    def evaluate(self, vertices, *, project_hessian=False, barrier_clearance_m=0.0):
        """Evaluate with an optional barrier origin inside the minimum gap.

        Setting clearance equal to min_gap_m makes the barrier diverge at the
        actual admissible boundary. The default preserves the legacy primitive
        law. Reported gap and CCD always retain their geometric meaning.
        """
        clearance = positive_scalar(barrier_clearance_m, "barrier clearance", allow_zero=True)
        if clearance > self.min_gap_m:
            raise ValueError("barrier clearance must not exceed minimum gap")
        d, g, h, w = distance_derivatives(vertices, self.kind)
        gap = d - self.thickness_m
        if gap <= self.min_gap_m:
            raise ValueError("IPC primitive is at or beyond the minimum gap")
        b, first, second = (
            float(v[0]) for v in _barrier_terms([gap - clearance], self.activation_distance_m)
        )
        hessian = self.barrier_stiffness * (second * np.outer(g, g) + first * h)
        if project_hessian:
            values, vectors = np.linalg.eigh(hessian)
            hessian = (vectors * np.maximum(values, 0.0)) @ vectors.T
        return PrimitiveEvaluation(
            gap,
            self.barrier_stiffness * b,
            (-self.barrier_stiffness * first * g).reshape(4, 3),
            hessian,
            w,
        )

    def conservative_step(
        self, old_vertices, new_vertices, *, safety=0.9, max_iterations=128
    ):
        old, new = _vertices(old_vertices), _vertices(new_vertices)
        if (
            not np.isfinite(safety)
            or not 0 < safety < 1
            or isinstance(max_iterations, bool)
            or int(max_iterations) != max_iterations
            or max_iterations < 1
        ):
            raise ValueError(
                "safety must be in (0,1) and max_iterations a positive integer"
            )
        gap = self.gap(old)
        if gap <= self.min_gap_m:
            raise ValueError("old primitive state is not separated")
        # Common translation may round distinct float64 vertices onto the same
        # feature even when subtracting that translation gives zero speed. Keep
        # a coordinate-scale guard just as the float32 Warp path does.
        scale = max(
            np.linalg.norm(old, axis=1).max(), np.linalg.norm(new, axis=1).max()
        )
        rounding = 16 * np.finfo(float).eps * scale
        threshold = self.min_gap_m + rounding
        if not np.isfinite(gap) or not np.isfinite(rounding) or gap <= threshold:
            raise ValueError("primitive CCD lacks float64 precision margin")
        velocity = new - old
        # Remove common translation to tighten the bound without changing CCD.
        velocity = velocity - velocity.mean(axis=0)
        speeds = np.linalg.norm(velocity, axis=1)
        speed = (
            speeds[0] + max(speeds[1:])
            if self.kind == "point_triangle"
            else max(speeds[:2]) + max(speeds[2:])
        )
        def accepted_step(alpha, lower, iterations):
            vertices = new.copy() if alpha == 1.0 else old + alpha * (new - old)
            certified = min(lower, self.gap(vertices)) - rounding
            if not np.isfinite(certified) or certified <= self.min_gap_m:
                raise RuntimeError("primitive CCD rounded output lost positive gap")
            return PrimitiveStep(vertices, alpha, certified, iterations)

        if speed == 0:
            return accepted_step(1.0, gap, 0)
        alpha, lower = 0.0, gap
        for iteration in range(1, int(max_iterations) + 1):
            remaining = 1.0 - alpha
            if gap - speed * remaining > threshold:
                lower = min(lower, gap - speed * remaining)
                return accepted_step(1.0, lower, iteration)
            step = safety * (gap - threshold) / speed
            if step <= np.finfo(float).eps * max(1.0, alpha):
                break
            step = min(step, remaining)
            lower = min(lower, gap - speed * step)
            alpha += step
            gap = self.gap(old + alpha * (new - old))
            if gap <= self.min_gap_m:
                raise RuntimeError("primitive CCD lost positive gap")
        return accepted_step(alpha, lower, iteration)
