"""Warp force/gap and conservative CCD for batches of supplied primitive pairs.

Vertices are an (N,4) Warp vec3 array. As in the CPU reference, full Hessians are
available through IPCPrimitiveContact.evaluate; these kernels return forces and
certified CCD prefixes, not a general implicit mesh contact solver.
"""

import numpy as np
import warp as wp


@wp.func
def _segment_t(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d):
    e = b - a
    den = wp.dot(e, e)
    t = wp.float64(0.0)
    if den > wp.float64(0.0):
        t = wp.clamp(wp.dot(p - a, e) / den, wp.float64(0.0), wp.float64(1.0))
    return t


@wp.func
def _delta(w: wp.vec4d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d):
    # All callers supply zero-sum weights. Anchor on an ACTIVE vertex so a
    # distant, zero-weight endpoint cannot cancel a small local separation.
    if w[3] != wp.float64(0):
        return w[0] * (a-d) + w[1] * (b-d) + w[2] * (c-d)
    if w[2] != wp.float64(0):
        return w[0] * (a-c) + w[1] * (b-c)
    return w[0] * (a-b)


@wp.func
def _choose(
    w: wp.vec4d, candidate: wp.vec4d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d
):
    r0, r1 = _delta(w, a, b, c, d), _delta(candidate, a, b, c, d)
    result = w
    if wp.dot(r1, r1) < wp.dot(r0, r0):
        result = candidate
    return result


@wp.func
def _weights(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, kind: int):
    one, zero = wp.float64(1.0), wp.float64(0.0)
    w = wp.vec4d(0.0)
    if kind == 0:
        t = _segment_t(a, b, c)
        w = wp.vec4d(one, t - one, -t, zero)
        t = _segment_t(a, c, d)
        w = _choose(w, wp.vec4d(one, zero, t - one, -t), a, b, c, d)
        t = _segment_t(a, d, b)
        w = _choose(w, wp.vec4d(one, -t, zero, t - one), a, b, c, d)
        e0, e1 = c - b, d - b
        n = wp.cross(e0, e1)
        den = wp.dot(n, n)
        if den > zero:
            u = wp.dot(wp.cross(a - b, e1), n) / den
            v = wp.dot(wp.cross(e0, a - b), n) / den
            if u >= zero and v >= zero and u + v <= one:
                w = _choose(w, wp.vec4d(one, u + v - one, -u, -v), a, b, c, d)
    elif kind == 2:
        # Convergent IPC correction: distance from c to segment (a,b), with
        # d retained only for the edge-edge mollifier and its derivatives.
        s = _segment_t(c, a, b)
        w = wp.vec4d(one - s, s, -one, zero)
    else:
        t = _segment_t(a, c, d)
        w = wp.vec4d(one, zero, t - one, -t)
        t = _segment_t(b, c, d)
        w = _choose(w, wp.vec4d(zero, one, t - one, -t), a, b, c, d)
        s = _segment_t(c, a, b)
        w = _choose(w, wp.vec4d(one - s, s, -one, zero), a, b, c, d)
        s = _segment_t(d, a, b)
        w = _choose(w, wp.vec4d(one - s, s, zero, -one), a, b, c, d)
        e0, e1 = b - a, d - c
        n = wp.cross(e0, e1)
        den = wp.dot(n, n)
        if den > zero:
            s = wp.dot(wp.cross(c - a, e1), n) / den
            t = wp.dot(wp.cross(c - a, e0), n) / den
            if s >= zero and s <= one and t >= zero and t <= one:
                w = _choose(w, wp.vec4d(one - s, s, t - one, -t), a, b, c, d)
    return w


@wp.func
def _gap(
    a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, kind: int, thickness: wp.float64
):
    return wp.length(_delta(_weights(a, b, c, d, kind), a, b, c, d)) - thickness


@wp.func
def _finite(x: wp.vec3d):
    return wp.isfinite(x[0]) and wp.isfinite(x[1]) and wp.isfinite(x[2])


@wp.kernel
def _evaluate(
    x: wp.array2d(dtype=wp.vec3),
    kind: int,
    thickness: wp.float64,
    dhat: wp.float64,
    stiffness: wp.float64,
    min_gap: wp.float64,
    force: wp.array2d(dtype=wp.vec3),
    gap: wp.array(dtype=wp.float64),
    energy: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=int),
):
    i = wp.tid()
    a, b, c, d = (
        wp.vec3d(x[i, 0]),
        wp.vec3d(x[i, 1]),
        wp.vec3d(x[i, 2]),
        wp.vec3d(x[i, 3]),
    )
    w = _weights(a, b, c, d, kind)
    delta = _delta(w, a, b, c, d)
    distance = wp.length(delta)
    g = distance - thickness
    gap[i] = g
    energy[i] = wp.float64(0.0)
    for j in range(4):
        force[i, j] = wp.vec3(0.0)
    if (
        not _finite(a)
        or not _finite(b)
        or not _finite(c)
        or not _finite(d)
        or g <= min_gap
        or not wp.isfinite(g)
    ):
        wp.atomic_add(invalid, 0, 1)
    elif g < dhat:
        t = g - dhat
        log = wp.log(g / dhat)
        first = -wp.float64(2.0) * t * log - t * t / g
        energy[i] = -stiffness * t * t * log
        for j in range(4):
            f = wp.vec3(-stiffness * first * w[j] * delta / distance)
            force[i, j] = f
            if not _finite(wp.vec3d(f)):
                wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _ccd(
    old: wp.array2d(dtype=wp.vec3),
    new: wp.array2d(dtype=wp.vec3),
    kind: int,
    thickness: wp.float64,
    min_gap: wp.float64,
    safety: wp.float64,
    max_iterations: int,
    accepted: wp.array2d(dtype=wp.vec3),
    alpha_out: wp.array(dtype=wp.float64),
    lower_out: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=int),
):
    i = wp.tid()
    a, b, c, d = (
        wp.vec3d(old[i, 0]),
        wp.vec3d(old[i, 1]),
        wp.vec3d(old[i, 2]),
        wp.vec3d(old[i, 3]),
    )
    va, vb, vc, vd = (
        wp.vec3d(new[i, 0]) - a,
        wp.vec3d(new[i, 1]) - b,
        wp.vec3d(new[i, 2]) - c,
        wp.vec3d(new[i, 3]) - d,
    )
    mean = (va + vb + vc + vd) / wp.float64(4.0)
    sa, sb, sc, sd = (
        wp.length(va - mean),
        wp.length(vb - mean),
        wp.length(vc - mean),
        wp.length(vd - mean),
    )
    speed = wp.max(sa, sb) + wp.max(sc, sd)
    if kind == 0:
        speed = sa + wp.max(sb, wp.max(sc, sd))
    scale = wp.max(
        wp.max(wp.length(a), wp.length(b)), wp.max(wp.length(c), wp.length(d))
    )
    scale += wp.max(
        wp.max(wp.length(va), wp.length(vb)), wp.max(wp.length(vc), wp.length(vd))
    )
    # Distance between two convex features is 1-Lipschitz in each feature's
    # maximum vertex perturbation. Cover both endpoints' float32 rounding.
    rounding = wp.float64(8.0 * 1.1920928955078125e-7) * scale
    threshold = min_gap + rounding
    gap = _gap(a, b, c, d, kind, thickness)
    alpha = wp.float64(0.0)
    lower = gap
    if (
        not _finite(a)
        or not _finite(b)
        or not _finite(c)
        or not _finite(d)
        or not _finite(va)
        or not _finite(vb)
        or not _finite(vc)
        or not _finite(vd)
        or gap <= threshold
    ):
        wp.atomic_add(invalid, 0, 1)
    else:
        if speed == wp.float64(0.0):
            alpha = wp.float64(1.0)
        else:
            for it in range(max_iterations):
                remain = wp.float64(1.0) - alpha
                if gap - speed * remain > threshold:
                    lower = wp.min(lower, gap - speed * remain)
                    alpha = wp.float64(1.0)
                    break
                step = safety * (gap - threshold) / speed
                if step <= wp.float64(2.220446049250313e-16):
                    break
                step = wp.min(step, remain)
                lower = wp.min(lower, gap - speed * step)
                alpha += step
                gap = _gap(
                    a + alpha * va,
                    b + alpha * vb,
                    c + alpha * vc,
                    d + alpha * vd,
                    kind,
                    thickness,
                )
        # The certified lower bound is reduced by the rounding bound, so it
        # applies to the entire rounded affine trajectory, not just its end.
        if lower - rounding <= min_gap:
            wp.atomic_add(invalid, 0, 1)
    for j in range(4):
        accepted[i, j] = wp.vec3(
            wp.vec3d(old[i, j]) + alpha * (wp.vec3d(new[i, j]) - wp.vec3d(old[i, j]))
        )
    alpha_out[i] = alpha
    lower_out[i] = lower - rounding


class WarpPrimitiveContact:
    def __init__(self, model, count, device="cuda:0"):
        self.model, self.count, self.device = model, int(count), wp.get_device(device)
        self.kind = 0 if model.kind == "point_triangle" else 1
        self.forces = wp.zeros((count, 4), dtype=wp.vec3, device=self.device)
        self.gaps = wp.zeros(count, dtype=wp.float64, device=self.device)
        self.energies = wp.zeros(count, dtype=wp.float64, device=self.device)
        self.accepted = wp.zeros((count, 4), dtype=wp.vec3, device=self.device)
        self.alphas = wp.zeros(count, dtype=wp.float64, device=self.device)
        self.lower_bounds = wp.zeros(count, dtype=wp.float64, device=self.device)
        self.invalid = wp.zeros(1, dtype=int, device=self.device)

    def _check(self, x):
        if x.shape != (self.count, 4) or x.dtype != wp.vec3 or x.device != self.device:
            raise ValueError("vertices must be a matching (N,4) vec3 array")

    def evaluate(self, x):
        self._check(x)
        self.invalid.zero_()
        m = self.model
        if self.count:
            wp.launch(
                _evaluate,
                self.count,
                inputs=[
                    x,
                    self.kind,
                    m.thickness_m,
                    m.activation_distance_m,
                    m.barrier_stiffness,
                    m.min_gap_m,
                    self.forces,
                    self.gaps,
                    self.energies,
                    self.invalid,
                ],
                device=self.device,
            )
        if self.invalid.numpy()[0]:
            raise ValueError("invalid or intersecting primitive pair")
        return self.forces, self.gaps, self.energies

    def conservative_step(self, old, new, *, safety=0.9, max_iterations=128):
        self._check(old)
        self._check(new)
        self.invalid.zero_()
        m = self.model
        if (
            not np.isfinite(safety)
            or not 0 < safety < 1
            or isinstance(max_iterations, bool)
            or int(max_iterations) != max_iterations
            or max_iterations < 1
        ):
            raise ValueError("invalid safety or iteration budget")
        if self.count:
            wp.launch(
                _ccd,
                self.count,
                inputs=[
                    old,
                    new,
                    self.kind,
                    m.thickness_m,
                    m.min_gap_m,
                    safety,
                    int(max_iterations),
                    self.accepted,
                    self.alphas,
                    self.lower_bounds,
                    self.invalid,
                ],
                device=self.device,
            )
        if self.invalid.numpy()[0]:
            raise ValueError(
                "primitive CCD inputs invalid or lack float32 precision margin"
            )
        return self.accepted, self.alphas, self.lower_bounds
