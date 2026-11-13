"""Fail-closed sphere/surface path guard for the single-sphere tet adapter.

This is a geometric acceptance guard, not a triangle contact force model or
self-contact solver. Distances use float64 math on the actual stored vertices.
"""

from typing import Any
from .ipc_precision import position_overloads, pose_value
import warp as wp

from .ipc_primitives_warp import _delta, _finite, _weights

wp.set_module_options({"enable_backward": False})


@wp.func
def _support_gap(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, radius: wp.float64):
    # Closest-point weights choose a useful separating direction. The support
    # plane itself is a lower bound even if those weights are inaccurate.
    zero = wp.vec3d(0.0)
    weights = _weights(zero, a, b, c, 0)
    delta = -_delta(weights, zero, a, b, c)
    length = wp.length(delta)
    result = -radius
    if length > wp.float64(0.0):
        n = delta / length
        result = wp.min(wp.dot(n, a), wp.min(wp.dot(n, b), wp.dot(n, c))) - radius
    return result


@wp.func
def _sphere_prefix(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d,
                   aa: wp.vec3d, bb: wp.vec3d, cc: wp.vec3d,
                   radius: wp.float64, threshold: wp.float64):
    """Return the certified prefix length and its conservative gap bound."""
    va, vb, vc = aa-a, bb-b, cc-c
    speed = wp.max(wp.length(va), wp.max(wp.length(vb), wp.length(vc)))
    t, lower = wp.float64(0.0), wp.float64(wp.inf)
    for attempt in range(128):
        at, bt, ct = a+t*va, b+t*vb, c+t*vc
        gap = _support_gap(at, bt, ct, radius)
        if not wp.isfinite(gap) or gap <= threshold:
            break
        remaining = wp.float64(1.0)-t
        lipschitz = gap-speed*remaining
        if lipschitz > threshold:
            lower = wp.min(lower, lipschitz)
            t = wp.float64(1.0)
            break
        zero = wp.vec3d(0.0)
        delta = -_delta(_weights(zero, at, bt, ct, 0), zero, at, bt, ct)
        normal = delta/wp.length(delta)
        end_gap = wp.min(wp.dot(normal, aa), wp.min(wp.dot(normal, bb), wp.dot(normal, cc)))-radius
        if not wp.isfinite(end_gap):
            break
        if end_gap > threshold:
            lower = wp.min(lower, wp.min(gap, end_gap))
            t = wp.float64(1.0)
            break
        advance = wp.float64(0.9)*(gap-threshold)/speed
        plane = wp.float64(0.9)*(gap-threshold)*remaining/(gap-end_gap)
        advance = wp.min(wp.max(advance, plane), remaining)
        if advance <= wp.float64(2.220446049250313e-16):
            break
        bound = wp.max(gap-speed*advance, gap+(end_gap-gap)*advance/remaining)
        if not wp.isfinite(bound) or bound <= threshold:
            break
        lower = wp.min(lower, bound)
        t += advance
    return wp.vec2d(t, lower)


@wp.func
def _sphere_motion(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d,
                   aa: wp.vec3d, bb: wp.vec3d, cc: wp.vec3d,
                   radius: wp.float64, threshold: wp.float64):
    lower = wp.float64(wp.inf)
    complete = bool(False)
    for contraction in range(32):
        prefix = _sphere_prefix(a, b, c, aa, bb, cc, radius, threshold)
        lower = wp.min(lower, prefix[1])
        if prefix[0] >= wp.float64(1.0):
            complete = True
            break
        suffix = _sphere_prefix(aa, bb, cc, a, b, c, radius, threshold)
        lower = wp.min(lower, suffix[1])
        if prefix[0]+suffix[0] >= wp.float64(1.0):
            complete = True
            break
        if prefix[0]+suffix[0] <= wp.float64(1e-12):
            break
        # Only remove intervals whose lower bounds were proved safe. Positive
        # endpoints or a sampled midpoint never certify an uncovered interval.
        va, vb, vc = aa-a, bb-b, cc-c
        aa, bb, cc = aa-suffix[0]*va, bb-suffix[0]*vb, cc-suffix[0]*vc
        a, b, c = a+prefix[0]*va, b+prefix[0]*vb, c+prefix[0]*vc
    return complete, lower


@position_overloads("old", "new", poses=('center0', 'center1'))
@wp.kernel
def validate_surface_paths(
    old: wp.array(dtype=Any),
    new: wp.array(dtype=Any),
    faces: wp.array2d(dtype=int),
    center0: Any,
    center1: Any,
    radius: wp.float64,
    min_gap: wp.float64,
    diagnostic: wp.array(dtype=wp.float64),
):
    _center0 = pose_value(center0)
    _center1 = pose_value(center1)
    i = wp.tid()
    a0 = wp.vec3d(old[faces[i, 0]]) - _center0
    b0 = wp.vec3d(old[faces[i, 1]]) - _center0
    c0 = wp.vec3d(old[faces[i, 2]]) - _center0
    a1 = wp.vec3d(new[faces[i, 0]]) - _center1
    b1 = wp.vec3d(new[faces[i, 1]]) - _center1
    c1 = wp.vec3d(new[faces[i, 2]]) - _center1
    va, vb, vc = a1 - a0, b1 - b0, c1 - c0
    speed = wp.max(wp.length(va), wp.max(wp.length(vb), wp.length(vc)))
    scale = wp.max(wp.length(_center0), wp.length(_center1)) + radius
    scale += wp.max(wp.length(a0), wp.max(wp.length(b0), wp.length(c0))) + speed
    guard = wp.float64(128.0 * 2.220446049250313e-16) * scale
    threshold = min_gap + guard
    lower = wp.float64(wp.inf)
    complete = bool(False)  # noqa: UP018 -- Warp dynamic-loop variable
    if (
        _finite(a0)
        and _finite(b0)
        and _finite(c0)
        and _finite(a1)
        and _finite(b1)
        and _finite(c1)
        and wp.isfinite(speed)
        and wp.isfinite(guard)
    ):
        # Every interpolated triangle lies in this swept relative AABB.
        squared = wp.float64(0.0)
        for axis in range(3):
            lo = wp.min(
                wp.min(a0[axis], a1[axis]),
                wp.min(wp.min(b0[axis], b1[axis]), wp.min(c0[axis], c1[axis])),
            )
            hi = wp.max(
                wp.max(a0[axis], a1[axis]),
                wp.max(wp.max(b0[axis], b1[axis]), wp.max(c0[axis], c1[axis])),
            )
            offset = wp.max(lo, wp.max(-hi, wp.float64(0.0)))
            squared += offset * offset
        box_gap = wp.sqrt(squared) - radius
        if box_gap > threshold:
            lower = box_gap
            complete = True
        else:
            complete, lower = _sphere_motion(a0, b0, c0, a1, b1, c1, radius, threshold)
    if not complete or not wp.isfinite(lower) or lower - guard <= min_gap:
        wp.atomic_add(diagnostic, 0, wp.float64(1.0))
    wp.atomic_min(diagnostic, 5, lower - guard)


@position_overloads("positions", poses=('center',))
@wp.kernel
def validate_sphere_outside_tets(
    positions: wp.array(dtype=Any),
    tets: wp.array2d(dtype=int),
    center: Any,
    diagnostic: wp.array(dtype=wp.float64),
):
    _center = pose_value(center)
    i = wp.tid()
    a = wp.vec3d(positions[tets[i, 0]]) - _center
    b = wp.vec3d(positions[tets[i, 1]]) - _center
    c = wp.vec3d(positions[tets[i, 2]]) - _center
    d = wp.vec3d(positions[tets[i, 3]]) - _center
    candidate = bool(True)  # noqa: UP018 -- Warp dynamic-loop variable
    for axis in range(3):
        lo = wp.min(wp.min(a[axis], b[axis]), wp.min(c[axis], d[axis]))
        hi = wp.max(wp.max(a[axis], b[axis]), wp.max(c[axis], d[axis]))
        if lo > wp.float64(0.0) or hi < wp.float64(0.0):
            candidate = False
    if candidate:
        bary = wp.inverse(wp.matrix_from_cols(b - a, c - a, d - a)) * (-a)
        tolerance = wp.float64(1.0e-10)
        if not _finite(bary) or (
            bary[0] >= -tolerance
            and bary[1] >= -tolerance
            and bary[2] >= -tolerance
            and bary[0] + bary[1] + bary[2] <= wp.float64(1.0) + tolerance
        ):
            wp.atomic_add(diagnostic, 0, wp.float64(1.0))
