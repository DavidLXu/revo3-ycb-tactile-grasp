"""Reuse a self-CCD certificate only inside its proven, padded path tube.

A path certified above gap+skin tolerates a per-vertex perturbation delta
when 2*delta < skin: convex feature points move by at most delta each.
Both endpoints are checked against the cached affine prefix; floating-point
evaluation error is included. This never certifies sphere/tet constraints or
replaces the independent final old-state-to-target-state physical path check.
"""
from typing import Any
import numpy as np
import warp as wp
from .ipc import positive_scalar

wp.set_module_options({"enable_backward": False})


@wp.kernel
def begin(valid: wp.array(dtype=int), alpha: wp.array(dtype=wp.float64),
          saved_alpha: wp.array(dtype=wp.float64), full: wp.array(dtype=int)):
    a, b = alpha[2], saved_alpha[0]
    full[0] = int(valid[0] == 0 or not wp.isfinite(a) or not wp.isfinite(b)
                  or a <= wp.float64(0.) or b <= wp.float64(0.) or a > b)


@wp.kernel
def guard(old: wp.array(dtype=Any), target: wp.array(dtype=Any),
          origin: wp.array(dtype=wp.vec3d), endpoint: wp.array(dtype=wp.vec3d),
          alpha: wp.array(dtype=wp.float64), saved_alpha: wp.array(dtype=wp.float64),
          skin: wp.float64, full: wp.array(dtype=int)):
    if full[0] != 0:
        return
    i = wp.tid()
    ratio = alpha[2]/saved_alpha[0]
    p, q = wp.vec3d(old[i]), wp.vec3d(target[i])
    reference = origin[i]+ratio*(endpoint[i]-origin[i])
    scale = wp.float64(1e-30)
    finite = bool(True)
    for k in range(3):
        finite = finite and wp.isfinite(p[k]) and wp.isfinite(q[k])
        finite = finite and wp.isfinite(origin[i][k]) and wp.isfinite(endpoint[i][k]) and wp.isfinite(reference[k])
        scale = wp.max(scale, wp.max(wp.abs(p[k]), wp.abs(q[k])))
        scale = wp.max(scale, wp.max(wp.abs(origin[i][k]), wp.abs(endpoint[i][k])))
    if not finite:
        wp.atomic_max(full, 0, 1)
        return
    rounding = wp.float64(128.*2.220446049250313e-16)*scale
    delta = wp.max(wp.length(p-origin[i]), wp.length(q-reference))+rounding
    if not wp.isfinite(delta) or wp.float64(2.)*delta >= skin:
        wp.atomic_max(full, 0, 1)


@wp.kernel
def classify(stats: wp.array(dtype=int), valid: wp.array(dtype=int),
             alpha: wp.array(dtype=wp.float64), saved_alpha: wp.array(dtype=wp.float64),
             original: wp.array(dtype=int), cached_stats: wp.array(dtype=int), counters: wp.array(dtype=int)):
    good = stats[1] == 0 and stats[2] == 0
    valid[0] = int(good)
    original[0] = int(not good)
    counters[0] += 1
    if good:
        saved_alpha[0] = alpha[2]
        for k in range(3):
            cached_stats[k] = stats[k]
    else:
        counters[2] += 1


@wp.kernel
def save(old: wp.array(dtype=Any), target: wp.array(dtype=Any), valid: wp.array(dtype=int),
         origin: wp.array(dtype=wp.vec3d), endpoint: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if valid[0] != 0:
        origin[i] = wp.vec3d(old[i])
        endpoint[i] = wp.vec3d(target[i])


@wp.kernel
def reuse(full: wp.array(dtype=int), stats: wp.array(dtype=int),
          cached_stats: wp.array(dtype=int), counters: wp.array(dtype=int)):
    if full[0] == 0:
        for k in range(3):
            stats[k] = cached_stats[k]
        counters[1] += 1


class GPUPathPrefixCache:
    def __init__(self, audit, skin):
        self.audit, self.skin = audit, positive_scalar(skin, "path certificate skin")
        if self.skin+audit.min_gap > np.finfo(np.float32).max/4:
            raise ValueError("path certificate gap exceeds conservative bound range")
        device = audit.device
        self.valid = wp.zeros(1, dtype=int, device=device)
        self.full = wp.zeros(1, dtype=int, device=device)
        self.original = wp.zeros(1, dtype=int, device=device)
        self.alpha = wp.zeros(1, dtype=wp.float64, device=device)
        self.origin = wp.zeros(audit.particle_count, dtype=wp.vec3d, device=device)
        self.endpoint = wp.zeros_like(self.origin)
        self.cached_stats = wp.zeros(3, dtype=int, device=device)
        self.counters = wp.zeros(3, dtype=int, device=device)  # strong / reused / original retry

    def reset(self):
        self.valid.zero_()

    def check(self, old, new, alpha):
        a, device = self.audit, self.audit.device
        wp.launch(begin, 1, [self.valid, alpha, self.alpha, self.full], device=device)
        wp.launch(guard, a.particle_count,
                  [old, new, self.origin, self.endpoint, alpha, self.alpha, wp.float64(self.skin), self.full], device=device)
        wp.launch(reuse, 1, [self.full, a.stats, self.cached_stats, self.counters], device=device)
        def full_check():
            # Round outward: the tested margin must never be smaller than skin.
            gap = float(np.nextafter(a.min_gap+self.skin, np.inf))
            a.check(old, new, readback=False, gap_override=gap)
            wp.launch(classify, 1, [a.stats, self.valid, alpha, self.alpha, self.original,
                                   self.cached_stats, self.counters], device=device)
            wp.launch(save, a.particle_count, [old, new, self.valid, self.origin, self.endpoint], device=device)
            # Failure of the stronger certificate does not change acceptance:
            # immediately execute the original check at the original gap.
            wp.capture_if(self.original, lambda: a.check(old, new, readback=False))
        wp.capture_if(self.full, full_check)
