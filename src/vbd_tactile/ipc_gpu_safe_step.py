"""Conservative GPU line-search step and same-direction CCD certificate.

The broad phase covers the complete proposed segment at gap + skin. Each
candidate contributes a certified prefix; their minimum covers every pair.
Smaller materialized trials may reuse it only after an endpoint tube guard.
Overflow, invalid data and insufficient certificates use the original audit.
The final physical trajectory and volume checks are never replaced here.
"""
import numpy as np
import warp as wp
from .ipc import positive_scalar
from .ipc_self_audit import _feature_motion_prefix
from . import ipc_gpu_ccd_pairs as pairs
from . import ipc_gpu_path_cache as cache

wp.set_module_options({'enable_backward': False})


@wp.kernel
def prefixes(old: wp.array(dtype=wp.vec3d), target: wp.array(dtype=wp.vec3d),
             invalid: wp.array(dtype=int), count: wp.array(dtype=int),
             ids: wp.array(dtype=wp.vec4i), meta: wp.array(dtype=wp.vec3i),
             gap: wp.float64, alpha: wp.array(dtype=wp.float64)):
    i = wp.tid()
    if invalid[0] or count[0] > ids.shape[0] or i >= count[0]:
        return
    q = ids[i]
    t = _feature_motion_prefix(old[q[0]], old[q[1]], old[q[2]], old[q[3]],
                              target[q[0]], target[q[1]], target[q[2]], target[q[3]],
                              meta[i][0], gap)
    if not wp.isfinite(t) or t < wp.float64(0.):
        t = wp.float64(0.)
    wp.atomic_min(alpha, 0, wp.min(t, wp.float64(1.)))


@wp.kernel
def limit(invalid: wp.array(dtype=int), count: wp.array(dtype=int), capacity: int,
          prefix: wp.array(dtype=wp.float64), values: wp.array(dtype=wp.float64),
          reference_alpha: wp.array(dtype=wp.float64), safe_alpha: wp.array(dtype=wp.float64),
          valid: wp.array(dtype=int), counters: wp.array(dtype=int)):
    counters[0] += 1
    reference_alpha[0] = values[2]
    good = invalid[0] == 0 and count[0] <= capacity
    good = good and wp.isfinite(prefix[0]) and prefix[0] > wp.float64(0.)
    good = good and wp.isfinite(values[2]) and values[2] > wp.float64(0.)
    # Retain the existing dyadic line-search grid. Do not silently introduce
    # an arbitrary new accepted alpha while optimizing the path work.
    fraction = wp.float64(1.)
    bound = prefix[0]
    if bound < wp.float64(1.):
        bound *= wp.float64(.9)
    for k in range(60):
        if fraction <= bound:
            break
        fraction *= wp.float64(.5)
    good = good and fraction > wp.float64(1.734723475976807e-18)
    if good:
        values[2] *= fraction
        if fraction < wp.float64(1.):
            counters[1] += 1
    else:
        if count[0] > capacity:
            counters[4] += 1
        else:
            counters[5] += 1
    safe_alpha[0] = values[2]
    valid[0] = int(good)


@wp.kernel
def begin(valid: wp.array(dtype=int), values: wp.array(dtype=wp.float64),
          safe_alpha: wp.array(dtype=wp.float64), full: wp.array(dtype=int)):
    alpha = values[2]
    full[0] = int(valid[0] == 0 or not wp.isfinite(alpha)
                  or alpha <= wp.float64(0.) or alpha > safe_alpha[0])


@wp.kernel
def publish(full: wp.array(dtype=int), count: wp.array(dtype=int),
            stats: wp.array(dtype=int), counters: wp.array(dtype=int)):
    if full[0] == 0:
        stats[0] = count[0]
        stats[1] = 0
        stats[2] = 0
        counters[2] += 1
    else:
        counters[3] += 1


class GPUCollisionSafeStep:
    """Allocated before graph capture; all query decisions stay on device."""
    def __init__(self, audit, skin=1.e-10, *, candidate_skin=0.,reject_prepass=False,cancel_rejected=False):
        self.audit = audit
        self.reject_prepass=bool(reject_prepass)
        self.cancel_rejected=bool(cancel_rejected)
        self.min_gap = audit.min_gap
        self.skin = positive_scalar(skin, 'CCD step certificate skin')
        self.gap = float(np.nextafter(audit.min_gap + self.skin, np.inf))
        if self.gap > np.finfo(np.float32).max / 4:
            raise ValueError('CCD step certificate gap exceeds bound range')
        d = audit.device
        self.origin = wp.zeros(audit.particle_count, dtype=wp.vec3d, device=d)
        self.endpoint = wp.zeros_like(self.origin)
        self.prefix = wp.ones(1, dtype=wp.float64, device=d)
        self.reference_alpha = wp.ones(1, dtype=wp.float64, device=d)
        self.safe_alpha = wp.ones(1, dtype=wp.float64, device=d)
        self.valid = wp.zeros(1, dtype=int, device=d)
        self.full = wp.zeros(1, dtype=int, device=d)
        # builds / limited / reused / original audit / overflow / no prefix
        self.counters = wp.zeros(6, dtype=int, device=d)
        self.saved_count = wp.zeros(1, dtype=int, device=d)
        self.candidate_skin=float(candidate_skin)
        if not np.isfinite(self.candidate_skin) or self.candidate_skin<0:
            raise ValueError('CCD candidate skin must be nonnegative and finite')
        self.candidate_cache=None
        if self.candidate_skin:
            from .ipc_gpu_swept_cache import GPUSweptCandidateCache
            self.candidate_cache=GPUSweptCandidateCache(audit,self.gap,self.candidate_skin)

    def reset(self):
        if self.candidate_cache is not None:self.candidate_cache.reset()

    def _validate_inputs(self, old, target, values):
        for array in (old, target):
            if (array.shape != (self.audit.particle_count,) or array.dtype != wp.vec3d
                    or array.device != self.audit.device):
                raise ValueError('CCD safe steps require matching float64 position arrays on the audit device')
        if (values.ndim != 1 or values.shape[0] < 3 or values.dtype != wp.float64
                or values.device != self.audit.device):
            raise ValueError('CCD safe steps require float64 line-search values on the audit device')

    def compute(self, old, endpoint, values):
        a, d = self.audit, self.audit.device
        self._validate_inputs(old, endpoint, values)
        if a.min_gap != self.min_gap:
            raise ValueError('rebuild CCD step certificate after changing minimum clearance')
        if not a.parallel_pairs or a.pairs is None:
            raise ValueError('CCD safe steps require configured parallel CCD buffers')
        wp.copy(self.origin, old)
        if endpoint is not self.endpoint:
            wp.copy(self.endpoint, endpoint)
        if self.candidate_cache is None:
            a._prepare_query(self.origin, self.endpoint, self.gap)
            a.pair_count.zero_()
            wp.launch(pairs.collect_points, len(a.vertices),
                      [a.invalid, a.groups, a.vertices, a.faces, *a.bounds[0], a.face_bvh.id,
                       a.pair_count, a.pairs, a.pair_meta], device=d)
            wp.launch(pairs.collect_edges, len(a.edges),
                      [a.invalid, a.groups, a.edges, *a.bounds[2], a.edge_bvh.id,
                       a.pair_count, a.pairs, a.pair_meta], device=d)
            invalid,count,ids,meta=a.invalid,a.pair_count,a.pairs,a.pair_meta
        else:
            cached=self.candidate_cache
            cached.query(self.origin,self.endpoint)
            invalid,count,ids,meta=cached.invalid,cached.count,cached.pairs,cached.meta
        self.prefix.fill_(1.)
        wp.launch(prefixes, a.pair_capacity,
                  [self.origin, self.endpoint, invalid, count, ids,
                   meta, self.gap, self.prefix], device=d, block_dim=a.query_block_dim)
        wp.launch(limit, 1, [invalid, count, a.pair_capacity, self.prefix, values,
                            self.reference_alpha, self.safe_alpha, self.valid, self.counters], device=d)
        # Original fallback audits overwrite the shared pair count; retain
        # the original certificate's count separately for later reuse.
        wp.copy(self.saved_count, count)

    def check(self, old, target, values):
        a, d = self.audit, self.audit.device
        self._validate_inputs(old, target, values)
        if a.min_gap != self.min_gap:
            raise ValueError('rebuild CCD step certificate after changing minimum clearance')
        wp.launch(begin, 1, [self.valid, values, self.safe_alpha, self.full], device=d)
        wp.launch(cache.guard, a.particle_count,
                  [old, target, self.origin, self.endpoint, values, self.reference_alpha,
                   wp.float64(self.skin), self.full], device=d)
        wp.launch(publish, 1, [self.full, self.saved_count, a.stats, self.counters], device=d)
        wp.capture_if(self.full, lambda: a.check(old, target, readback=False,
            stop_on_witness=self.reject_prepass,stop_after_rejection=self.cancel_rejected))
