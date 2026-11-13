"""GPU swept candidate reuse, with both endpoints guarded on every query.

Only broad-phase candidates are cached. Prefixes are recomputed on the current
path. Each coordinate moves less than skin/2 at both saved endpoints, including
roundoff. Each swept AABB face thus moves less than skin/2; expanding the saved
boxes by skin includes every pair relevant at the original gap. Saved pairs/counts
are private: seed/final audits and fallback queries use the audit's own buffers.
"""
import numpy as np
import warp as wp
from .ipc import positive_scalar
from . import ipc_gpu_ccd_pairs as pairs

wp.set_module_options({'enable_backward': False})


@wp.kernel
def start(valid:wp.array(dtype=int),rebuild:wp.array(dtype=int)):
    rebuild[0]=int(valid[0]==0)


@wp.kernel
def guard(old:wp.array(dtype=wp.vec3d),target:wp.array(dtype=wp.vec3d),
          origin:wp.array(dtype=wp.vec3d),endpoint:wp.array(dtype=wp.vec3d),
          skin:wp.float64,rebuild:wp.array(dtype=int)):
    i=wp.tid()
    if rebuild[0]:return
    delta=wp.float64(0.);scale=wp.float64(1e-30);finite=bool(True)
    for k in range(3):
        p,q,a,b=old[i][k],target[i][k],origin[i][k],endpoint[i][k]
        finite=finite and wp.isfinite(p) and wp.isfinite(q) and wp.isfinite(a) and wp.isfinite(b)
        delta=wp.max(delta,wp.max(wp.abs(p-a),wp.abs(q-b)))
        scale=wp.max(scale,wp.max(wp.max(wp.abs(p),wp.abs(q)),wp.max(wp.abs(a),wp.abs(b))))
    rounding=wp.float64(128.*2.220446049250313e-16)*scale
    if not finite or not wp.isfinite(delta) or wp.float64(2.)*(delta+rounding)>=skin:
        wp.atomic_max(rebuild,0,1)


@wp.kernel
def classify(invalid:wp.array(dtype=int),count:wp.array(dtype=int),capacity:int,
             valid:wp.array(dtype=int),counters:wp.array(dtype=int)):
    valid[0]=int(invalid[0]==0 and count[0]<=capacity)
    counters[0]+=1
    if count[0]>capacity:counters[2]+=1


@wp.kernel
def count_reuse(rebuild:wp.array(dtype=int),counters:wp.array(dtype=int)):
    if rebuild[0]==0:counters[1]+=1


class GPUSweptCandidateCache:
    def __init__(self,audit,gap,skin):
        self.audit=audit;self.skin=positive_scalar(skin,'CCD candidate skin')
        self.capacity=audit.pair_capacity
        self.gap=float(np.nextafter(gap+self.skin,np.inf))
        if not np.isfinite(self.gap) or self.gap>np.finfo(np.float32).max/4:
            raise ValueError('CCD candidate bound exceeds float32 range')
        d=audit.device
        self.origin=wp.zeros(audit.particle_count,dtype=wp.vec3d,device=d)
        self.endpoint=wp.zeros_like(self.origin)
        self.valid=wp.zeros(1,dtype=int,device=d)
        self.rebuild=wp.ones(1,dtype=int,device=d)
        self.invalid=wp.zeros(1,dtype=int,device=d)
        self.count=wp.zeros(1,dtype=int,device=d)
        self.pairs=wp.empty(self.capacity,dtype=wp.vec4i,device=d)
        self.meta=wp.empty(self.capacity,dtype=wp.vec3i,device=d)
        self.counters=wp.zeros(3,dtype=int,device=d)  # rebuild / reuse / overflow

    def reset(self):
        self.valid.zero_()

    def query(self,old,target):
        a=self.audit;d=a.device
        if a.pair_capacity!=self.capacity:
            raise ValueError('rebuild swept cache after changing pair capacity')
        wp.launch(start,1,[self.valid,self.rebuild],device=d)
        wp.launch(guard,a.particle_count,[old,target,self.origin,self.endpoint,self.skin,self.rebuild],device=d)
        def rebuild():
            a._prepare_query(old,target,self.gap)
            self.count.zero_()
            wp.launch(pairs.collect_points,len(a.vertices),[a.invalid,a.groups,a.vertices,a.faces,
                *a.bounds[0],a.face_bvh.id,self.count,self.pairs,self.meta],device=d)
            wp.launch(pairs.collect_edges,len(a.edges),[a.invalid,a.groups,a.edges,
                *a.bounds[2],a.edge_bvh.id,self.count,self.pairs,self.meta],device=d)
            wp.copy(self.invalid,a.invalid)
            wp.copy(self.origin,old);wp.copy(self.endpoint,target)
            wp.launch(classify,1,[self.invalid,self.count,self.capacity,self.valid,self.counters],device=d)
        wp.capture_if(self.rebuild,rebuild)
        wp.launch(count_reuse,1,[self.rebuild,self.counters],device=d)
        # Reusing a private candidate list must not inherit flags from a
        # previous failed seed/final audit. Actual current prefixes run next.
        a.stats.zero_()
