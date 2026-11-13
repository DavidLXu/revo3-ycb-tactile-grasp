"""Separate swept BVH traversal from divergent per-pair CCD certification.

If storage overflows, the caller runs the complete original GPU traversal.
No candidate is truncated and no host count readback is needed.
"""
from typing import Any
import warp as wp
from .ipc_precision import position_overloads, scalar_value
from .ipc_self_audit import same_rigid, _certify, _feature_motion_prefix
from .ipc_primitives_warp import _weights, _delta

wp.set_module_options({'enable_backward': False})


@wp.kernel
def collect_points(invalid:wp.array(dtype=int),groups:wp.array(dtype=int),
        vertices:wp.array2d(dtype=int),faces:wp.array2d(dtype=int),
        lower:wp.array(dtype=wp.vec3),upper:wp.array(dtype=wp.vec3),bvh:wp.uint64,
        count:wp.array(dtype=int),pairs:wp.array(dtype=wp.vec4i),meta:wp.array(dtype=wp.vec3i)):
    if invalid[0]:return
    i=wp.tid();v=vertices[i,0];query=wp.bvh_query_aabb(bvh,lower[i],upper[i]);j=int(0)
    while wp.bvh_query_next(query,j):
        a,b,c=faces[j,0],faces[j,1],faces[j,2]
        if v!=a and v!=b and v!=c and not same_rigid(groups,v,a,b,c):
            slot=wp.atomic_add(count,0,1)
            if slot<pairs.shape[0]:
                pairs[slot]=wp.vec4i(v,a,b,c);meta[slot]=wp.vec3i(0,v,j)


@wp.kernel
def collect_edges(invalid:wp.array(dtype=int),groups:wp.array(dtype=int),edges:wp.array2d(dtype=int),
        lower:wp.array(dtype=wp.vec3),upper:wp.array(dtype=wp.vec3),bvh:wp.uint64,
        count:wp.array(dtype=int),pairs:wp.array(dtype=wp.vec4i),meta:wp.array(dtype=wp.vec3i)):
    if invalid[0]:return
    i=wp.tid();a,b=edges[i,0],edges[i,1];query=wp.bvh_query_aabb(bvh,lower[i],upper[i]);j=int(0)
    while wp.bvh_query_next(query,j):
        c,d=edges[j,0],edges[j,1]
        if j>i and a!=c and a!=d and b!=c and b!=d and not same_rigid(groups,a,b,c,d):
            slot=wp.atomic_add(count,0,1)
            if slot<pairs.shape[0]:
                pairs[slot]=wp.vec4i(a,b,c,d);meta[slot]=wp.vec3i(1,i,j)


@wp.kernel
def classify(count:wp.array(dtype=int),capacity:int,enabled:wp.array(dtype=int),
             stats:wp.array(dtype=int),work:wp.array(dtype=int)):
    overflow=count[0]>capacity
    enabled[0]=int(overflow)
    work[0]+=1
    if overflow:work[1]+=1
    else:stats[0]=count[0]


@position_overloads('old','target',scalars={'min_gap':wp.float64})
@wp.kernel
def certify_pairs(old:wp.array(dtype=Any),target:wp.array(dtype=Any),invalid:wp.array(dtype=int),
        enabled:wp.array(dtype=int),count:wp.array(dtype=int),pairs:wp.array(dtype=wp.vec4i),
        meta:wp.array(dtype=wp.vec3i),min_gap:Any,stats:wp.array(dtype=int),witness:wp.array2d(dtype=int)):
    i=wp.tid()
    if invalid[0] or enabled[0] or i>=count[0]:return
    m=meta[i]
    if not _certify(old,target,pairs[i],m[0],scalar_value(min_gap)):
        slot=wp.atomic_add(stats,1,1)
        if slot<witness.shape[0]:
            for k in range(3):witness[slot,k]=m[k]


@wp.func
def midpoint_witness(a0:wp.vec3d,b0:wp.vec3d,c0:wp.vec3d,d0:wp.vec3d,
                     a1:wp.vec3d,b1:wp.vec3d,c1:wp.vec3d,d1:wp.vec3d,
                     kind:int,gap:wp.float64):
    # A closest-feature result is only used as a witness UPPER bound, never
    # as a separation certificate. Round outward, including interpolation
    # and barycentric evaluation. Ambiguous cases use the original audit.
    a=(a0+a1)*wp.float64(.5);b=(b0+b1)*wp.float64(.5)
    c=(c0+c1)*wp.float64(.5);d=(d0+d1)*wp.float64(.5)
    scale=wp.max(wp.max(wp.length(a0),wp.length(b0)),wp.max(wp.length(c0),wp.length(d0)))
    scale=wp.max(scale,wp.max(wp.max(wp.length(a1),wp.length(b1)),wp.max(wp.length(c1),wp.length(d1))))
    # Local coordinates avoid subtracting large weighted world positions.
    b,c,d=b-a,c-a,d-a
    a=wp.vec3d(0.)
    w=_weights(a,b,c,d,kind)
    valid=w[0]>=wp.float64(0.) and w[2]<=wp.float64(0.) and w[3]<=wp.float64(0.)
    if kind==0:
        valid=valid and w[0]==wp.float64(1.) and w[1]<=wp.float64(0.)
        valid=valid and wp.abs(w[1]+w[2]+w[3]+wp.float64(1.))<wp.float64(32.*2.220446049250313e-16)
    else:
        valid=valid and w[1]>=wp.float64(0.)
        valid=valid and wp.abs(w[0]+w[1]-wp.float64(1.))<wp.float64(32.*2.220446049250313e-16)
        valid=valid and wp.abs(w[2]+w[3]+wp.float64(1.))<wp.float64(32.*2.220446049250313e-16)
    distance=wp.length(_delta(w,a,b,c,d))
    # Includes repeated endpoint interpolation (at most 32 contractions).
    # At huge world offsets this deliberately disables the early rejection.
    rounding=wp.float64(16384.*2.220446049250313e-16)*scale
    return valid and wp.isfinite(distance) and wp.isfinite(rounding) and distance+rounding<gap


@wp.func
def midpoint_collision(old:wp.array(dtype=Any),target:wp.array(dtype=Any),ids:wp.vec4i,kind:int,gap:wp.float64):
    return midpoint_witness(wp.vec3d(old[ids[0]]),wp.vec3d(old[ids[1]]),wp.vec3d(old[ids[2]]),wp.vec3d(old[ids[3]]),
        wp.vec3d(target[ids[0]]),wp.vec3d(target[ids[1]]),wp.vec3d(target[ids[2]]),wp.vec3d(target[ids[3]]),kind,gap)


@wp.kernel
def rejection_witnesses(old:wp.array(dtype=wp.vec3d),target:wp.array(dtype=wp.vec3d),
        invalid:wp.array(dtype=int),overflow:wp.array(dtype=int),count:wp.array(dtype=int),
        pairs:wp.array(dtype=wp.vec4i),meta:wp.array(dtype=wp.vec3i),gap:wp.float64,
        stats:wp.array(dtype=int),witness:wp.array2d(dtype=int)):
    """Only reject with a witness at t=.25,.5,.75,1; never certify by sampling."""
    i=wp.tid()
    if invalid[0] or overflow[0] or i>=count[0]:return
    ids=pairs[i];kind=meta[i][0]
    hit=midpoint_collision(old,target,ids,kind,gap)
    if not hit:hit=midpoint_collision(target,target,ids,kind,gap)
    if not hit:
        a,b,c,d=old[ids[0]],old[ids[1]],old[ids[2]],old[ids[3]]
        aa,bb,cc,dd=target[ids[0]],target[ids[1]],target[ids[2]],target[ids[3]]
        am,bm,cm,dm=(a+aa)*wp.float64(.5),(b+bb)*wp.float64(.5),(c+cc)*wp.float64(.5),(d+dd)*wp.float64(.5)
        hit=midpoint_witness(a,b,c,d,am,bm,cm,dm,kind,gap)
        if not hit:hit=midpoint_witness(am,bm,cm,dm,aa,bb,cc,dd,kind,gap)
    if hit:
        slot=wp.atomic_add(stats,1,1)
        if slot<witness.shape[0]:
            for k in range(3):witness[slot,k]=meta[i][k]


@wp.kernel
def needs_full_certificate(stats:wp.array(dtype=int),run:wp.array(dtype=int)):
    run[0]=int(stats[1]==0)


@wp.func
def fast_certify(old:wp.array(dtype=Any),target:wp.array(dtype=Any),ids:wp.vec4i,kind:int,gap:wp.float64):
    a,b,c,d=wp.vec3d(old[ids[0]]),wp.vec3d(old[ids[1]]),wp.vec3d(old[ids[2]]),wp.vec3d(old[ids[3]])
    aa,bb,cc,dd=wp.vec3d(target[ids[0]]),wp.vec3d(target[ids[1]]),wp.vec3d(target[ids[2]]),wp.vec3d(target[ids[3]])
    certified=bool(False)
    for contraction in range(32):
        # Only a positive collision witness terminates early. All acceptance
        # paths and uncovered-interval updates match the original certificate.
        if midpoint_witness(a,b,c,d,aa,bb,cc,dd,kind,gap):break
        prefix=_feature_motion_prefix(a,b,c,d,aa,bb,cc,dd,kind,gap)
        if prefix>=wp.float64(1.):
            certified=True
            break
        suffix=_feature_motion_prefix(aa,bb,cc,dd,a,b,c,d,kind,gap)
        if prefix+suffix>=wp.float64(1.):
            certified=True
            break
        if prefix+suffix<=wp.float64(1e-12):break
        va,vb,vc,vd=aa-a,bb-b,cc-c,dd-d
        aa,bb,cc,dd=aa-suffix*va,bb-suffix*vb,cc-suffix*vc,dd-suffix*vd
        a,b,c,d=a+prefix*va,b+prefix*vb,c+prefix*vc,d+prefix*vd
    return certified


@position_overloads('old','target',scalars={'min_gap':wp.float64})
@wp.kernel
def certify_pairs_fast(old:wp.array(dtype=Any),target:wp.array(dtype=Any),invalid:wp.array(dtype=int),
        enabled:wp.array(dtype=int),count:wp.array(dtype=int),pairs:wp.array(dtype=wp.vec4i),
        meta:wp.array(dtype=wp.vec3i),min_gap:Any,stats:wp.array(dtype=int),witness:wp.array2d(dtype=int)):
    i=wp.tid()
    if invalid[0] or enabled[0] or i>=count[0]:return
    m=meta[i];gap=scalar_value(min_gap)
    unsafe=not fast_certify(old,target,pairs[i],m[0],gap)
    if unsafe:
        slot=wp.atomic_add(stats,1,1)
        if slot<witness.shape[0]:
            for k in range(3):witness[slot,k]=m[k]


@wp.func_native(r'''
#ifdef __CUDA_ARCH__
// One atomic load per currently active warp, also inside divergent loops.
// All writes to this counter are atomic. A cancelled thread contributes
// neither a new failure nor a spurious diagnostic witness.
unsigned mask = __activemask();
int leader = __ffs(mask) - 1;
int value = 0;
if ((threadIdx.x & 31) == leader) value = atomicAdd(&stats.data[1], 0);
return __shfl_sync(mask, value, leader);
#else
return stats.data[1];
#endif
''')
def query_rejected(stats:wp.array(dtype=int))->int: ...


@wp.func
def boolean_certify(old:wp.array(dtype=wp.vec3d),target:wp.array(dtype=wp.vec3d),
                    ids:wp.vec4i,kind:int,gap:wp.float64,fast:bool,stats:wp.array(dtype=int)):
    # Same certificate as fast_certify/certify_feature_motion. -1 denotes
    # cancellation AFTER another pair has already rejected this whole query.
    a,b,c,d=wp.vec3d(old[ids[0]]),wp.vec3d(old[ids[1]]),wp.vec3d(old[ids[2]]),wp.vec3d(old[ids[3]])
    aa,bb,cc,dd=wp.vec3d(target[ids[0]]),wp.vec3d(target[ids[1]]),wp.vec3d(target[ids[2]]),wp.vec3d(target[ids[3]])
    for contraction in range(32):
        if query_rejected(stats)!=0:return int(-1)
        if fast and midpoint_witness(a,b,c,d,aa,bb,cc,dd,kind,gap):return int(0)
        prefix=_feature_motion_prefix(a,b,c,d,aa,bb,cc,dd,kind,gap)
        if prefix>=wp.float64(1.):return int(1)
        if query_rejected(stats)!=0:return int(-1)
        suffix=_feature_motion_prefix(aa,bb,cc,dd,a,b,c,d,kind,gap)
        if prefix+suffix>=wp.float64(1.):return int(1)
        if prefix+suffix<=wp.float64(1e-12):return int(0)
        va,vb,vc,vd=aa-a,bb-b,cc-c,dd-d
        aa,bb,cc,dd=aa-suffix*va,bb-suffix*vb,cc-suffix*vc,dd-suffix*vd
        a,b,c,d=a+prefix*va,b+prefix*vb,c+prefix*vc,d+prefix*vd
    return int(0)


@wp.kernel
def certify_pairs_boolean(old:wp.array(dtype=wp.vec3d),target:wp.array(dtype=wp.vec3d),
        invalid:wp.array(dtype=int),overflow:wp.array(dtype=int),count:wp.array(dtype=int),
        pairs:wp.array(dtype=wp.vec4i),meta:wp.array(dtype=wp.vec3i),gap:wp.float64,
        stats:wp.array(dtype=int),witness:wp.array2d(dtype=int),fast:bool):
    i=wp.tid()
    if invalid[0] or overflow[0] or i>=count[0]:return
    m=meta[i]
    result=boolean_certify(old,target,pairs[i],m[0],gap,fast,stats)
    if result==0:
        slot=wp.atomic_add(stats,1,1)
        if slot<witness.shape[0]:
            for k in range(3):witness[slot,k]=m[k]
