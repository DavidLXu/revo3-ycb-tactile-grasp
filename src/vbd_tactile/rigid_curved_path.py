"""GPU certificates for affine FEM motion and prescribed rotating boundaries.

Only the geometric query is subdivided. Its inputs remain the endpoints of one
physical time step. Each feature/tet traverses a bounded dyadic tree on-device;
an exhausted tree is an uncertified path, never a successful collision check.
"""
import numpy as np
import warp as wp
from .rigid_dynamics_gpu import exponential, rotation
from .ipc_self_audit import certify_feature_motion, same_rigid, _motion_status
from .ipc_vbd_kernels import determinant_path_extremum

wp.set_module_options({"enable_backward": False})


@wp.struct
class Curve:
    old: wp.array(dtype=wp.vec3d)
    target: wp.array(dtype=wp.vec3d)
    owner: wp.array(dtype=int)
    local: wp.array(dtype=wp.vec3d)
    start: wp.array(dtype=wp.mat44d)
    end: wp.array(dtype=wp.mat44d)
    rotvec: wp.array(dtype=wp.vec3d)


@wp.func
def point(c: Curve, i: int, t: wp.float64):
    if t == wp.float64(0.):
        return c.old[i]
    if t == wp.float64(1.):
        return c.target[i]
    p = c.old[i] + t * (c.target[i] - c.old[i])
    owner = c.owner[i]
    if owner >= 0:
        r0, r1 = rotation(c.start[owner]), rotation(c.end[owner])
        rt = exponential(t*c.rotvec[owner])*r0
        # Add the rotational arc to the endpoint chord; translation is affine.
        p += (rt - ((wp.float64(1.)-t)*r0 + t*r1))*c.local[i]
    return p


@wp.func
def arc_error(c: Curve, i: int, width: wp.float64):
    result = wp.float64(0.)
    owner = c.owner[i]
    if owner >= 0:
        angle = wp.length(c.rotvec[owner])*width
        radius = wp.length(c.local[i])
        # Linear interpolation error <= max|x''| h^2 / 8. This bound also
        # handles tangential chord error; it is not a midpoint-only test.
        result = radius*angle*angle/wp.float64(8.)
        if wp.length(c.rotvec[owner]) > wp.float64(0.):
            result += wp.float64(256.*2.220446049250313e-16)*radius
    return result


@wp.func
def next_node(node: int):
    n = node
    while n > 1 and n % 2 == 1:
        n = n // 2
    if n == 1:
        return int(0)
    return n + 1


@wp.func
def interval(node: int):
    base = int(1)
    width = wp.float64(1.)
    while base*2 <= node:
        base *= 2
        width *= wp.float64(.5)
    return wp.vec2d(wp.float64(node-base)*width, width)


@wp.func
def certify(c: Curve, ids: wp.vec4i, kind: int, gap: wp.float64,
            max_checks: int, work: wp.array(dtype=int)):
    node = int(1)
    used = int(0)
    while node > 0 and used < max_checks:
        used += 1
        span = interval(node)
        lo, hi = span[0], span[0]+span[1]
        a,b,d,e = point(c,ids[0],lo),point(c,ids[1],lo),point(c,ids[2],lo),point(c,ids[3],lo)
        aa,bb,dd,ee = point(c,ids[0],hi),point(c,ids[1],hi),point(c,ids[2],hi),point(c,ids[3],hi)
        er = wp.vec4d(arc_error(c,ids[0],span[1]),arc_error(c,ids[1],span[1]),
                     arc_error(c,ids[2],span[1]),arc_error(c,ids[3],span[1]))
        padding = er[0]+wp.max(er[1],wp.max(er[2],er[3]))
        if kind == 1:
            padding = wp.max(er[0],er[1])+wp.max(er[2],er[3])
        good = certify_feature_motion(a,b,d,e,aa,bb,dd,ee,kind,gap+padding)
        if good:
            node = next_node(node)
        elif padding == wp.float64(0.) or node >= 16777216:
            break
        else:
            node *= 2
    wp.atomic_add(work,0,used)
    if node != 0:
        wp.atomic_add(work,2,1)
    return node == 0


@wp.kernel
def bounds(c: Curve, features: wp.array2d(dtype=int), width: int, gap: wp.float64,
           lower: wp.array(dtype=wp.vec3), upper: wp.array(dtype=wp.vec3),
           invalid: wp.array(dtype=int)):
    i = wp.tid()
    lo,hi = wp.vec3d(wp.inf),wp.vec3d(-wp.inf)
    valid = bool(True)
    for j in range(width):
        v = features[i,j]
        error = arc_error(c,v,wp.float64(1.))
        a,b = c.old[v],c.target[v]
        for k in range(3):
            valid = valid and wp.isfinite(a[k]) and wp.isfinite(b[k]) and wp.isfinite(error)
        lo = wp.min(lo,wp.min(a,b)-wp.vec3d(error))
        hi = wp.max(hi,wp.max(a,b)+wp.vec3d(error))
    scale = wp.max(wp.length(lo),wp.length(hi))
    pad = gap+wp.float64(8.*1.1920928955078125e-7)*scale
    lower[i] = wp.vec3(lo-wp.vec3d(pad))
    upper[i] = wp.vec3(hi+wp.vec3d(pad))
    for k in range(3):
        valid = valid and wp.isfinite(lower[i][k]) and wp.isfinite(upper[i][k])
    if not valid:
        wp.atomic_add(invalid,0,1)
        lower[i] = wp.vec3(0.)
        upper[i] = wp.vec3(0.)


@wp.kernel
def features(c: Curve, groups: wp.array(dtype=int), first: wp.array2d(dtype=int),
             second: wp.array2d(dtype=int), lower: wp.array(dtype=wp.vec3),
             upper: wp.array(dtype=wp.vec3), bvh: wp.uint64, kind: int,
             gap: wp.float64, max_checks: int, invalid: wp.array(dtype=int),
             stats: wp.array(dtype=int), witness: wp.array2d(dtype=int), work: wp.array(dtype=int)):
    if invalid[0]:
        return
    i = wp.tid()
    query = wp.bvh_query_aabb(bvh,lower[i],upper[i])
    j = int(0)
    while wp.bvh_query_next(query,j):
        a,b,d,e = int(first[i,0]),int(second[j,0]),int(second[j,1]),int(second[j,2 if kind == 0 else 1])
        eligible = a != b and a != d and a != e
        if kind == 1:
            a,b,d,e = int(first[i,0]),int(first[i,1]),int(second[j,0]),int(second[j,1])
            eligible = j > i and a != d and a != e and b != d and b != e
        if eligible and not same_rigid(groups,a,b,d,e):
            wp.atomic_add(stats,0,1)
            if not certify(c,wp.vec4i(a,b,d,e),kind,gap,max_checks,work):
                slot = wp.atomic_add(stats,1,1)
                if slot < witness.shape[0]:
                    witness[slot,0] = kind
                    witness[slot,1] = i
                    witness[slot,2] = j


@wp.kernel
def tet_paths(c: Curve, tets: wp.array2d(dtype=int), inverse: wp.array(dtype=wp.mat33),
              min_j: wp.float64, max_checks: int, diagnostic: wp.array(dtype=wp.float64),
              work: wp.array(dtype=int)):
    t = wp.tid()
    ids = wp.vec4i(tets[t,0],tets[t,1],tets[t,2],tets[t,3])
    dm = wp.mat33d(inverse[t])
    node,used = int(1),int(0)
    certified_min = wp.float64(wp.inf)
    while node > 0 and used < max_checks:
        used += 1
        span = interval(node)
        lo,hi = span[0],span[0]+span[1]
        a,b,d,e = point(c,ids[0],lo),point(c,ids[1],lo),point(c,ids[2],lo),point(c,ids[3],lo)
        aa,bb,dd,ee = point(c,ids[0],hi),point(c,ids[1],hi),point(c,ids[2],hi),point(c,ids[3],hi)
        f0 = wp.matrix_from_cols(b-a,d-a,e-a)*dm
        f1 = wp.matrix_from_cols(bb-aa,dd-aa,ee-aa)*dm
        er = wp.vec4d(arc_error(c,ids[0],span[1]),arc_error(c,ids[1],span[1]),
                     arc_error(c,ids[2],span[1]),arc_error(c,ids[3],span[1]))
        edge_error = wp.length(wp.vec3d(er[1]+er[0],er[2]+er[0],er[3]+er[0]))
        n0,n1,nd = wp.float64(0.),wp.float64(0.),wp.float64(0.)
        for j in range(3):
            for k in range(3):
                n0 += f0[j,k]*f0[j,k]
                n1 += f1[j,k]*f1[j,k]
                nd += dm[j,k]*dm[j,k]
        alpha,eps = wp.sqrt(wp.max(n0,n1)),edge_error*wp.sqrt(nd)
        lower = determinant_path_extremum(f0,f1-f0)[0] - (wp.float64(3.)*alpha*alpha*eps+wp.float64(3.)*alpha*eps*eps+eps*eps*eps)
        if wp.isfinite(lower) and lower > min_j:
            certified_min = wp.min(certified_min,lower)
            node = next_node(node)
        elif edge_error == wp.float64(0.) or node >= 16777216:
            break
        else:
            node *= 2
    wp.atomic_add(work,1,used)
    if node != 0:
        certified_min = wp.float64(-wp.inf)
        wp.atomic_add(work,2,1)
    wp.atomic_min(diagnostic,3,certified_min)


class CurvedPathAudit:
    def __init__(self, solver):
        self.s = solver
        self.max_checks = 4096
        owners = np.full(solver.model.particle_count,-1,dtype=np.int32)
        local = np.zeros((solver.model.particle_count,3))
        ids = solver._pin_ids.numpy()
        owners[ids] = solver._pin_owners.numpy()
        local[ids] = solver._pin_local.numpy()
        self.owners = wp.array(owners,dtype=int,device=solver.model.device)
        self.local = wp.array(local,dtype=wp.vec3d,device=solver.model.device)
        self.rotvec = wp.zeros(len(solver.mount_poses),dtype=wp.vec3d,device=solver.model.device)
        self.work = wp.zeros(3,dtype=int,device=solver.model.device)

    def check(self, graph):
        from .ipc_gpu_control import reset_path
        s,motion = self.s,self.s.self_motion
        c = Curve()
        c.old,c.target,c.owner,c.local = s.old,s.positions,self.owners,self.local
        c.start,c.end,c.rotvec = s._start_mount_device,s._mount_device,self.rotvec
        self.work.zero_()
        motion.stats.zero_()
        motion.invalid.zero_()
        graph.launch(reset_path,1,[s.diagnostic])
        for mesh,limits in zip((motion.vertices,motion.faces,motion.edges),motion.bounds):
            graph.launch(bounds,len(mesh),[c,mesh,mesh.shape[1],motion.min_gap,*limits,motion.invalid])
        # The graph's preparation builds these trees before recording. This
        # refit uses arc-inflated bounds, covering the full curved trajectory.
        motion.face_bvh.refit()
        motion.edge_bvh.refit()
        for kind,first,second,limits,bvh in (
            (0,motion.vertices,motion.faces,motion.bounds[0],motion.face_bvh),
            (1,motion.edges,motion.edges,motion.bounds[2],motion.edge_bvh)):
            graph.launch(features,len(first),[c,motion.groups,first,second,*limits,bvh.id,kind,
                motion.min_gap,self.max_checks,motion.invalid,motion.stats,motion.witness,self.work])
        graph.launch(_motion_status,1,[motion.invalid,motion.stats])
        graph.launch(tet_paths,s.model.tet_count,[c,s.model.tet_indices,s.model.tet_poses,
                     s.min_jacobian,self.max_checks,s.diagnostic,self.work])


@wp.kernel
def report_work(work: wp.array(dtype=int), output: wp.array2d(dtype=wp.float64)):
    for i in range(3):
        output[0,18+i] = wp.float64(work[i])
