"""GPU kernels for prescribed triangle-mesh boundaries and solid containment."""
from typing import Any
import warp as wp
from .ipc_precision import position_overloads
from .ipc_primitives_warp import _weights, _delta
from .mounted_ipc import transform_point
from .ipc_vbd_kernels import determinant_path_extremum


@position_overloads('old', 'velocity', 'positions', poses=('c0', 'c1'))
@wp.kernel
def initialize_mesh(old: wp.array(dtype=Any), velocity: wp.array(dtype=Any),
    external_force: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
    gravity: wp.array(dtype=wp.vec3), worlds: wp.array(dtype=int), radii: wp.array(dtype=wp.float64),
    c0: Any, c1: Any, radius: wp.float64, dhat: wp.float64, stiffness: wp.float64,
    dt: wp.float64, min_gap: wp.float64, positions: wp.array(dtype=Any),
    inertia: wp.array(dtype=wp.vec3d), normals: wp.array(dtype=wp.vec3d), loads: wp.array(dtype=wp.float64)):
    i = wp.tid()
    x = wp.vec3d(old[i])
    positions[i] = old[i]
    inertia[i] = x
    normals[i] = wp.vec3d(0.)
    loads[i] = wp.float64(0.)
    if mass[i] > 0.:
        inertia[i] = x+dt*wp.vec3d(velocity[i])+dt*dt*(wp.vec3d(gravity[worlds[i]])+wp.vec3d(external_force[i])/wp.float64(mass[i]))


@wp.kernel
def mesh_seed(old: wp.array(dtype=wp.vec3d), mass: wp.array(dtype=float),
              owners: wp.array(dtype=int), delta: wp.array(dtype=wp.mat44d),
              faces: wp.array2d(dtype=int), bvh: wp.uint64, skin: wp.array(dtype=wp.float64),
              positions: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if mass[i] > 0.:
        p = old[i]
        base = transform_point(delta[owners[i]], p)
        best = skin[0]
        displacement = wp.vec3d(0.)
        pad = float(skin[0])+8.*1.1920928955078125e-7*wp.length(wp.vec3(p))
        query = wp.bvh_query_aabb(bvh, wp.vec3(p)-wp.vec3(pad), wp.vec3(p)+wp.vec3(pad))
        f = int(0)
        while wp.bvh_query_next(query, f):
            ia, ib, ic = faces[f, 0], faces[f, 1], faces[f, 2]
            a, b, c = old[ia], old[ib], old[ic]
            rel = _delta(_weights(p, a, b, c, 0), p, a, b, c)
            distance = wp.length(rel)
            if distance < best:
                best = distance
                nearest = p-rel
                displacement = transform_point(delta[owners[ia]], nearest)-nearest-(base-p)
        weight = wp.clamp((skin[0]-best)/(skin[0]*wp.float64(.5)), wp.float64(0.), wp.float64(1.))
        positions[i] = base+weight*displacement


@wp.kernel
def inside_closed_rigid(x: wp.array(dtype=wp.vec3d), groups: wp.array(dtype=int),
    faces: wp.array2d(dtype=int), offsets: wp.array(dtype=int), status: wp.array(dtype=int)):
    i, body = wp.tid()
    if groups[i] == body:
        return
    total = wp.float64(0.)
    p = x[i]
    for f in range(offsets[body], offsets[body+1]):
        a, b, c = x[faces[f, 0]]-p, x[faces[f, 1]]-p, x[faces[f, 2]]-p
        la, lb, lc = wp.length(a), wp.length(b), wp.length(c)
        numerator = wp.dot(a, wp.cross(b, c))
        denominator = la*lb*lc+wp.dot(a,b)*lc+wp.dot(b,c)*la+wp.dot(c,a)*lb
        total += wp.float64(2.)*wp.atan2(numerator, denominator)
    if not wp.isfinite(total) or wp.abs(total) > wp.float64(2.*3.141592653589793):
        wp.atomic_max(status, 0, 31)


@wp.kernel
def rigid_inside_tets(x: wp.array(dtype=wp.vec3d), ids: wp.array(dtype=int),
    tets: wp.array2d(dtype=int), bvh: wp.uint64, status: wp.array(dtype=int)):
    p = x[ids[wp.tid()]]
    pad = float(8.*1.1920928955078125e-7)*wp.length(wp.vec3(p))+1.e-12
    query = wp.bvh_query_aabb(bvh, wp.vec3(p)-wp.vec3(pad), wp.vec3(p)+wp.vec3(pad))
    t = int(0)
    while wp.bvh_query_next(query, t):
        a = x[tets[t, 0]]
        matrix = wp.matrix_from_cols(x[tets[t, 1]]-a, x[tets[t, 2]]-a, x[tets[t, 3]]-a)
        q = wp.inverse(matrix)*(p-a)
        if wp.min(q[0], wp.min(q[1], q[2])) >= wp.float64(-1.e-12) and q[0]+q[1]+q[2] <= wp.float64(1.+1.e-12):
            wp.atomic_max(status, 0, 31)


@wp.kernel
def check_bindings(x: wp.array(dtype=wp.vec3d), ids: wp.array(dtype=int), owners: wp.array(dtype=int),
    local: wp.array(dtype=wp.vec3d), poses: wp.array(dtype=wp.mat44d), status: wp.array(dtype=int)):
    i = wp.tid()
    expected = transform_point(poses[owners[i]], local[i])
    if wp.length(x[ids[i]]-expected) > wp.float64(1.e-12):
        wp.atomic_max(status, 0, 30)


@wp.func
def triangle_axis_separates(a: wp.mat33d, b: wp.mat33d, axis: wp.vec3d):
    length = wp.length(axis)
    if length <= wp.float64(0.) or not wp.isfinite(length):
        return False
    n = axis/length
    alo, ahi, blo, bhi = wp.float64(wp.inf),wp.float64(-wp.inf),wp.float64(wp.inf),wp.float64(-wp.inf)
    for j in range(3):
        av, bv = wp.dot(a[j],n),wp.dot(b[j],n)
        alo,ahi,blo,bhi = wp.min(alo,av),wp.max(ahi,av),wp.min(blo,bv),wp.max(bhi,bv)
    return ahi < blo or bhi < alo


@wp.func
def triangle_separated(a: wp.mat33d, b: wp.mat33d):
    na,nb = wp.cross(a[1]-a[0],a[2]-a[0]),wp.cross(b[1]-b[0],b[2]-b[0])
    if triangle_axis_separates(a,b,na) or triangle_axis_separates(a,b,nb):
        return True
    for i in range(3):
        ea,eb = a[(i+1)%3]-a[i],b[(i+1)%3]-b[i]
        if triangle_axis_separates(a,b,wp.cross(na,ea)) or triangle_axis_separates(a,b,wp.cross(nb,eb)):
            return True
        for j in range(3):
            if triangle_axis_separates(a,b,wp.cross(ea,b[(j+1)%3]-b[j])):
                return True
    return False


@wp.kernel
def rigid_surface_intersections(x: wp.array(dtype=wp.vec3d), groups: wp.array(dtype=int),
    rigid_faces: wp.array2d(dtype=int), all_faces: wp.array2d(dtype=int),
    bvh: wp.uint64, self_check: bool, status: wp.array(dtype=int)):
    i = wp.tid()
    ids = wp.vec3i(rigid_faces[i,0],rigid_faces[i,1],rigid_faces[i,2])
    a,b,c = x[ids[0]],x[ids[1]],x[ids[2]]
    pad = 8.*1.1920928955078125e-7*float(wp.max(wp.length(a),wp.max(wp.length(b),wp.length(c))))+1.e-12
    query = wp.bvh_query_aabb(bvh,wp.vec3(wp.min(a,wp.min(b,c)))-wp.vec3(pad),wp.vec3(wp.max(a,wp.max(b,c)))+wp.vec3(pad))
    j = int(0)
    while wp.bvh_query_next(query,j):
        u,v,w = all_faces[j,0],all_faces[j,1],all_faces[j,2]
        shared = (u==ids[0] or u==ids[1] or u==ids[2] or v==ids[0] or v==ids[1] or v==ids[2]
                  or w==ids[0] or w==ids[1] or w==ids[2])
        eligible = groups[u] != groups[ids[0]]
        if self_check:
            eligible = groups[u] == groups[ids[0]] and not shared and j > i
        if eligible:
            ta = wp.matrix_from_rows(wp.vec3d(0.),b-a,c-a)
            tb = wp.matrix_from_rows(x[u]-a,x[v]-a,x[w]-a)
            if not triangle_separated(ta,tb):
                wp.atomic_max(status,0,31)


@wp.kernel
def curved_tet_paths(old: wp.array(dtype=wp.vec3d), new: wp.array(dtype=wp.vec3d),
    tets: wp.array2d(dtype=int), rest_inverse: wp.array(dtype=wp.mat33),
    mass: wp.array(dtype=float), owners: wp.array(dtype=int),
    arc_errors: wp.array(dtype=wp.float64), diagnostic: wp.array(dtype=wp.float64)):
    # Free vertices follow the step's affine path. Fixed vertices may follow
    # SLERP: bound the resulting determinant perturbation as well as collision.
    from_error = wp.vec4d(0.)
    t=wp.tid()
    ids=wp.vec4i(tets[t,0],tets[t,1],tets[t,2],tets[t,3])
    for j in range(4):
        if mass[ids[j]] == 0.:
            from_error[j]=arc_errors[owners[ids[j]]]
    errors=wp.vec3d(from_error[1]+from_error[0],from_error[2]+from_error[0],from_error[3]+from_error[0])
    edge_error=wp.length(errors)
    if edge_error > wp.float64(0.):
        dm=wp.mat33d(rest_inverse[t])
        f0=wp.matrix_from_cols(old[ids[1]]-old[ids[0]],old[ids[2]]-old[ids[0]],old[ids[3]]-old[ids[0]])*dm
        f1=wp.matrix_from_cols(new[ids[1]]-new[ids[0]],new[ids[2]]-new[ids[0]],new[ids[3]]-new[ids[0]])*dm
        n0,n1,nd=wp.float64(0.),wp.float64(0.),wp.float64(0.)
        for a in range(3):
            for b in range(3):
                n0+=f0[a,b]*f0[a,b];n1+=f1[a,b]*f1[a,b];nd+=dm[a,b]*dm[a,b]
        alpha=wp.sqrt(wp.max(n0,n1))
        eps=edge_error*wp.sqrt(nd)
        bound=wp.float64(3.)*alpha*alpha*eps+wp.float64(3.)*alpha*eps*eps+eps*eps*eps
        minimum=determinant_path_extremum(f0,f1-f0)[0]-bound
        wp.atomic_min(diagnostic,3,minimum)
