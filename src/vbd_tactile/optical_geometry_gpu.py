"""Device resident optical geometry; fixed marker bindings never change."""
from typing import Any
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def local_positions(world: wp.array(dtype=Any), offset: int, pose: wp.mat44d,
                    local: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    p = wp.vec3d(world[offset+i])-wp.vec3d(pose[0, 3], pose[1, 3], pose[2, 3])
    local[i] = wp.vec3d(p[0]*pose[0, 0]+p[1]*pose[1, 0]+p[2]*pose[2, 0],
                      p[0]*pose[0, 1]+p[1]*pose[1, 1]+p[2]*pose[2, 1],
                      p[0]*pose[0, 2]+p[1]*pose[1, 2]+p[2]*pose[2, 2])


@wp.kernel
def project_vertices(q: wp.array(dtype=wp.vec3d), rotation: wp.mat33d, translation: wp.vec3d,
                     intrinsics: wp.vec4d, near: wp.float64, surface: wp.array(dtype=int),
                     camera_points: wp.array(dtype=wp.vec3d), uv: wp.array(dtype=wp.vec2d),
                     z: wp.array(dtype=wp.float64), status: wp.array(dtype=int)):
    i = wp.tid()
    p = rotation*q[i]+translation
    valid = wp.isfinite(q[i][0]) and wp.isfinite(q[i][1]) and wp.isfinite(q[i][2])
    valid = valid and wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2])
    if not valid:
        wp.atomic_max(status, 0, 3)
        p = wp.vec3d(0., 0., 1.)
    if surface[i] != 0 and p[2] <= near:
        wp.atomic_max(status, 0, 1)
    camera_points[i] = p
    z[i] = p[2]
    pixel = wp.vec2d(0.)
    if p[2] > near:
        pixel = wp.vec2d(p[0]/p[2]*intrinsics[0]+intrinsics[2],
                         p[1]/p[2]*intrinsics[1]+intrinsics[3])
    # Never feed nonfinite/overflowed screen boxes to a BVH, even on a rejected frame.
    if not wp.isfinite(pixel[0]) or not wp.isfinite(pixel[1]) or wp.max(wp.abs(pixel[0]), wp.abs(pixel[1])) > wp.float64(1e30):
        wp.atomic_max(status, 0, 3)
        pixel = wp.vec2d(0.)
    uv[i] = pixel


@wp.kernel
def face_geometry(q: wp.array(dtype=wp.vec3d), cam: wp.array(dtype=wp.vec3d),
                  triangles: wp.array(dtype=wp.vec3i), sign: wp.array(dtype=wp.float64),
                  outward: bool, normals: wp.array(dtype=wp.vec3d), visible: wp.array(dtype=bool)):
    i = wp.tid()
    ids = triangles[i]
    orientation = sign[i]
    if outward:
        orientation = wp.float64(-1.)
    normals[i] = wp.cross(q[ids[1]]-q[ids[0]], q[ids[2]]-q[ids[0]])*orientation
    a, b, c = cam[ids[0]], cam[ids[1]], cam[ids[2]]
    visible[i] = not outward or wp.dot(wp.cross(b-a, c-a), (a+b+c)/wp.float64(3.)) > wp.float64(0.)


@wp.kernel
def vertex_lighting(faces: wp.array(dtype=wp.vec3d), offsets: wp.array(dtype=int),
                    adjacency: wp.array(dtype=int), rotation: wp.mat33d,
                    lights: wp.mat33d, flat: wp.vec3d, light: wp.array(dtype=wp.vec3d),
                    status: wp.array(dtype=int)):
    i = wp.tid()
    normal = wp.vec3d(0.)
    # Fixed adjacency order matches NumPy's corner-by-corner accumulation.
    # No floating atomic reductions or cross-frame nondeterminism.
    for j in range(offsets[i], offsets[i+1]):
        normal += faces[adjacency[j]]
    length = wp.length(normal)
    if offsets[i] != offsets[i+1] and length < wp.float64(1e-16):
        wp.atomic_max(status, 0, 2)
    if not wp.isfinite(length):
        wp.atomic_max(status, 0, 3)
        normal = wp.vec3d(0.)
        length = wp.float64(1.)
    normal = rotation*(normal/wp.max(length, wp.float64(1e-30)))
    value = wp.vec3d(0.)
    for k in range(3):
        value[k] = (wp.float64(.20)+wp.float64(.68)*wp.max(wp.dot(normal, lights[k]), wp.float64(0.)))/flat[k]*wp.float64(.34)
    light[i] = value


@wp.kernel
def marker_geometry(q: wp.array(dtype=wp.vec3d), ids: wp.array(dtype=wp.vec3i),
                    weights: wp.array(dtype=wp.vec3d), rotation: wp.mat33d, translation: wp.vec3d,
                    intrinsics: wp.vec4d, near: wp.float64, width: int, height: int,
                    reference: wp.array(dtype=wp.vec3d), reference_uv: wp.array(dtype=wp.vec2d),
                    reference_in_frame: wp.array(dtype=bool), samples: wp.array(dtype=wp.vec3d),
                    data: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    ind, w = ids[i], weights[i]
    p = w[0]*q[ind[0]]+w[1]*q[ind[1]]+w[2]*q[ind[2]]
    cam = rotation*p+translation
    uv = wp.vec2d(wp.nan)
    front = cam[2] > near
    if front:
        uv = wp.vec2d(cam[0]/cam[2]*intrinsics[0]+intrinsics[2], cam[1]/cam[2]*intrinsics[1]+intrinsics[3])
    valid = front and uv[0] >= wp.float64(0.) and uv[0] <= wp.float64(width-1) and uv[1] >= wp.float64(0.) and uv[1] <= wp.float64(height-1)
    # Invalid/offscreen markers must never submit NaNs to the BVH query.
    samples[i] = wp.vec3d(0., 0., wp.inf)
    if valid:
        samples[i] = wp.vec3d(uv[0], uv[1], cam[2])
    for j in range(3):
        data[i+1, j] = p[j]
        data[i+1, 3+j] = p[j]-reference[i][j]
        data[i+1, 6+j] = cam[j]
    for j in range(2):
        data[i+1, 9+j] = uv[j]
        data[i+1, 11+j] = uv[j]-reference_uv[i][j]
    data[i+1, 13] = wp.float64(valid)
    data[i+1, 14] = wp.float64(valid and reference_in_frame[i])


@wp.kernel
def finish_markers(visible: wp.array(dtype=bool), reference_visible: wp.array(dtype=bool),
                   status: wp.array(dtype=int), data: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    if i == 0:
        data[0, 0] = wp.float64(status[0])
    if i < visible.shape[0]:
        valid = visible[i] and data[i+1, 13] != wp.float64(0.)
        data[i+1, 15] = wp.float64(valid)
        data[i+1, 16] = wp.float64(valid and reference_visible[i] and data[i+1, 14] != wp.float64(0.))
