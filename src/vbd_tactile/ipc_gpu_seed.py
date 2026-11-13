"""Bounded device-only feasibility repair of the initial guess.

These projections never change the inertial predictor or fixed nodes. They are
not a contact solver: the full swept sphere, self-contact and tet guards must
pass afterwards, and Newton still solves the original objective to tolerance.
"""
from typing import Any
import warp as wp
from .ipc_precision import position_overloads, pose_value
from .ipc_primitives_warp import _weights, _delta
from .ipc_self_audit import _projected_gap
from .ipc_vbd_kernels import determinant_path_extremum
from .ipc_volume_barrier import jacobian_gradients
from .ipc_gpu_math import vec12

wp.set_module_options({"enable_backward": False})


@wp.kernel
def condition(d: wp.array(dtype=wp.float64), motion: wp.array(dtype=int),
              gap: wp.float64, jac: wp.float64, maximum: int,
              repair_volume: bool,
              status: wp.array(dtype=int), out: wp.array(dtype=int)):
    bad = d[0] != wp.float64(0) or d[2] <= gap or d[5] <= gap or d[3] <= jac or motion[1] != 0
    # Finite volume defects can be projected with the contact defects. The
    # complete swept audit still has to pass before evaluating the objective.
    progress = status[12] == 0 or status[13] != 0
    volume_allowed = d[3] > jac or repair_volume
    out[0] = int(status[0] == 0 and bad and volume_allowed and wp.isfinite(d[3]) and motion[2] == 0
                 and status[12] < maximum and progress)


@wp.kernel
def reset_progress(status: wp.array(dtype=int)):
    status[13] = 0


@position_overloads('old', 'x')
@wp.kernel
def volume(old: Any, x: Any, indices: wp.array2d(dtype=int),
           poses: wp.array(dtype=wp.mat33), mass: wp.array(dtype=float),
           floor: wp.float64, delta: wp.array(dtype=wp.vec3d),
           count: wp.array(dtype=wp.float64)):
    t = wp.tid()
    ids = wp.vec4i(indices[t, 0], indices[t, 1], indices[t, 2], indices[t, 3])
    dm = wp.mat33d(poses[t])
    a, b = wp.vec3d(old[ids[0]]), wp.vec3d(x[ids[0]])
    f0 = wp.matrix_from_cols(wp.vec3d(old[ids[1]])-a,
        wp.vec3d(old[ids[2]])-a, wp.vec3d(old[ids[3]])-a)*dm
    f1 = wp.matrix_from_cols(wp.vec3d(x[ids[1]])-b,
        wp.vec3d(x[ids[2]])-b, wp.vec3d(x[ids[3]])-b)*dm
    minimum = determinant_path_extremum(f0, f1-f0)
    target = floor*wp.float64(1.05)
    # Only repair a violated constraint. Pushing already-feasible tets toward
    # the slack target can fight a sphere correction unnecessarily.
    if wp.isfinite(minimum[0]) and minimum[0] <= floor and minimum[1] > wp.float64(1e-12):
        # Differentiate the limiting intermediate F, not only its endpoint.
        g0, g1, g2, g3 = jacobian_gradients(f0+minimum[1]*(f1-f0), dm)
        g = minimum[1]*vec12(g0[0], g0[1], g0[2], g1[0], g1[1], g1[2],
                             g2[0], g2[1], g2[2], g3[0], g3[1], g3[2])
        denominator = wp.float64(0)
        for j in range(4):
            if mass[ids[j]] > 0.0:
                for axis in range(3):
                    denominator += g[3*j+axis]*g[3*j+axis]
        if denominator > wp.float64(1e-20) and wp.isfinite(denominator):
            multiplier = (target-minimum[0])/denominator
            for j in range(4):
                if mass[ids[j]] > 0.0:
                    correction = multiplier*wp.vec3d(g[3*j], g[3*j+1], g[3*j+2])
                    wp.atomic_add(delta, ids[j], correction)
                    wp.atomic_add(count, ids[j], wp.float64(1))


@position_overloads('x', poses=('center',))
@wp.kernel
def fixed_sphere_target(x: Any, faces: wp.array2d(dtype=int), mass: wp.array(dtype=float),
                        center: Any, radius: wp.float64, gap: wp.float64,
                        status: wp.array(dtype=int)):
    f = wp.tid()
    if status[0] == 0:
        c = pose_value(center)
        if (mass[faces[f, 0]] == 0.0 and mass[faces[f, 1]] == 0.0
                and mass[faces[f, 2]] == 0.0):
            a, b, d = wp.vec3d(x[faces[f, 0]]), wp.vec3d(x[faces[f, 1]]), wp.vec3d(x[faces[f, 2]])
            distance = wp.length(_delta(_weights(c, a, b, d, 0), c, a, b, d))
            if wp.isfinite(distance) and distance <= radius+gap:
                wp.atomic_max(status, 0, 11)
        else:
            # A mixed face can deform, but its fixed vertex / two-fixed-end
            # edge cannot. Check only those features, never the whole face.
            # No time subdivision can make such a prescribed target feasible.
            for j in range(3):
                a_id = faces[f, j]
                b_id = faces[f, (j+1)%3]
                if mass[a_id] == 0.0:
                    nearest = wp.vec3d(x[a_id])
                    if mass[b_id] == 0.0:
                        edge = wp.vec3d(x[b_id])-nearest
                        denominator = wp.dot(edge, edge)
                        if denominator > wp.float64(0):
                            t = wp.clamp(wp.dot(c-nearest, edge)/denominator, wp.float64(0), wp.float64(1))
                            nearest += t*edge
                    distance = wp.length(c-nearest)
                    if wp.isfinite(distance) and distance <= radius+gap:
                        wp.atomic_max(status, 0, 11)


@position_overloads('x', poses=('center',))
@wp.kernel
def sphere(x: Any, faces: wp.array2d(dtype=int), mass: wp.array(dtype=float),
           center: Any, radius: wp.float64, gap: wp.float64,
           delta: wp.array(dtype=wp.vec3d), count: wp.array(dtype=wp.float64), status: wp.array(dtype=int)):
    f = wp.tid()
    c = pose_value(center)
    a, b, d = wp.vec3d(x[faces[f, 0]]), wp.vec3d(x[faces[f, 1]]), wp.vec3d(x[faces[f, 2]])
    w = _weights(c, a, b, d, 0)
    v = _delta(w, c, a, b, d)
    distance = wp.length(v)
    deficit = radius + gap * wp.float64(1.01) - distance
    if (mass[faces[f, 0]] == 0.0 and mass[faces[f, 1]] == 0.0 and mass[faces[f, 2]] == 0.0
            and distance <= radius+gap):
        wp.atomic_max(status, 0, 11)
    if deficit > wp.float64(0) and distance > wp.float64(0) and wp.isfinite(deficit):
        den = wp.float64(0)
        for j in range(3):
            if mass[faces[f, j]] > 0.0:
                den += w[j+1]*w[j+1]
        if den > wp.float64(1e-20):
            n = v / distance
            for j in range(3):
                node = faces[f, j]
                if mass[node] > 0.0 and w[j+1] != wp.float64(0):
                    wp.atomic_add(delta, node, n*(deficit*w[j+1]/den))
                    wp.atomic_add(count, node, wp.float64(1))


@position_overloads('old', 'x')
@wp.kernel
def self_witness(old: Any, x: Any, faces: wp.array2d(dtype=int), edges: wp.array2d(dtype=int),
                 mass: wp.array(dtype=float), witness: wp.array2d(dtype=int), stats: wp.array(dtype=int),
                 gap: wp.float64, delta: wp.array(dtype=wp.vec3d), count: wp.array(dtype=wp.float64)):
    i = wp.tid()
    if i < wp.min(stats[1], witness.shape[0]):
        kind, a, b = witness[i, 0], witness[i, 1], witness[i, 2]
        ids = wp.vec4i(0)
        if kind == 0:
            ids = wp.vec4i(a, faces[b, 0], faces[b, 1], faces[b, 2])
        else:
            ids = wp.vec4i(edges[a, 0], edges[a, 1], edges[b, 0], edges[b, 1])
        p, q, r, s = wp.vec3d(old[ids[0]]), wp.vec3d(old[ids[1]]), wp.vec3d(old[ids[2]]), wp.vec3d(old[ids[3]])
        v = _delta(_weights(p, q, r, s, kind), p, q, r, s)
        distance = wp.length(v)
        if distance > wp.float64(0) and wp.isfinite(distance):
            # Keep the original collision side. The end closest normal alone
            # could push an already crossed point farther through the surface.
            n = v / distance
            # A single old normal may become impossible when two fixed
            # vertices move tangentially past each other. Prefer the current
            # closest normal if it also separates the OLD features. This lets
            # the plane rotate while preventing a repair to the crossed side.
            pa, pb, pc, pd = wp.vec3d(x[ids[0]]), wp.vec3d(x[ids[1]]), wp.vec3d(x[ids[2]]), wp.vec3d(x[ids[3]])
            weights = _weights(pa, pb, pc, pd, kind)
            current = _delta(weights, pa, pb, pc, pd)
            length = wp.length(current)
            if length > wp.float64(0) and wp.isfinite(length):
                alternative = current/length
                if _projected_gap(alternative, p, q, r, s, kind) > gap:
                    n = alternative
            # 25 nm of seed slack for the default 0.1 um self gap avoids
            # generating guesses arbitrarily close to the CCD threshold.
            # This does not change the barrier, force solve or acceptance gap.
            deficit = gap*wp.float64(1.25)-wp.dot(n, current)
            den = wp.float64(0)
            for j in range(4):
                if mass[ids[j]] > 0.0:
                    den += weights[j]*weights[j]
            # Correct the active closest feature, not every vertex pair on an
            # old fixed plane. The latter can be impossible even for a feasible
            # motion, since a faraway fixed corner cannot change its projection.
            # This is only a candidate; the full swept audit remains mandatory.
            if deficit > wp.float64(0) and den > wp.float64(1e-20) and wp.isfinite(deficit):
                for j in range(4):
                    if mass[ids[j]] > 0.0 and weights[j] != wp.float64(0):
                        wp.atomic_add(delta, ids[j], n*(deficit*weights[j]/den))
                        wp.atomic_add(count, ids[j], wp.float64(1))


@position_overloads('x')
@wp.kernel
def apply(x: Any, mass: wp.array(dtype=float), delta: wp.array(dtype=wp.vec3d),
          count: wp.array(dtype=wp.float64), status: wp.array(dtype=int)):
    i = wp.tid()
    if i == 0:
        status[12] += 1
    if mass[i] > 0.0 and count[i] > wp.float64(0):
        value = type(x[i])(wp.vec3d(x[i])+delta[i]/count[i])
        if wp.length(wp.vec3d(value)-wp.vec3d(x[i])) > wp.float64(0):
            wp.atomic_max(status, 13, 1)
        x[i] = value
