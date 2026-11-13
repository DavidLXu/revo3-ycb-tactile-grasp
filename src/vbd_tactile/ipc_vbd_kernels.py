"""Colored implicit Euler blocks for the version-pinned tetrahedral IPC adapter.

The elastic/damping potentials match Newton 1.5.2's stable Neo-Hookean tet
model. Each color contains at most one vertex of any tet. Thus local energy
line searches and affine per-vertex determinant checks compose safely.
"""

from typing import Any
from .ipc_precision import position_overloads, pose_value
import warp as wp
from newton._src.utils.mesh import (
    MeshAdjacencyData,
    get_vertex_adjacent_tet_id_order,
    get_vertex_num_adjacent_tets,
)

from .ipc_warp import _swept_gap

wp.set_module_options({"enable_backward": False})


@wp.kernel
def validate_passive_control(
    values: wp.array(dtype=float), invalid: wp.array(dtype=int)
):
    if values[wp.tid()] != 0.0:
        wp.atomic_add(invalid, 0, 1)


@wp.func
def _cofactor(f: wp.mat33d):
    a = wp.vec3d(f[0, 0], f[1, 0], f[2, 0])
    b = wp.vec3d(f[0, 1], f[1, 1], f[2, 1])
    c = wp.vec3d(f[0, 2], f[1, 2], f[2, 2])
    return wp.matrix_from_cols(wp.cross(b, c), wp.cross(c, a), wp.cross(a, b))


@wp.func
def _frobenius2(f: wp.mat33d):
    total = wp.float64(0.0)
    for i in range(3):
        for j in range(3):
            total += f[i, j] * f[i, j]
    return total


@wp.func
def _tet_terms(
    f: wp.mat33d,
    previous_f: wp.mat33d,
    shape_gradient: wp.vec3d,
    volume: wp.float64,
    mu: wp.float64,
    lame: wp.float64,
    damping: wp.float64,
    dt: wp.float64,
):
    lam = lame + mu
    alpha = wp.float64(1.0) + mu / lam
    j = wp.determinant(f)
    cof = _cofactor(f)
    g = cof * shape_gradient
    identity = wp.identity(n=3, dtype=wp.float64)
    energy = volume * (
        wp.float64(0.5) * mu * (_frobenius2(f) - wp.float64(3.0))
        + wp.float64(0.5)
        * lam
        * (
            (j - alpha) * (j - alpha)
            - (wp.float64(1.0) - alpha) * (wp.float64(1.0) - alpha)
        )
    )
    force = -volume * (mu * (f * shape_gradient) + lam * (j - alpha) * g)
    hessian = volume * (
        mu * wp.dot(shape_gradient, shape_gradient) * identity + lam * wp.outer(g, g)
    )
    # Newton's damping force is the gradient of V*d/(2*dt)||C-C_old||_F^2.
    c = wp.transpose(f) * f - wp.transpose(previous_f) * previous_f
    energy += volume * damping / (wp.float64(2.0) * dt) * _frobenius2(c)
    if damping > wp.float64(0.0):
        force -= wp.float64(2.0) * volume * damping / dt * (f * c * shape_gradient)
        for row in range(3):
            for col in range(3):
                fr = wp.vec3d(f[0, row], f[1, row], f[2, row])
                fc = wp.vec3d(f[0, col], f[1, col], f[2, col])
                dc = shape_gradient[row] * fc + shape_gradient[col] * fr
                hessian += volume * damping / dt * wp.outer(dc, dc)
    return energy, force, hessian, j


@wp.func
def _contact_terms(
    x: wp.vec3d,
    old: wp.vec3d,
    center: wp.vec3d,
    old_center: wp.vec3d,
    radius: wp.float64,
    dhat: wp.float64,
    stiffness: wp.float64,
    friction_normal: wp.vec3d,
    friction_load: wp.float64,
    friction_mu: wp.float64,
    friction_eps: wp.float64,
):
    delta = x - center
    length = wp.length(delta)
    gap = length - radius
    energy = wp.float64(0.0)
    force = wp.vec3d(0.0)
    hessian = wp.mat33d(0.0)
    if gap <= wp.float64(0.0):
        energy = wp.float64(wp.inf)
    else:
        if gap < dhat:
            n = delta / length
            a = gap - dhat
            log = wp.log(gap / dhat)
            first = -wp.float64(2.0) * a * log - a * a / gap
            second = (
                -wp.float64(2.0) * log - wp.float64(4.0) * a / gap + a * a / (gap * gap)
            )
            energy = -stiffness * a * a * log
            force = -stiffness * first * n
            # PSD projection removes only the negative tangential eigenvalues.
            hessian = stiffness * second * wp.outer(n, n)
        if friction_mu > wp.float64(0.0) and friction_load > wp.float64(0.0):
            n = friction_normal
            displacement = x - old - (center - old_center)
            tangent = displacement - n * wp.dot(n, displacement)
            r = wp.length(tangent)
            factor, second = wp.float64(0.0), wp.float64(0.0)
            potential = r - friction_eps / wp.float64(3.0)
            if r < friction_eps:
                potential = r * r / friction_eps - r * r * r / (
                    wp.float64(3.0) * friction_eps * friction_eps
                )
                factor = wp.float64(2.0) / friction_eps - r / (
                    friction_eps * friction_eps
                )
                second = wp.float64(2.0) / friction_eps - wp.float64(2.0) * r / (
                    friction_eps * friction_eps
                )
            else:
                factor = wp.float64(1.0) / r
            scale = friction_mu * friction_load
            energy += scale * potential
            force -= scale * factor * tangent
            unit = wp.vec3d(0.0)
            if r > wp.float64(0.0):
                unit = tangent / r
            hessian += scale * (
                factor * (wp.identity(n=3, dtype=wp.float64) - wp.outer(n, n))
                + (second - factor) * wp.outer(unit, unit)
            )
    return energy, force, hessian, gap


@wp.func
def _vertex_terms(
    index: int,
    candidate: wp.vec3d,
    positions: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    inertia: wp.array(dtype=wp.vec3d),
    masses: wp.array(dtype=float),
    radii: wp.array(dtype=wp.float64),
    indices: wp.array2d(dtype=int),
    poses: wp.array(dtype=wp.mat33),
    materials: wp.array2d(dtype=float),
    adjacency: MeshAdjacencyData,
    dt: wp.float64,
    center: wp.vec3d,
    old_center: wp.vec3d,
    sphere_radius: wp.float64,
    dhat: wp.float64,
    stiffness: wp.float64,
    normals: wp.array(dtype=wp.vec3d),
    loads: wp.array(dtype=wp.float64),
    friction_mu: wp.float64,
    friction_eps: wp.float64,
):
    energy, force, hessian, gap = _contact_terms(
        candidate,
        wp.vec3d(old[index]),
        center,
        old_center,
        sphere_radius + radii[index],
        dhat,
        stiffness,
        normals[index],
        loads[index],
        friction_mu,
        friction_eps,
    )
    inertial_weight = wp.float64(masses[index]) / (dt * dt)
    offset = candidate - inertia[index]
    energy += wp.float64(0.5) * inertial_weight * wp.dot(offset, offset)
    force -= inertial_weight * offset
    hessian += inertial_weight * wp.identity(n=3, dtype=wp.float64)
    min_j = wp.float64(wp.inf)
    for adjacent in range(get_vertex_num_adjacent_tets(adjacency, index)):
        tet, order = get_vertex_adjacent_tet_id_order(adjacency, index, adjacent)
        ia, ib, ic, id = (
            indices[tet, 0],
            indices[tet, 1],
            indices[tet, 2],
            indices[tet, 3],
        )
        a, b, c, d = (
            wp.vec3d(positions[ia]),
            wp.vec3d(positions[ib]),
            wp.vec3d(positions[ic]),
            wp.vec3d(positions[id]),
        )
        if order == 0:
            a = candidate
        elif order == 1:
            b = candidate
        elif order == 2:
            c = candidate
        else:
            d = candidate
        dm = wp.mat33d(poses[tet])
        f = wp.matrix_from_cols(b - a, c - a, d - a) * dm
        previous_a = wp.vec3d(old[ia])
        f0 = (
            wp.matrix_from_cols(
                wp.vec3d(old[ib]) - previous_a,
                wp.vec3d(old[ic]) - previous_a,
                wp.vec3d(old[id]) - previous_a,
            )
            * dm
        )
        m = wp.vec3d(0.0)
        for j in range(3):
            if order == 0:
                m[j] = -(dm[0, j] + dm[1, j] + dm[2, j])
            else:
                m[j] = dm[order - 1, j]
        e, f_tet, h_tet, jacobian = _tet_terms(
            f,
            f0,
            m,
            wp.float64(1.0) / (wp.float64(6.0) * wp.determinant(dm)),
            wp.float64(materials[tet, 0]),
            wp.float64(materials[tet, 1]),
            wp.float64(materials[tet, 2]),
            dt,
        )
        energy += e
        force += f_tet
        hessian += h_tet
        min_j = wp.min(min_j, jacobian)
    return energy, force, hessian, gap, min_j


@position_overloads("old", "velocity", "positions", poses=('c0', 'c1'))
@wp.kernel
def initialize(
    old: wp.array(dtype=Any),
    velocity: wp.array(dtype=Any),
    external_force: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    gravity: wp.array(dtype=wp.vec3),
    worlds: wp.array(dtype=int),
    radii: wp.array(dtype=wp.float64),
    c0: Any,
    c1: Any,
    radius: wp.float64,
    dhat: wp.float64,
    stiffness: wp.float64,
    dt: wp.float64,
    min_gap: wp.float64,
    positions: wp.array(dtype=Any),
    inertia: wp.array(dtype=wp.vec3d),
    normals: wp.array(dtype=wp.vec3d),
    loads: wp.array(dtype=wp.float64),
):
    _c0 = pose_value(c0)
    _c1 = pose_value(c1)
    i = wp.tid()
    x = wp.vec3d(old[i])
    normal = (x - _c0) / wp.length(x - _c0)
    normals[i] = normal
    _energy, reaction, _hessian, gap = _contact_terms(
        x,
        x,
        _c0,
        _c0,
        radius + radii[i],
        dhat,
        stiffness,
        normal,
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(1.0),
    )
    loads[i] = wp.length(reaction)
    inertia[i] = x
    positions[i] = old[i]
    if mass[i] > 0.0:
        inertia[i] = (
            x
            + dt * wp.vec3d(velocity[i])
            + dt
            * dt
            * (
                wp.vec3d(gravity[worlds[i]])
                + wp.vec3d(external_force[i]) / wp.float64(mass[i])
            )
        )
        # Translate the near-contact region with the kinematic sphere to obtain
        # a feasible seed at its new center; fade to zero away from contact.
        inner = min_gap + wp.float64(2.0) * wp.length(_c1 - _c0)
        weight = wp.clamp(
            (inner + wp.float64(2.0) * dhat - gap) / (wp.float64(2.0) * dhat),
            wp.float64(0.0),
            wp.float64(1.0),
        )
        positions[i] = type(old[i])(x + weight * (_c1 - _c0))


@wp.kernel
def solve_color(
    ids: wp.array(dtype=int),
    positions: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    inertia: wp.array(dtype=wp.vec3d),
    masses: wp.array(dtype=float),
    radii: wp.array(dtype=wp.float64),
    indices: wp.array2d(dtype=int),
    poses: wp.array(dtype=wp.mat33),
    materials: wp.array2d(dtype=float),
    adjacency: MeshAdjacencyData,
    dt: wp.float64,
    center: wp.vec3d,
    old_center: wp.vec3d,
    sphere_radius: wp.float64,
    dhat: wp.float64,
    stiffness: wp.float64,
    normals: wp.array(dtype=wp.vec3d),
    loads: wp.array(dtype=wp.float64),
    friction_mu: wp.float64,
    friction_eps: wp.float64,
    min_gap: wp.float64,
    min_jacobian: wp.float64,
    line_iterations: int,
    diagnostic: wp.array(dtype=wp.float64),
):
    index = ids[wp.tid()]
    if masses[index] > 0.0:
        x = wp.vec3d(positions[index])
        energy, force, hessian, gap, min_j = _vertex_terms(
            index,
            x,
            positions,
            old,
            inertia,
            masses,
            radii,
            indices,
            poses,
            materials,
            adjacency,
            dt,
            center,
            old_center,
            sphere_radius,
            dhat,
            stiffness,
            normals,
            loads,
            friction_mu,
            friction_eps,
        )
        if not wp.isfinite(energy) or gap <= min_gap or min_j <= min_jacobian:
            wp.atomic_add(diagnostic, 0, wp.float64(1.0))
        else:
            direction = wp.inverse(hessian) * force
            if not wp.isfinite(wp.length(direction)) or wp.dot(
                direction, force
            ) < wp.float64(0.0):
                wp.atomic_add(diagnostic, 0, wp.float64(1.0))
            else:
                alpha = wp.float64(1.0)
                for attempt in range(line_iterations):
                    rounded = wp.vec3(x + alpha * direction)
                    candidate = wp.vec3d(rounded)
                    # Each tet in this color has only one moving vertex:
                    # det F is affine, so endpoint positivity covers its path.
                    trial, _trial_force, _trial_hessian, new_gap, new_j = _vertex_terms(
                        index,
                        candidate,
                        positions,
                        old,
                        inertia,
                        masses,
                        radii,
                        indices,
                        poses,
                        materials,
                        adjacency,
                        dt,
                        center,
                        old_center,
                        sphere_radius,
                        dhat,
                        stiffness,
                        normals,
                        loads,
                        friction_mu,
                        friction_eps,
                    )
                    swept = _swept_gap(
                        x, candidate, center, center, sphere_radius + radii[index]
                    )
                    decrease = wp.dot(force, candidate - x)
                    if (
                        wp.isfinite(trial)
                        and new_gap > min_gap
                        and swept > min_gap
                        and new_j > min_jacobian
                        and trial
                        <= energy
                        - wp.float64(1.0e-4) * wp.max(decrease, wp.float64(0.0))
                    ):
                        positions[index] = rounded
                        wp.atomic_max(diagnostic, 1, trial - energy)
                        wp.atomic_min(diagnostic, 2, swept)
                        wp.atomic_min(diagnostic, 3, new_j)
                        break
                    alpha *= wp.float64(0.5)


@wp.kernel
def residual(
    positions: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    inertia: wp.array(dtype=wp.vec3d),
    masses: wp.array(dtype=float),
    radii: wp.array(dtype=wp.float64),
    indices: wp.array2d(dtype=int),
    poses: wp.array(dtype=wp.mat33),
    materials: wp.array2d(dtype=float),
    adjacency: MeshAdjacencyData,
    dt: wp.float64,
    center: wp.vec3d,
    old_center: wp.vec3d,
    sphere_radius: wp.float64,
    dhat: wp.float64,
    stiffness: wp.float64,
    normals: wp.array(dtype=wp.vec3d),
    loads: wp.array(dtype=wp.float64),
    friction_mu: wp.float64,
    friction_eps: wp.float64,
    diagnostic: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    if masses[i] > 0.0:
        e, f, _hessian, _gap, _jacobian = _vertex_terms(
            i,
            wp.vec3d(positions[i]),
            positions,
            old,
            inertia,
            masses,
            radii,
            indices,
            poses,
            materials,
            adjacency,
            dt,
            center,
            old_center,
            sphere_radius,
            dhat,
            stiffness,
            normals,
            loads,
            friction_mu,
            friction_eps,
        )
        if not wp.isfinite(e) or not wp.isfinite(wp.length(f)):
            wp.atomic_add(diagnostic, 0, wp.float64(1.0))
        else:
            wp.atomic_max(diagnostic, 4, wp.length(f))


@wp.func
def _poly(
    t: wp.float64, c0: wp.float64, c1: wp.float64, c2: wp.float64, c3: wp.float64
):
    return ((c3 * t + c2) * t + c1) * t + c0


@wp.func
def determinant_path_extremum(a: wp.mat33d, b: wp.mat33d):
    # Exact cubic determinant extrema of F(t)=a+t*b on [0,1].
    a0, a1, a2 = (
        wp.vec3d(a[0, 0], a[1, 0], a[2, 0]),
        wp.vec3d(a[0, 1], a[1, 1], a[2, 1]),
        wp.vec3d(a[0, 2], a[1, 2], a[2, 2]),
    )
    b0, b1, b2 = (
        wp.vec3d(b[0, 0], b[1, 0], b[2, 0]),
        wp.vec3d(b[0, 1], b[1, 1], b[2, 1]),
        wp.vec3d(b[0, 2], b[1, 2], b[2, 2]),
    )
    c0 = wp.dot(a0, wp.cross(a1, a2))
    c1 = (
        wp.dot(b0, wp.cross(a1, a2))
        + wp.dot(a0, wp.cross(b1, a2))
        + wp.dot(a0, wp.cross(a1, b2))
    )
    c2 = (
        wp.dot(b0, wp.cross(b1, a2))
        + wp.dot(b0, wp.cross(a1, b2))
        + wp.dot(a0, wp.cross(b1, b2))
    )
    c3 = wp.dot(b0, wp.cross(b1, b2))
    result = wp.min(c0, c0 + c1 + c2 + c3)
    time = wp.float64(0.0)
    if c0 + c1 + c2 + c3 < c0:
        time = wp.float64(1.0)
    qa, qb, qc = wp.float64(3.0) * c3, wp.float64(2.0) * c2, c1
    if qa == wp.float64(0.0):
        if qb != wp.float64(0.0):
            root = -qc / qb
            if root > wp.float64(0.0) and root < wp.float64(1.0):
                value = _poly(root, c0, c1, c2, c3)
                if value < result:
                    result, time = value, root
    else:
        discriminant = qb * qb - wp.float64(4.0) * qa * qc
        if discriminant >= wp.float64(0.0):
            sign = wp.float64(1.0)
            if qb < wp.float64(0.0):
                sign = wp.float64(-1.0)
            q = -wp.float64(0.5) * (qb + sign * wp.sqrt(discriminant))
            r0 = -qb / (wp.float64(2.0) * qa)
            r1 = r0
            if q != wp.float64(0.0):
                r0 = q / qa
                r1 = qc / q
            if r0 > wp.float64(0.0) and r0 < wp.float64(1.0):
                value = _poly(r0, c0, c1, c2, c3)
                if value < result:
                    result, time = value, r0
            if r1 > wp.float64(0.0) and r1 < wp.float64(1.0):
                value = _poly(r1, c0, c1, c2, c3)
                if value < result:
                    result, time = value, r1
    return wp.vec2d(result, time)


@wp.func
def determinant_path_min(a: wp.mat33d, b: wp.mat33d):
    return determinant_path_extremum(a, b)[0]


@position_overloads("old", "new")
@wp.kernel
def validate_tet_paths(
    old: wp.array(dtype=Any),
    new: wp.array(dtype=Any),
    indices: wp.array2d(dtype=int),
    poses: wp.array(dtype=wp.mat33),
    diagnostic: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    a, b, c, d = indices[i, 0], indices[i, 1], indices[i, 2], indices[i, 3]
    old_a, new_a = wp.vec3d(old[a]), wp.vec3d(new[a])
    dm = wp.mat33d(poses[i])
    f0 = (
        wp.matrix_from_cols(
            wp.vec3d(old[b]) - old_a, wp.vec3d(old[c]) - old_a, wp.vec3d(old[d]) - old_a
        )
        * dm
    )
    f1 = (
        wp.matrix_from_cols(
            wp.vec3d(new[b]) - new_a, wp.vec3d(new[c]) - new_a, wp.vec3d(new[d]) - new_a
        )
        * dm
    )
    minimum = determinant_path_min(f0, f1 - f0)
    if not wp.isfinite(minimum):
        wp.atomic_add(diagnostic, 0, wp.float64(1.0))
    else:
        wp.atomic_min(diagnostic, 3, minimum)


@position_overloads("old", "new", poses=('c0', 'c1'))
@wp.kernel
def validate_sphere_paths(
    old: wp.array(dtype=Any),
    new: wp.array(dtype=Any),
    radii: wp.array(dtype=wp.float64),
    radius: wp.float64,
    c0: Any,
    c1: Any,
    diagnostic: wp.array(dtype=wp.float64),
):
    _c0 = pose_value(c0)
    _c1 = pose_value(c1)
    i = wp.tid()
    gap = _swept_gap(wp.vec3d(old[i]), wp.vec3d(new[i]), _c0, _c1, radius + radii[i])
    if not wp.isfinite(gap):
        wp.atomic_add(diagnostic, 0, wp.float64(1.0))
    else:
        wp.atomic_min(diagnostic, 2, gap)


@wp.kernel
def validate_finite_data(values: wp.array(dtype=Any), invalid: wp.array(dtype=int)):
    if not wp.isfinite(values[wp.tid()]):
        wp.atomic_add(invalid, 0, 1)


@position_overloads("old", "positions", "output", "velocity")
@wp.kernel
def finalize(
    old: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    dt: wp.float64,
    output: wp.array(dtype=Any),
    velocity: wp.array(dtype=Any),
):
    i = wp.tid()
    output[i] = positions[i]
    velocity[i] = type(old[i])((wp.vec3d(positions[i]) - wp.vec3d(old[i])) / dt)


@wp.kernel
def contact_response(
    positions: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float64),
    center: wp.vec3d,
    old_center: wp.vec3d,
    sphere_radius: wp.float64,
    dhat: wp.float64,
    stiffness: wp.float64,
    normals: wp.array(dtype=wp.vec3d),
    loads: wp.array(dtype=wp.float64),
    friction_mu: wp.float64,
    friction_eps: wp.float64,
    forces: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    _energy, force, _hessian, _gap = _contact_terms(
        wp.vec3d(positions[i]),
        wp.vec3d(old[i]),
        center,
        old_center,
        sphere_radius + radii[i],
        dhat,
        stiffness,
        normals[i],
        loads[i],
        friction_mu,
        friction_eps,
    )
    forces[i] = wp.vec3(force)


@wp.kernel
def set_sphere_state(
    center: wp.vec3,
    linear_velocity: wp.vec3,
    pose: wp.array(dtype=wp.transform),
    velocity: wp.array(dtype=wp.spatial_vector),
):
    pose[0] = wp.transform(center, wp.transform_get_rotation(pose[0]))
    velocity[0] = wp.spatial_vector(linear_velocity, wp.vec3(0.0))


validate_finite_data_double = wp.overload(
    validate_finite_data, {"values": wp.array(dtype=wp.float64)}
)
validate_finite_data = wp.overload(
    validate_finite_data, {"values": wp.array(dtype=wp.float32)}
)
