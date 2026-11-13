"""Boundary extraction and independent NumPy sphere/surface diagnostics."""

import numpy as np


def boundary_triangles(tets):
    """Extract the boundary of consistently oriented, nonduplicate tetrahedra."""
    tets = np.asarray(tets)
    if (
        tets.ndim != 2
        or tets.shape[1] != 4
        or not np.issubdtype(tets.dtype, np.integer)
    ):
        raise ValueError("tet indices must be an integer (N,4) array")
    if np.any(tets < 0) or np.any(np.diff(np.sort(tets, axis=1), axis=1) == 0):
        raise ValueError("tet indices must be nonnegative and distinct")
    if np.any(tets > np.iinfo(np.int32).max):
        raise ValueError("tet indices must fit in int32")
    if len(np.unique(np.sort(tets, axis=1), axis=0)) != len(tets):
        raise ValueError("duplicate tetrahedra would hide collision boundaries")
    # Outward winding for a positively oriented tet. An interior face must
    # occur twice with opposite winding; equal winding hides overlapping cells.
    faces = tets[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    unique, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError("nonmanifold tet faces are unsupported")
    inversions = np.sum(faces[:, [0, 0, 1]] > faces[:, [1, 2, 2]], axis=1)
    orientation = np.bincount(inverse, weights=1 - 2 * (inversions % 2))
    if np.any((counts == 2) & (orientation != 0)):
        raise ValueError("shared tet faces must have opposite orientations")
    return unique[counts == 1].astype(np.int32)


def sphere_surface_gaps(positions, triangles, center, radius):
    """Exact-feature endpoint distances; no particle-radius padding is applied.

    This independent CPU diagnostic does not test containment or continuous
    motion. Positive results alone do not prove disjoint solid interiors.
    """
    q = np.asarray(positions, dtype=np.float64)
    faces = np.asarray(triangles)
    center = np.asarray(center, dtype=np.float64)
    if (
        q.ndim != 2
        or q.shape[1] != 3
        or center.shape != (3,)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(center))
    ):
        raise ValueError("positions and center must be finite 3D points")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not np.issubdtype(faces.dtype, np.integer)
        or np.any(faces < 0)
        or np.any(faces >= len(q))
    ):
        raise ValueError("triangles must contain valid integer particle indices")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("sphere radius must be positive and finite")
    a, b, c = (q[faces[:, i]] - center for i in range(3))
    nearest = np.full(len(faces), np.inf)
    for u, v in ((a, b), (b, c), (c, a)):
        edge = v - u
        den = np.einsum("ij,ij->i", edge, edge)
        t = np.clip(
            np.divide(
                -np.einsum("ij,ij->i", u, edge),
                den,
                out=np.zeros_like(den),
                where=den > 0,
            ),
            0,
            1,
        )
        delta = u + t[:, None] * edge
        nearest = np.minimum(nearest, np.einsum("ij,ij->i", delta, delta))
    e0, e1 = b - a, c - a
    normal = np.cross(e0, e1)
    den = np.einsum("ij,ij->i", normal, normal)
    u = np.divide(
        np.einsum("ij,ij->i", np.cross(-a, e1), normal),
        den,
        out=np.full_like(den, -1),
        where=den > 0,
    )
    v = np.divide(
        np.einsum("ij,ij->i", np.cross(e0, -a), normal),
        den,
        out=np.full_like(den, -1),
        where=den > 0,
    )
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    plane = np.divide(
        np.einsum("ij,ij->i", a, normal) ** 2,
        den,
        out=np.full_like(den, np.inf),
        where=den > 0,
    )
    return np.sqrt(np.minimum(nearest, np.where(inside, plane, np.inf))) - radius


def nonincident_surface_intersections(positions, triangles, *, tolerance=1e-10):
    """Conservative static SAT audit of triangle pairs without shared vertices.

    Shared-vertex pairs and motion between snapshots are intentionally outside
    this diagnostic's scope. An empty result is not a global injectivity proof.
    """
    q = np.asarray(positions, dtype=np.float64)
    faces = np.asarray(triangles)
    # Reuse the endpoint diagnostic's finite-data and connectivity validation.
    sphere_surface_gaps(q, faces, np.zeros(3), 1.0)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    vertices = q[faces]
    lo, hi = vertices.min(axis=1), vertices.max(axis=1)
    order = np.argsort(lo[:, 0])
    x = lo[order, 0]
    candidates = []
    for rank, i in enumerate(order):
        end = np.searchsorted(x, hi[i, 0] + tolerance, side="right")
        other = order[rank + 1 : end]
        mask = np.all(lo[other, 1:] <= hi[i, 1:] + tolerance, axis=1)
        mask &= np.all(hi[other, 1:] >= lo[i, 1:] - tolerance, axis=1)
        other = other[mask]
        shared = np.any(faces[other, :, None] == faces[i, None, :], axis=(1, 2))
        other = other[~shared]
        if len(other):
            candidates.append(np.column_stack((np.full(len(other), i), other)))
    if not candidates:
        return np.empty((0, 2), dtype=np.int32)
    pairs = np.concatenate(candidates)
    overlaps = []
    for start in range(0, len(pairs), 1024):
        batch = pairs[start : start + 1024]
        a, b = vertices[batch[:, 0]], vertices[batch[:, 1]]
        ea, eb = np.roll(a, -1, axis=1) - a, np.roll(b, -1, axis=1) - b
        na, nb = np.cross(ea[:, 0], ea[:, 1]), np.cross(eb[:, 0], eb[:, 1])
        axes = np.concatenate(
            (
                na[:, None],
                nb[:, None],
                np.cross(ea[:, :, None], eb[:, None, :]).reshape(-1, 9, 3),
                np.cross(na[:, None], ea),
                np.cross(nb[:, None], eb),
            ),
            axis=1,
        )
        length = np.linalg.norm(axes, axis=2, keepdims=True)
        axes = np.divide(axes, length, out=np.zeros_like(axes), where=length > 0)
        # Shift each pair to a common local origin before projection.
        origin = a[:, :1].copy()
        pa = np.einsum("nvj,naj->nva", a - origin, axes)
        pb = np.einsum("nvj,naj->nva", b - origin, axes)
        separate = (
            (pa.max(axis=1) < pb.min(axis=1) - tolerance)
            | (pb.max(axis=1) < pa.min(axis=1) - tolerance)
        ).any(axis=1)
        overlaps.append(batch[~separate])
    return np.concatenate(overlaps).astype(np.int32)
