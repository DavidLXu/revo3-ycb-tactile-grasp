"""Geometric observation helpers, independent of the physics integrator."""
import numpy as np

from .markers import _elements, _points


def projected_surface_depth(uv, z, triangles, samples):
    """Nearest coating depth at continuous pixel coordinates, without rounding.

    Callers must clip/reject triangles crossing the camera near plane first.
    Bounding-box batches avoid allocating markers x triangles x 3 arrays.
    """
    projected = uv[triangles]
    lo, hi = projected.min(axis=1), projected.max(axis=1)
    depth = np.full(len(samples), np.inf)
    for start in range(0, len(samples), 64):
        points = samples[start:start + 64]
        candidates = ((points[:, None] >= lo[None] - 1e-9)
                      & (points[:, None] <= hi[None] + 1e-9)).all(axis=2)
        m, f = np.nonzero(candidates)
        v = projected[f]
        a, b = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
        det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        good = np.abs(det) > 1e-12
        m, f, a, b, det = m[good], f[good], a[good], b[good], det[good]
        d = points[m] - projected[f, 0]
        s = (d[:, 0] * b[:, 1] - d[:, 1] * b[:, 0]) / det
        t = (a[:, 0] * d[:, 1] - a[:, 1] * d[:, 0]) / det
        w = np.column_stack((1 - s - t, s, t))
        inside = (w >= -1e-9).all(axis=1)
        distances = 1 / (w[inside] / z[triangles[f[inside]]]).sum(axis=1)
        np.minimum.at(depth, start + m[inside], distances)
    return depth


def triangle_plane_section(positions, triangles, *, y_m=0., tolerance_m=1e-10):
    """Actual deformed triangle / world-y plane intersection as (N, 2, 3).

    Returns independent segments, never joins disconnected/folded components.
    Shared edges are deduplicated; fully coplanar faces contribute only the
    boundary of their coplanar patch. Isolated tangent points are not lines.
    """
    q = _points(positions, "positions")
    tris = _elements(triangles, 3, len(q))
    if not np.isfinite(y_m) or not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("plane and positive tolerance must be finite")
    d = q[:, 1] - y_m
    d = np.where(abs(d) <= tolerance_m, 0., d)
    active = tris[(d[tris].min(axis=1) <= 0) & (d[tris].max(axis=1) >= 0)]
    segments, coplanar_edges = [], {}
    for ids in active:
        if np.all(d[ids] == 0):
            for i, j in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((int(ids[i]), int(ids[j]))))
                coplanar_edges[edge] = coplanar_edges.get(edge, 0) + 1
            continue
        points = [q[i].copy() for i in ids if d[i] == 0]
        for i, j in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            if d[i] * d[j] < 0:
                points.append(q[i] + d[i] / (d[i] - d[j]) * (q[j] - q[i]))
        if len(points) == 2:
            segments.append(points)
    segments.extend(q[list(edge)] for edge, count in coplanar_edges.items() if count == 1)
    unique = {}
    for segment in segments:
        segment = np.array(segment, dtype=float)
        segment[:, 1] = y_m
        if np.linalg.norm(segment[1] - segment[0]) <= tolerance_m:
            continue
        key = tuple(sorted(tuple(np.rint(p / tolerance_m).astype(np.int64)) for p in segment))
        unique.setdefault(key, segment)
    return np.array(list(unique.values()), dtype=float).reshape(-1, 2, 3)


def section_height_change(segments, reference, *, tolerance_m=1e-10):
    """Section endpoint dz against the unloaded profile at the SAME world x.

    No extrapolation over holes. Multiple distinct reference heights at one x
    are ambiguous and return NaN instead of inventing a correspondence.
    """
    result = np.full((len(segments), 2), np.nan)
    for index, point in enumerate(np.asarray(segments).reshape(-1, 3)):
        heights = []
        for a, b in reference:
            if abs(b[0] - a[0]) <= tolerance_m:
                continue
            t = (point[0] - a[0]) / (b[0] - a[0])
            if -1e-10 <= t <= 1 + 1e-10:
                heights.append(a[2] + t * (b[2] - a[2]))
        if heights and np.ptp(heights) <= tolerance_m:
            result.flat[index] = point[2] - np.mean(heights)
    return result
