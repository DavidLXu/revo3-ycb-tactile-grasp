"""Validated closed rigid surface meshes. No tetrahedralization or ABD required."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class RigidTriangleMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    name: str = "rigid"

    def __post_init__(self):
        v, f = np.asarray(self.vertices, float), np.asarray(self.triangles)
        if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
            raise ValueError("rigid vertices must be finite (N,3) coordinates in metres")
        if (f.ndim != 2 or f.shape[1] != 3 or len(f) < 4
                or not np.issubdtype(f.dtype, np.integer) or f.min() < 0 or f.max() >= len(v)
                or np.any(np.diff(np.sort(f, axis=1), axis=1) == 0)):
            raise ValueError("rigid triangles must contain valid distinct integer indices")
        if len(np.unique(f)) != len(v) or len(np.unique(v, axis=0)) != len(v):
            raise ValueError("rigid mesh must be welded and have no unused vertices")
        if len(np.unique(np.sort(f, axis=1), axis=0)) != len(f):
            raise ValueError("duplicate rigid triangles")
        tri = v[f]
        scale = np.ptp(v, axis=0).max()
        if np.any(np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) <= 1e-14*scale**2):
            raise ValueError("degenerate rigid triangles")
        directed = f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
        winding = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
        if np.any(counts != 2) or np.any(winding != 0):
            raise ValueError("rigid mesh must be closed, manifold and consistently wound")
        # One connected shell: disconnected/nested shells need explicit solid semantics.
        neighbors = [[] for _ in v]
        for a, b in edges:
            neighbors[a].append(b); neighbors[b].append(a)
        visited, stack = {0}, [0]
        while stack:
            for j in neighbors[stack.pop()]:
                if j not in visited:
                    visited.add(j); stack.append(j)
        if len(visited) != len(v):
            raise ValueError("use one connected closed shell per rigid object")
        local = tri-v.mean(axis=0)
        volume = np.einsum('ij,ij->i', local[:, 0], np.cross(local[:, 1], local[:, 2])).sum()/6
        if abs(volume) <= 1e-14*scale**3:
            raise ValueError("rigid mesh must enclose nonzero volume")
        if volume < 0:
            f = f[:, ::-1]
        v, f = np.array(v, copy=True), np.array(f, dtype=np.int32, copy=True)
        v.setflags(write=False); f.setflags(write=False)
        object.__setattr__(self, 'vertices', v)
        object.__setattr__(self, 'triangles', f)

    @classmethod
    def load(cls, path, *, scale_to_m=1.0):
        """OBJ/STL/PLY etc. via Trimesh; units must be supplied explicitly."""
        import trimesh
        if not np.isfinite(scale_to_m) or scale_to_m <= 0:
            raise ValueError("scale_to_m must be finite and positive")
        mesh = trimesh.load(str(path), force='mesh', process=True)
        return cls(np.asarray(mesh.vertices)*scale_to_m, mesh.faces, Path(path).stem)

    @classmethod
    def box(cls, extents_m, *, name="box"):
        e = np.asarray(extents_m, float)
        if e.shape != (3,) or not np.isfinite(e).all() or np.any(e <= 0):
            raise ValueError("box extents must be three positive lengths")
        v = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)*e/2
        f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                      [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
        return cls(v, f, name)

    def subdivided(self, levels=1):
        """Split each triangle into four without changing the piecewise-flat solid."""
        if isinstance(levels, bool) or not isinstance(levels, (int, np.integer)) or not 0 <= levels <= 5:
            raise ValueError("subdivision levels must be an integer in [0,5]")
        v, f = self.vertices.copy(), self.triangles.copy()
        for _ in range(levels):
            edges, inverse = np.unique(np.sort(f[:, [[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_inverse=True)
            mid = inverse.reshape(-1,3)+len(v)
            v = np.vstack((v,v[edges].mean(1)))
            a,b,c = f.T; ab,bc,ca=mid.T
            f = np.vstack((np.column_stack((a,ab,ca)),np.column_stack((ab,b,bc)),
                           np.column_stack((ca,bc,c)),np.column_stack((ab,bc,ca))))
        return type(self)(v,f,self.name)

    def canonicalized_planar(self):
        """Remove redundant coplanar subdivision, preserving the rigid solid."""
        from .rigid_mesh_planar import canonical_planar_mesh
        return canonical_planar_mesh(self)


def append_rigid_meshes(builder, meshes, poses, *, particle_radius_m):
    """Append mass-zero collision vertices, never elastic tetrahedra."""
    from .mounted_ipc import rigid_poses
    import warp as wp
    poses = rigid_poses(poses)
    if len(meshes) != len(poses):
        raise ValueError("provide one pose per rigid mesh")
    ranges, triangles = [], []
    for mesh, pose in zip(meshes, poses):
        if not isinstance(mesh, RigidTriangleMesh):
            raise TypeError("meshes must be validated RigidTriangleMesh instances")
        start = len(builder.particle_mass)
        world = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
        for p in world:
            builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0), mass=0., radius=particle_radius_m)
        ranges.append(slice(start, start+len(world)))
        triangles.append(mesh.triangles+start)
    return ranges, np.concatenate(triangles)
