"""Surface ink on the foremost z(x,y) chart of a curved coating."""
import numpy as np

from .markers import MaterialBinding


def lift_to_coating(rest, triangles, uv):
    """Lift material UV samples onto the foremost rest coating, without snapping."""
    points = np.asarray(rest, float)[np.asarray(triangles)]
    a, b, c = points[:, 0], points[:, 1], points[:, 2]
    e1, e2 = b[:, :2]-a[:, :2], c[:, :2]-a[:, :2]
    det = e1[:, 0]*e2[:, 1]-e1[:, 1]*e2[:, 0]
    good = np.abs(det) > 1e-16
    result=[]
    for sample in np.asarray(uv, float):
        delta=sample-a[:, :2]
        wb=np.divide(delta[:, 0]*e2[:, 1]-delta[:, 1]*e2[:, 0],det,out=np.zeros_like(det),where=good)
        wc=np.divide(e1[:, 0]*delta[:, 1]-e1[:, 1]*delta[:, 0],det,out=np.zeros_like(det),where=good)
        bary=np.c_[1-wb-wc,wb,wc]
        inside=good & (bary>=-1e-10).all(1)
        ids=np.flatnonzero(inside)
        if not len(ids):
            raise ValueError("UV marker sample lies outside coating")
        candidates=np.einsum('ni,nij->nj',bary[ids],points[ids])
        result.append(candidates[np.argmax(candidates[:, 2])])
    return np.asarray(result)


class CurvedMarkerLayer:
    """Finite marker disks in rest material UV, transported by FEM barycentrics.

    The grid uses the single-valued foremost z(x,y) chart. The complete rendered
    skin may wrap around the finger; the grid does not cover every side/back
    face. Neither screen-space dots nor horizontal disks cutting the gel are used.
    """
    projected_surface_ink = True

    def __init__(self, rest, triangles, *, spacing_m=.0015, radius_m=.00012, rim_samples=12):
        if spacing_m<=2*radius_m or radius_m<=0 or rim_samples<8:
            raise ValueError("Invalid curved marker spacing/radius/rim count")
        tri=np.asarray(rest)[np.asarray(triangles)]
        lo,hi=tri.reshape(-1,3).min(0),tri.reshape(-1,3).max(0)
        x=np.arange(lo[0]+spacing_m*.5,hi[0]-spacing_m*.5,spacing_m)
        y=np.arange(lo[1]+spacing_m*.5,hi[1]-spacing_m*.5,spacing_m)
        angles=np.arange(rim_samples)*2*np.pi/rim_samples
        ring=np.c_[np.cos(angles),np.sin(angles)]*radius_m
        centres=[];rims=[]
        for uv in np.stack(np.meshgrid(x,y),axis=-1).reshape(-1,2):
            try:
                points=lift_to_coating(rest,triangles,np.vstack([uv,uv+ring]))
            except ValueError:
                continue  # Disks touching the coating edge are deliberately omitted.
            centres.append(points[0]);rims.append(points[1:])
        if not centres:
            raise ValueError("No complete marker disk fits inside the curved coating")
        self.centres=MaterialBinding.bind(rest,triangles,np.asarray(centres))
        self.rim_shape=(len(centres),rim_samples,3)
        self.rims=MaterialBinding.bind(rest,triangles,np.asarray(rims).reshape(-1,3))

    def rim_positions(self,positions):
        return self.rims.evaluate(positions).reshape(self.rim_shape)
