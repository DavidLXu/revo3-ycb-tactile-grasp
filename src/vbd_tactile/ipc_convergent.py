"""GPU inclusion/exclusion terms for convergent IPC on triangle surfaces.

Implements the area-weighted IMPROVED_MAX_APPROX construction described by
IPC Toolkit 1.6.0, normal_collisions_builder.cpp (MIT; license in licenses/).
The terms remain unmerged: their signed energies/derivatives add exactly.
PT: subtract interior target edges, add interior target vertices.
EE: for each source edge and target vertex, add its PE barrier and subtract
the PE barrier times each incident edge's EE mollifier. Keeping this expanded
form avoids caching a classification or a weight that changes with geometry.

This repairs duplicate contributions, not quadrature underresolution. It is
not GCP, and convergence does not mean exact force equality on coarse meshes.
"""
from typing import Any
import numpy as np
import warp as wp
from .ipc_gpu_contacts import append_pair, candidate_close
from .ipc_precision import position_kernel, position_overloads
from .ipc_self_audit import same_rigid

wp.set_module_options({'enable_backward': False})


@position_overloads('x')
@wp.kernel
def corrections(x:wp.array(dtype=Any), invalid:wp.array(dtype=int), groups:wp.array(dtype=int),
        vertices:wp.array2d(dtype=int), edges:wp.array2d(dtype=int),
        va:wp.array(dtype=wp.float64), ea:wp.array(dtype=wp.float64), length2:wp.array(dtype=wp.float64),
        vo:wp.array(dtype=int), ve:wp.array(dtype=int), eo:wp.array(dtype=int), opposite:wp.array(dtype=int),
        interior:wp.array(dtype=int), bvh:wp.uint64, threshold:wp.float64, conservative:bool,
        output:wp.array(dtype=wp.vec4i), kinds:wp.array(dtype=int), areas:wp.array(dtype=wp.float64),
        eps:wp.array(dtype=wp.float64), count:wp.array(dtype=int), capacity:int):
    if invalid[0]:return
    p=vertices[wp.tid(),0];point=wp.vec3d(x[p])
    pad=float(threshold)*.5+8.*1.1920928955078125e-7*wp.length(wp.vec3(x[p]))
    query=wp.bvh_query_aabb(bvh,wp.vec3(x[p])-wp.vec3(pad),wp.vec3(x[p])+wp.vec3(pad))
    edge=int(0)
    while wp.bvh_query_next(query,edge):
        a=edges[edge,0];b=edges[edge,1]
        if p!=a and p!=b and not same_rigid(groups,p,a,b,b):
            if candidate_close(point,wp.vec3d(x[a]),wp.vec3d(x[b]),wp.vec3d(x[b]),0,threshold,conservative):
                faces=eo[edge+1]-eo[edge]
                for k in range(eo[edge],eo[edge+1]):
                    if opposite[k]==p:faces-=1
                eligible=int(0)
                for k in range(vo[p],vo[p+1]):
                    other=ve[k];q=edges[other,0]
                    if q==p:q=edges[other,1]
                    if q!=a and q!=b:eligible+=1
                area=wp.float64(0.)
                if eligible>0:area=wp.float64(.25)*ea[edge]
                if faces>1:area+=wp.float64(.25)*wp.float64(1-faces)*va[p]
                if area!=wp.float64(0.):
                    append_pair(wp.vec4i(p,a,b,b),0,area,wp.float64(0.),output,kinds,areas,eps,count,capacity)
                # Every term uses the distance from endpoint p to edge (a,b),
                # but retains both endpoints of the other edge for mollifier
                # derivatives. kind=2 is this forced PE distance on an EE stencil.
                for k in range(vo[p],vo[p+1]):
                    other=ve[k];c=edges[other,0];d=edges[other,1];q=c
                    if c==p:q=d
                    if q!=a and q!=b:
                        append_pair(wp.vec4i(a,b,p,q),2,-wp.float64(.25)*ea[edge],
                            wp.float64(1e-3)*length2[edge]*length2[other],output,kinds,areas,eps,count,capacity)
            # Unique PP enumeration through q's first incident edge, independent
            # of the current closest-feature classification or candidate order.
            for endpoint in range(2):
                q=edges[edge,endpoint]
                if p<q and ve[vo[q]]==edge and not same_rigid(groups,p,q,q,q):
                    adjacent=bool(False)
                    for k in range(vo[p],vo[p+1]):
                        ee=ve[k]
                        adjacent=adjacent or edges[ee,0]==q or edges[ee,1]==q
                    if not adjacent and candidate_close(point,wp.vec3d(x[q]),wp.vec3d(x[q]),wp.vec3d(x[q]),0,threshold,conservative):
                        area=wp.float64(.25)*(wp.float64(interior[q])*va[p]+wp.float64(interior[p])*va[q])
                        if area!=wp.float64(0.):
                            append_pair(wp.vec4i(p,q,q,q),0,area,wp.float64(0.),output,kinds,areas,eps,count,capacity)


class ConvergentSurface:
    def __init__(self,mesh):
        self.mesh=mesh
        # Immutable reference topology only; evolving candidates remain on GPU.
        edges=mesh.edges.numpy();faces=mesh.faces.numpy();n=mesh._node_count
        lookup={tuple(e):i for i,e in enumerate(edges)}
        ve=[[] for _ in range(n)];opposite=[[] for _ in edges]
        for i,(a,b) in enumerate(edges):ve[a].append(i);ve[b].append(i)
        for face in faces:
            for i in range(3):
                a,b=sorted((int(face[i]),int(face[(i+1)%3])))
                opposite[lookup[a,b]].append(int(face[(i+2)%3]))
        max_degree=max(map(len,ve),default=0)
        domain=len(mesh.vertices)*len(edges)*(max_degree+4)+len(mesh.vertices)*len(faces)+len(edges)*(len(edges)-1)//2
        if domain>=np.iinfo(np.int32).max:
            raise ValueError('convergent candidates exceed the int32 counter domain')
        interior=np.ones(n,np.int32)
        for i,opp in enumerate(opposite):
            if len(opp)==1:interior[edges[i]]=0
        def array(v):return wp.array(np.asarray(v,np.int32),dtype=int,device=mesh.device)
        self.vo=array(np.r_[0,np.cumsum([len(v) for v in ve])]);self.ve=array([i for v in ve for i in v])
        self.eo=array(np.r_[0,np.cumsum([len(v) for v in opposite])]);self.opposite=array([i for v in opposite for i in v])
        self.interior=array(interior)

    def search(self,x,batch,threshold,conservative):
        m=self.mesh;b=batch
        wp.launch(position_kernel(corrections,x),len(m.vertices),[x,m.invalid,m.groups,m.vertices,m.edges,
            m.vertex_area,m.edge_area,m.edge_length2,self.vo,self.ve,self.eo,self.opposite,self.interior,
            m.bvhs[1].id,threshold,conservative,b.ids,b.kinds,b.areas,b.eps,b.count,b.capacity],device=m.device)
