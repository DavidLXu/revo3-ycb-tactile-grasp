"""Separate BVH traversal from per-pair distance work, preserving all pairs.

A bounded broad-phase pool is only an optimization. Overflow executes the
original complete GPU query; partial pools are never filtered or published.
"""
from typing import Any
import warp as wp
from .ipc_precision import position_overloads, position_kernel
from .ipc_self_audit import same_rigid
from .ipc_gpu_ccd_pairs import collect_edges
from .ipc_gpu_contacts import candidate_close, append_pair
wp.set_module_options({'enable_backward':False})


@position_overloads('x')
@wp.kernel
def collect_points(x:wp.array(dtype=Any),invalid:wp.array(dtype=int),groups:wp.array(dtype=int),
        vertices:wp.array2d(dtype=int),faces:wp.array2d(dtype=int),bvh:wp.uint64,threshold:wp.float64,
        count:wp.array(dtype=int),pairs:wp.array(dtype=wp.vec4i),meta:wp.array(dtype=wp.vec3i)):
    if invalid[0]:return
    i=vertices[wp.tid(),0]
    padding=float(threshold)*.5+8.*1.1920928955078125e-7*wp.length(wp.vec3(x[i]))
    query=wp.bvh_query_aabb(bvh,wp.vec3(x[i])-wp.vec3(padding),wp.vec3(x[i])+wp.vec3(padding))
    f=int(0)
    while wp.bvh_query_next(query,f):
        j,k,l=faces[f,0],faces[f,1],faces[f,2]
        if i!=j and i!=k and i!=l and not same_rigid(groups,i,j,k,l):
            slot=wp.atomic_add(count,0,1)
            if slot<pairs.shape[0]:
                pairs[slot]=wp.vec4i(i,j,k,l);meta[slot]=wp.vec3i(0,i,f)


@wp.kernel
def classify(count:wp.array(dtype=int),capacity:int,overflow:wp.array(dtype=int),work:wp.array(dtype=int)):
    overflow[0]=int(count[0]>capacity)
    work[0]+=1;work[1]+=overflow[0]


@position_overloads('x')
@wp.kernel
def filter_pairs(x:wp.array(dtype=Any),invalid:wp.array(dtype=int),overflow:wp.array(dtype=int),
        pairs:wp.array(dtype=wp.vec4i),meta:wp.array(dtype=wp.vec3i),count:wp.array(dtype=int),
        vertex_area:wp.array(dtype=wp.float64),edge_area:wp.array(dtype=wp.float64),length2:wp.array(dtype=wp.float64),
        threshold:wp.float64,conservative:bool,mollifier:wp.float64,multiplicity:wp.float64,
        output:wp.array(dtype=wp.vec4i),kinds:wp.array(dtype=int),areas:wp.array(dtype=wp.float64),
        eps:wp.array(dtype=wp.float64),out_count:wp.array(dtype=int),capacity:int):
    i=wp.tid()
    if invalid[0] or overflow[0] or i>=count[0]:return
    v=pairs[i];m=meta[i]
    if candidate_close(wp.vec3d(x[v[0]]),wp.vec3d(x[v[1]]),wp.vec3d(x[v[2]]),wp.vec3d(x[v[3]]),m[0],threshold,conservative):
        area=wp.float64(0.);epsilon=wp.float64(0.)
        if m[0]==0:
            area=wp.float64(.25)*vertex_area[m[1]]
        else:
            area=wp.float64(.25)*multiplicity*(edge_area[m[1]]+edge_area[m[2]])
            epsilon=mollifier*length2[m[1]]*length2[m[2]]
        append_pair(v,m[0],area,epsilon,output,kinds,areas,eps,out_count,capacity)


class ParallelContactSearch:
    def __init__(self,mesh,capacity):
        self.device=mesh.device;self.capacity=capacity
        self.pairs=wp.empty(capacity,dtype=wp.vec4i,device=self.device)
        self.meta=wp.empty(capacity,dtype=wp.vec3i,device=self.device)
        self.count=wp.zeros(1,dtype=int,device=self.device)
        self.overflow=wp.zeros(1,dtype=int,device=self.device)
        self.work=wp.zeros(2,dtype=int,device=self.device)

    def search(self,m,x,b,threshold,conservative,fallback):
        self.count.zero_()
        wp.launch(position_kernel(collect_points,x),len(m.vertices),
            [x,m.invalid,m.groups,m.vertices,m.faces,m.bvhs[0].id,threshold,self.count,self.pairs,self.meta],device=self.device)
        wp.launch(collect_edges,len(m.edges),[m.invalid,m.groups,m.edges,*m.bounds[1],m.bvhs[1].id,
            self.count,self.pairs,self.meta],device=self.device)
        wp.launch(classify,1,[self.count,self.capacity,self.overflow,self.work],device=self.device)
        wp.launch(position_kernel(filter_pairs,x),self.capacity,[x,m.invalid,self.overflow,self.pairs,self.meta,self.count,
            m.vertex_area,m.edge_area,m.edge_length2,threshold,conservative,m.mollifier_coefficient,m.ee_multiplicity,
            b.ids,b.kinds,b.areas,b.eps,b.count,b.capacity],device=self.device,block_dim=64)
        wp.capture_if(self.overflow,fallback)
