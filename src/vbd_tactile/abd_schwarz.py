"""Connectivity aggregates and bounded GPU additive Schwarz for ABD.

Immutable FEM adjacency/affine groups are built once on the host. All current
matrix values, unassembled contacts, Cholesky factors and PCG products stay on
GPU. This is an independent additive method, not an implementation of CEMAS.
"""
import numpy as np
import warp as wp
from . import ipc_gpu_multilevel as ml
from .abd_warp_product import sum_warp
wp.set_module_options({'enable_backward':False,'enable_mathdx_solver':False})


def connectivity_hierarchy(mass,tets,extra_groups=()):
    free=np.flatnonzero(np.asarray(mass)>0);n=len(mass)
    index=np.full(n,-1,np.int32);index[free]=np.arange(len(free))
    adjacency=[set() for _ in free]
    for element in [*np.asarray(tets),*extra_groups]:
        nodes=index[np.asarray(element,dtype=int)];nodes=nodes[nodes>=0]
        for a in nodes:adjacency[a].update(int(b) for b in nodes if b!=a)
    current=index.copy();maps=[];offsets=[];counts=[]
    while adjacency:
        unused=set(range(len(adjacency)));groups=[]
        while unused:
            seed=min(unused,key=lambda i:(len(adjacency[i]&unused),i))
            group=[seed];unused.remove(seed);frontier=adjacency[seed]&unused
            while frontier and len(group)<ml.B:
                chosen=min(frontier,key=lambda i:(-len(adjacency[i]&set(group)),len(adjacency[i]&unused),i))
                group.append(chosen);unused.remove(chosen)
                frontier=(frontier|adjacency[chosen])&unused
            groups.append(group)
        order=np.empty(len(adjacency),np.int32)
        for d,group in enumerate(groups):order[group]=d*ml.B+np.arange(len(group))
        mapping=np.full(n,-1,np.int32);mapping[free]=order[current[free]]
        maps.append(mapping);offsets.append(sum(counts));counts.append(len(groups))
        if len(groups)==len(adjacency):break
        contracted=[set() for _ in groups]
        for a,neighbors in enumerate(adjacency):
            ga=int(order[a]//ml.B)
            contracted[ga].update(int(order[b]//ml.B) for b in neighbors if order[b]//ml.B!=ga)
        adjacency=contracted;current[free]=mapping[free]//ml.B
    return np.asarray(maps,np.int32).reshape(-1,n),np.asarray(offsets,np.int32),counts


@wp.kernel
def assemble_remaining(ids:wp.array(dtype=wp.vec4i),indices:wp.array(dtype=int),count:wp.array(dtype=int),
        missing:wp.array(dtype=int),h:wp.array3d(dtype=wp.float64),maps:wp.array2d(dtype=int),
        offsets:wp.array(dtype=int),blocks:wp.array3d(dtype=wp.float64),lanes:int):
    lane,a,b=wp.tid()
    for slot in range(lane,count[0],lanes):
        i=indices[slot]
        if missing[i] & (1<<(4*a+b)):
            row,col=ids[i][a],ids[i][b]
            # JointMatrix contains all affine columns; these missing blocks
            # therefore refer only to ordinary FEM coordinates.
            if row>=0 and col>=0:
                for level in range(maps.shape[0]):
                    r,c=maps[level,row],maps[level,col]
                    if r>=0 and c>=0 and r//8==c//8:
                        d=offsets[level]+r//8
                        for k in range(3):
                            for l in range(3):wp.atomic_add(blocks,d,3*(r%8)+k,3*(c%8)+l,h[i,3*a+k,3*b+l])


@wp.kernel
def restrict_warp(x:wp.array(dtype=wp.vec3d),offsets:wp.array(dtype=int),
                  indices:wp.array(dtype=int),rhs:wp.array2d(dtype=wp.float64)):
    # One full warp per aggregate slot, including empty slots. Writing every
    # slot removes the memset and atomic accumulation without dropping nodes.
    key,lane=wp.tid();v=wp.vec3d(0.)
    for p in range(offsets[key]+lane,offsets[key+1],32):
        v+=x[indices[p]]
    a=sum_warp(v[0]);b=sum_warp(v[1]);c=sum_warp(v[2])
    if lane==0:
        rhs[key//8,3*(key%8)]=a
        rhs[key//8,3*(key%8)+1]=b
        rhs[key//8,3*(key%8)+2]=c


def complete_restriction_offsets(keys,offsets,slots):
    counts=np.zeros(slots,dtype=np.int64)
    np.add.at(counts,np.asarray(keys),np.diff(offsets))
    ends=np.r_[0,np.cumsum(counts)]
    if ends[-1]>np.iinfo(np.int32).max:raise ValueError('Schwarz restriction plan exceeds int32 capacity')
    return ends.astype(np.int32)


class JointSchwarz(ml.MultilevelSchwarz):
    def __init__(self,c,matrix,*,coupled_coarse=True,cache_symmetry=False,cooperative_restriction=False):
        self.cache_symmetry=bool(cache_symmetry)
        self.cooperative_restriction=bool(cooperative_restriction)
        mass=c.dof_mass.numpy();tets=c.s.model.tet_indices.numpy()
        groups=[np.arange(c.n+4*i,c.n+4*i+4) for i in range(c.nb)]
        data=connectivity_hierarchy(mass,tets,groups)
        base_levels=len(data[0])
        if coupled_coarse and base_levels:
            # Fine aggregates stay within connected components. Extra coarse
            # spaces couple each gel's translation to each body's four affine
            # coordinates; current contact couplings enter through P^T H P.
            # They add an SPD correction without changing the physical matrix.
            maps,offsets,counts=data;extra=[]
            components=maps[-1,:c.n]//8
            for component in np.unique(components[components>=0]):
                for body in range(c.nb):
                    mapping=np.full(c.nd,-1,np.int32)
                    mapping[np.flatnonzero(components==component)]=0
                    mapping[c.n+4*body:c.n+4*body+4]=np.arange(1,5)
                    extra.append(mapping)
            if extra:
                maps=np.vstack([maps,*extra]);counts=[*counts,*([1]*len(extra))]
                offsets=np.r_[0,np.cumsum(counts)[:-1]].astype(np.int32)
                data=maps,offsets,counts
        super().__init__(np.zeros((c.nd,3)),mass,tets,matrix,c.device,hierarchy_data=data)
        keys,offsets,_=self.restriction_plan
        self.restriction_offsets=wp.array(complete_restriction_offsets(keys.numpy(),offsets.numpy(),len(self.factors)*8),
                                         dtype=int,device=self.device)
        self.symmetric_inverse=wp.empty_like(self.inverse) if self.cache_symmetry else None
        weights=self.weights.numpy();weights[base_levels:]=1.;self.weights.assign(weights)

    def build(self,matrix,contacts,stats):
        self.blocks.zero_()
        self.launch(ml.assemble_fixed,len(self.assembly_plan[0]),[matrix.values,*self.assembly_plan,self.blocks])
        for b in contacts:
            missing,indices,count=matrix.remaining[b];lanes=min(b.capacity,128)
            self.launch(assemble_remaining,(lanes,4,4),[b.ids,indices,count,missing,
                b.h.view(wp.float64).reshape((b.capacity,12,12)),self.maps,self.offsets,self.blocks,lanes])
        self.launch(ml.prepare_blocks,(len(self.factors),24,24),[self.blocks,self.used,self.prepared])
        if len(self.factors):
            wp.launch_tiled(ml.factor_inverse,dim=len(self.factors),inputs=[self.prepared,self.identity,self.factors,self.inverse],
                block_dim=32,device=self.device)
        self.launch(ml.check_factor,(len(self.factors),24),[self.factors,self.inverse,stats])
        if self.cache_symmetry:
            self.launch(cache_symmetric_inverse,(len(self.factors),24,24),[self.inverse,self.symmetric_inverse])

    def apply(self,x,y,z,alpha,beta):
        # Bound each gather to 32 entries. A single coarse aggregate can
        # contain the whole fingertip. Avoid the unbounded serial gathers in
        # the first prototype (121 us total preconditioner application).
        if self.cooperative_restriction:
            wp.launch(restrict_warp,(len(self.restriction_offsets)-1,32),
                [x,self.restriction_offsets,self.restriction_plan[2],self.rhs],block_dim=32,device=self.device)
        else:
            self.rhs.zero_()
            self.launch(ml.restrict,len(self.restriction_plan[0]),[x,*self.restriction_plan,self.rhs])
        if self.cache_symmetry:
            self.launch(cached_local_product,(len(self.factors),24),[self.symmetric_inverse,self.rhs,self.solution])
        else:
            self.launch(ml.local_product,(len(self.factors),24),[self.inverse,self.rhs,self.solution])
        self.launch(ml.prolong,self.n,[x,y,self.maps,self.offsets,self.weights,self.solution,alpha,beta,z])


@wp.kernel
def cache_symmetric_inverse(inverse:wp.array3d(dtype=wp.float64),cached:wp.array3d(dtype=wp.float64)):
    d,row,col=wp.tid()
    cached[d,row,col]=wp.float64(.5)*(inverse[d,row,col]+inverse[d,col,row])


@wp.kernel
def cached_local_product(inverse:wp.array3d(dtype=wp.float64),rhs:wp.array2d(dtype=wp.float64),
                         solution:wp.array2d(dtype=wp.float64)):
    d,row=wp.tid()
    value=wp.float64(0.)
    for col in range(24):
        value+=inverse[d,row,col]*rhs[d,col]
    solution[d,row]=value
