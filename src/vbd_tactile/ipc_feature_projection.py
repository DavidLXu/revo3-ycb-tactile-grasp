"""Merge equivalent *unmollified* contact Hessians before PSD projection.

This only changes the Newton metric. Original candidates, signed energies,
forces, friction and stable energy increments remain untouched. Geometry is
reclassified on GPU for every Hessian evaluation; no feature keys are cached.
Mollified stencils remain separate because they depend on inactive endpoints.
"""
import warp as wp
from .ipc_primitives_warp import _weights
from .ipc_gpu_math import mat12

wp.set_module_options({'enable_backward':False})


@wp.func
def order2(a:int,b:int):
    return wp.min(a,b),wp.max(a,b)


@wp.kernel
def classify(x:wp.array(dtype=wp.vec3d),ids:wp.array(dtype=wp.vec4i),kinds:wp.array(dtype=int),
        areas:wp.array(dtype=wp.float64),eps:wp.array(dtype=wp.float64),count:wp.array(dtype=int),
        radix:wp.int64,unique_base:wp.int64,keys:wp.array(dtype=wp.int64),order:wp.array(dtype=int),
        canonical:wp.array(dtype=wp.vec4i),types:wp.array(dtype=int),moll_eps:wp.array(dtype=wp.float64),
        merged:wp.array(dtype=wp.float64)):
    i=wp.tid();order[i]=i;merged[i]=wp.float64(0)
    unique=unique_base+wp.int64(i)
    keys[i]=unique
    if i<count[0]:
        ind=ids[i];kind=kinds[i];canonical[i]=ind;types[i]=kind;moll_eps[i]=eps[i]
        a,b,c,d=x[ind[0]],x[ind[1]],x[ind[2]],x[ind[3]]
        cross=wp.cross(b-a,d-c)
        if kind==0 or wp.dot(cross,cross)>=eps[i]:
            w=_weights(a,b,c,d,kind)
            positive=wp.vec3i(-1);negative=wp.vec3i(-1);np=int(0);nn=int(0)
            for j in range(4):
                if w[j]>wp.float64(0):positive[np]=ind[j];np+=1
                elif w[j]<wp.float64(0):negative[nn]=ind[j];nn+=1
            feature=int(-1);out=ind
            if np==1 and nn==1:
                u,v=order2(positive[0],negative[0]);out=wp.vec4i(u,v,v,v);feature=0
            elif np+nn==3:
                p=positive[0];u,v=negative[0],negative[1]
                if np==2:p=negative[0];u,v=positive[0],positive[1]
                u,v=order2(u,v);out=wp.vec4i(p,u,v,v);feature=1
            elif np==1 and nn==3:
                u,v=order2(negative[0],negative[1]);v,z=order2(v,negative[2]);u,v=order2(u,v)
                out=wp.vec4i(positive[0],u,v,z);feature=2
            elif np==2 and nn==2:
                u,v=order2(positive[0],positive[1]);z,t=order2(negative[0],negative[1])
                if u>z:u,z=z,u;v,t=t,v
                out=wp.vec4i(u,v,z,t);feature=3
            if feature>=0:
                key=wp.int64(feature)
                for j in range(4):key=key*radix+wp.int64(out[j])
                keys[i]=key;canonical[i]=out;types[i]=int(feature==3);moll_eps[i]=wp.float64(0)


@wp.kernel
def sum_groups(keys:wp.array(dtype=wp.int64),order:wp.array(dtype=int),areas:wp.array(dtype=wp.float64),
        count:wp.array(dtype=int),capacity:int,merged:wp.array(dtype=wp.float64),stats:wp.array(dtype=int)):
    k=wp.tid();i=order[k]
    if i<count[0]:
        leader=bool(k==0)
        if k>0:leader=keys[k]!=keys[k-1]
        if leader:
            area=areas[i];j=k+1
            while j<capacity:
                if keys[j]!=keys[k]:break
                area+=areas[order[j]];j+=1
            merged[i]=area
            wp.atomic_add(stats,0,1)
            if area!=wp.float64(0):wp.atomic_add(stats,1,1)


@wp.kernel
def scatter_back(ids:wp.array(dtype=wp.vec4i),canonical:wp.array(dtype=wp.vec4i),
        areas:wp.array(dtype=wp.float64),count:wp.array(dtype=int),h:wp.array(dtype=mat12)):
    i=wp.tid()
    if i<count[0] and areas[i]!=wp.float64(0):
        src=canonical[i];dst=ids[i];same=True
        for a in range(4):same=same and src[a]==dst[a]
        if not same:
            old=h[i];out=mat12(0);mapping=wp.vec4i(-1)
            for a in range(4):
                for b in range(4):
                    if mapping[a]<0 and src[a]==dst[b]:mapping[a]=b
            for a in range(4):
                for b in range(4):
                    for r in range(3):
                        for s in range(3):
                            if mapping[a]>=0 and mapping[b]>=0:
                                out[3*mapping[a]+r,3*mapping[b]+s]+=old[3*a+r,3*b+s]
            h[i]=out


class FeatureProjection:
    def __init__(self,capacity,node_count,device):
        if not 1<=node_count<=32768:
            raise ValueError('merged contact projection supports 1..32768 vertices (collision-free int64 key)')
        self.capacity=capacity;self.node_count=node_count;self.device=wp.get_device(device)
        self.unique_base=4*node_count**4
        self.sort_bits=(self.unique_base+capacity-1).bit_length()
        self.keys=wp.empty(2*capacity,dtype=wp.int64,device=device)
        self.order=wp.empty(2*capacity,dtype=int,device=device)
        self.ids=wp.empty(capacity,dtype=wp.vec4i,device=device)
        self.kinds=wp.empty(capacity,dtype=int,device=device)
        self.eps=wp.empty(capacity,dtype=wp.float64,device=device)
        self.areas=wp.zeros(capacity,dtype=wp.float64,device=device)
        self.stats=wp.zeros(2,dtype=int,device=device)
        # Reserve sort scratch before CUDA graph capture.
        self.keys.zero_();self.order.zero_()
        wp.utils.radix_sort_pairs(self.keys,self.order,capacity,end_bit=self.sort_bits)

    def build(self,x,batch):
        if len(x)!=self.node_count or batch.capacity!=self.capacity:
            raise ValueError('merged projection allocation does not match contact state')
        self.stats.zero_()
        wp.launch(classify,self.capacity,[x,batch.ids,batch.kinds,batch.areas,batch.eps,batch.count,
            wp.int64(self.node_count),wp.int64(self.unique_base),self.keys,self.order,self.ids,self.kinds,self.eps,self.areas],device=self.device)
        wp.utils.radix_sort_pairs(self.keys,self.order,self.capacity,end_bit=self.sort_bits)
        wp.launch(sum_groups,self.capacity,[self.keys,self.order,batch.areas,batch.count,
            self.capacity,self.areas,self.stats],device=self.device)

    def scatter(self,batch):
        wp.launch(scatter_back,self.capacity,[batch.ids,self.ids,self.areas,batch.count,batch.h],device=self.device)
