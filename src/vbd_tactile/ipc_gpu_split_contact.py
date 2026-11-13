"""Separate the common rank-five projection from rare mollified EE work.

The original low-rank kernel carries a full 12x12 eigensolver in its alternate
branch. Separating it is intended to reduce register/local-memory pressure;
actual speedup must be benchmarked. The formulas and stencil-indexed output match
ipc_contact_spectral.normals. A GPU queue preserves every mollified stencil.
"""
import warp as wp
from .ipc_gpu_math import mat12,contact_terms,psd12
from .ipc_primitives_warp import _weights,_delta
from .ipc_gpu_contact_hessian import has_free_node
from .ipc_contact_spectral import contact_hessian
wp.set_module_options({'enable_backward':False})


@wp.kernel
def reduced(x:wp.array(dtype=wp.vec3d),ids:wp.array(dtype=wp.vec4i),kinds:wp.array(dtype=int),
        areas:wp.array(dtype=wp.float64),eps:wp.array(dtype=wp.float64),count:wp.array(dtype=int),
        dhat:wp.float64,dmin:wp.float64,stiff:wp.float64,hessians:wp.array(dtype=mat12),
        mass:wp.array(dtype=float),pending:wp.array(dtype=int),pending_count:wp.array(dtype=int)):
    i=wp.tid()
    if i<count[0]:
        ind=ids[i];h=mat12(0)
        if areas[i]!=wp.float64(0) and has_free_node(ind,mass):
            a,b,c,d=x[ind[0]],x[ind[1]],x[ind[2]],x[ind[3]];kind=kinds[i]
            w=_weights(a,b,c,d,kind);r=_delta(w,a,b,c,d);q=wp.dot(r,r)
            if wp.sqrt(q)<dmin+dhat:
                cross=wp.cross(b-a,d-c)
                if kind!=0 and wp.dot(cross,cross)<eps[i]:
                    slot=wp.atomic_add(pending_count,0,1);pending[slot]=i
                else:h=contact_hessian(a,b,c,d,w,r,kind,dhat,dmin,stiff,areas[i])
        hessians[i]=h


@wp.kernel
def mollified(x:wp.array(dtype=wp.vec3d),ids:wp.array(dtype=wp.vec4i),kinds:wp.array(dtype=int),
        areas:wp.array(dtype=wp.float64),eps:wp.array(dtype=wp.float64),
        dhat:wp.float64,dmin:wp.float64,stiff:wp.float64,hessians:wp.array(dtype=mat12),
        pending:wp.array(dtype=int),pending_count:wp.array(dtype=int)):
    slot=wp.tid()
    if slot<pending_count[0]:
        i=pending[slot];ind=ids[i]
        energy,gradient,h,w=contact_terms(x[ind[0]],x[ind[1]],x[ind[2]],x[ind[3]],
            kinds[i],dhat,dmin,stiff,areas[i],wp.float64(0),eps[i],True)
        hessians[i]=psd12(h)


@wp.kernel
def mollified_raw(x:wp.array(dtype=wp.vec3d),ids:wp.array(dtype=wp.vec4i),kinds:wp.array(dtype=int),
        areas:wp.array(dtype=wp.float64),eps:wp.array(dtype=wp.float64),
        dhat:wp.float64,dmin:wp.float64,stiff:wp.float64,hessians:wp.array(dtype=mat12),
        pending:wp.array(dtype=int),pending_count:wp.array(dtype=int)):
    slot=wp.tid()
    if slot<pending_count[0]:
        i=pending[slot];ind=ids[i]
        energy,gradient,h,w=contact_terms(x[ind[0]],x[ind[1]],x[ind[2]],x[ind[3]],
            kinds[i],dhat,dmin,stiff,areas[i],wp.float64(0),eps[i],True)
        hessians[i]=h


class SplitContactHessian:
    def __init__(self,capacity,device,*,cooperative=False,merge_features=False,node_count=0):
        self.capacity=capacity;self.device=wp.get_device(device)
        self.cooperative=bool(cooperative)
        if self.cooperative and not self.device.is_cuda:
            raise ValueError('cooperative contact projection requires CUDA')
        self.pending=wp.empty(capacity,dtype=int,device=device)
        self.count=wp.zeros(1,dtype=int,device=device)
        self.features=None
        if merge_features:
            from .ipc_feature_projection import FeatureProjection
            self.features=FeatureProjection(capacity,node_count,device)

    def build_unprojected(self,x,batch,mesh,mass):
        if batch.capacity!=self.capacity:raise ValueError('contact Hessian queue capacity mismatch')
        if self.features is not None:raise ValueError('raw contact tangent does not support feature merging')
        from .abd_coupled_tangent import raw_reduced
        self.count.zero_()
        wp.launch(raw_reduced,self.capacity,[x,batch.ids,batch.kinds,batch.areas,batch.eps,batch.count,
            mesh.dhat,mesh.dmin,mesh.stiffness,batch.h,mass,self.pending,self.count],device=self.device)
        def full():
            wp.launch(mollified_raw,self.capacity,[x,batch.ids,batch.kinds,batch.areas,batch.eps,
                mesh.dhat,mesh.dmin,mesh.stiffness,batch.h,self.pending,self.count],device=self.device)
        if self.device.is_cuda:wp.capture_if(self.count,full)
        else:full()

    def build(self,x,batch,mesh,mass):
        if batch.capacity!=self.capacity:raise ValueError('contact Hessian queue capacity mismatch')
        data=batch
        if self.features is not None:
            self.features.build(x,batch);data=self.features
        self.count.zero_()
        wp.launch(reduced,self.capacity,[x,data.ids,data.kinds,data.areas,data.eps,batch.count,
            mesh.dhat,mesh.dmin,mesh.stiffness,batch.h,mass,self.pending,self.count],device=self.device)
        def full():
            wp.launch(mollified_raw if self.cooperative else mollified,self.capacity,[x,data.ids,data.kinds,data.areas,data.eps,
                mesh.dhat,mesh.dmin,mesh.stiffness,batch.h,self.pending,self.count],device=self.device)
            if self.cooperative:
                from .ipc_gpu_warp_psd import project_queue
                project_queue(batch.h,self.pending,self.count,self.device)
        if self.device.is_cuda:wp.capture_if(self.count,full)
        else:full()
        if self.features is not None:self.features.scatter(batch)
