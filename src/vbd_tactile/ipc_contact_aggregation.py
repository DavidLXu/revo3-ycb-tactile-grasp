"""GPU Lp aggregation of contact barriers sharing a material quadrature source.

E = sum_s A_s ||b_s||_p, A_s = rest dual area / 4. PT has one vertex
source; EE has both edge sources. b includes the existing physical barrier
and EE mollifier, without area. No collision candidate is removed.

Euler homogeneity gives E=sum_i c_i*b_i and grad(E)=sum_i c_i*grad(b_i).
The Hessian also contains a PSD covariance, including NONLOCAL cross-contact
terms. Its positive diagonal is added to the existing projected stencils;
its negative rank-one terms are applied matrix-free. Dropping these terms
would not be Newton's method for the aggregated energy.
"""
from typing import Any
import numpy as np
import warp as wp
from .ipc_gpu_math import vec12, mat12, contact_terms
from .ipc_precision import position_overloads, position_kernel
from .abd_guides import projected

wp.set_module_options({'enable_backward': False})


@wp.func
def edge_index(a:int,b:int,edges:wp.array2d(dtype=int)):
    lo=int(0);hi=edges.shape[0];u=wp.min(a,b);v=wp.max(a,b)
    while lo<hi:
        mid=(lo+hi)//2
        if edges[mid,0]<u or (edges[mid,0]==u and edges[mid,1]<v):lo=mid+1
        else:hi=mid
    return lo


@position_overloads('x')
@wp.kernel
def unit_terms(x:wp.array(dtype=Any),ids:wp.array(dtype=wp.vec4i),kinds:wp.array(dtype=int),
        eps:wp.array(dtype=wp.float64),count:wp.array(dtype=int),edges:wp.array2d(dtype=int),
        component:wp.array(dtype=int),group_stride:int,
        n:int,dhat:wp.float64,dmin:wp.float64,stiff:wp.float64,
        source:wp.array(dtype=wp.vec2i),energy:wp.array(dtype=wp.float64),
        gradient:wp.array(dtype=vec12),maximum:wp.array(dtype=wp.float64),invalid:wp.array(dtype=int)):
    i=wp.tid()
    if count[0]<=ids.shape[0] and i<count[0]:
        ind=ids[i];s=wp.vec2i(ind[0]+component[ind[1]]*group_stride,-1)
        if kinds[i]==1:
            s=wp.vec2i(n+edge_index(ind[0],ind[1],edges)+component[ind[2]]*group_stride,
                      n+edge_index(ind[2],ind[3],edges)+component[ind[0]]*group_stride)
        source[i]=s
        e,g,h,w=contact_terms(wp.vec3d(x[ind[0]]),wp.vec3d(x[ind[1]]),wp.vec3d(x[ind[2]]),wp.vec3d(x[ind[3]]),
            kinds[i],dhat,dmin,stiff,wp.float64(1.),wp.float64(0.),eps[i],False)
        energy[i]=e;gradient[i]=g
        if not wp.isfinite(e) or e<wp.float64(0.):
            wp.atomic_max(invalid,0,1)
        else:
            for k in range(2):
                if s[k]>=0:wp.atomic_max(maximum,s[k],e)


@wp.kernel
def sums(source:wp.array(dtype=wp.vec2i),energy:wp.array(dtype=wp.float64),count:wp.array(dtype=int),
         maximum:wp.array(dtype=wp.float64),power:wp.float64,total:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if count[0]<=source.shape[0] and i<count[0] and wp.isfinite(energy[i]):
        for k in range(2):
            s=source[i][k]
            if s>=0:
                if maximum[s]>wp.float64(0.):
                    if power<wp.float64(0.):
                        if energy[i]==maximum[s]:wp.atomic_add(total,s,wp.float64(1.))
                    else:wp.atomic_add(total,s,wp.pow(energy[i]/maximum[s],power))


@wp.kernel
def source_norm(maximum:wp.array(dtype=wp.float64),total:wp.array(dtype=wp.float64),
        areas:wp.array(dtype=wp.float64),power:wp.float64,norm:wp.array(dtype=wp.float64),
        energy:wp.array(dtype=wp.float64),rank:wp.array(dtype=wp.float64)):
    s=wp.tid();value=maximum[s]
    if power>wp.float64(0.):value*=wp.pow(total[s],wp.float64(1.)/power)
    norm[s]=value;energy[s]=areas[s]*value;rank[s]=wp.float64(0.)
    if power>wp.float64(0.) and energy[s]>wp.float64(0.):rank[s]=(power-wp.float64(1.))/energy[s]


@wp.kernel
def coefficients(source:wp.array(dtype=wp.vec2i),unit_energy:wp.array(dtype=wp.float64),
        count:wp.array(dtype=int),norm:wp.array(dtype=wp.float64),total:wp.array(dtype=wp.float64),areas:wp.array(dtype=wp.float64),
        power:wp.float64,coeff:wp.array(dtype=wp.vec2d),effective:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if count[0]<=source.shape[0] and i<count[0]:
        c=wp.vec2d(0.)
        for k in range(2):
            s=source[i][k]
            if s>=0:
                if norm[s]>wp.float64(0.) and wp.isfinite(unit_energy[i]):
                    if power<wp.float64(0.):
                        if unit_energy[i]==norm[s]:c[k]=areas[s]/total[s]
                    else:c[k]=areas[s]*wp.pow(unit_energy[i]/norm[s],power-wp.float64(1.))
        coeff[i]=c;effective[i]=c[0]+c[1]


@wp.kernel
def positive_hessian(energy:wp.array(dtype=wp.float64),gradient:wp.array(dtype=vec12),
        coeff:wp.array(dtype=wp.vec2d),count:wp.array(dtype=int),power:wp.float64,
        ids:wp.array(dtype=wp.vec4i),mass:wp.array(dtype=float),h:wp.array(dtype=mat12)):
    i=wp.tid()
    if count[0]<=energy.shape[0] and i<count[0]:
        free=False
        for k in range(4):free=free or mass[ids[i][k]]>0.
        if power>wp.float64(0.) and free and energy[i]>wp.float64(0.):
            factor=(power-wp.float64(1.))*(coeff[i][0]+coeff[i][1])/energy[i]
            h[i]+=factor*wp.outer(gradient[i],gradient[i])


@wp.func
def mapped_direction(x:wp.array(dtype=wp.vec3d),node:int,owner:wp.array(dtype=int),
        weights:wp.array(dtype=wp.vec4d),index:wp.array(dtype=int),mass:wp.array(dtype=float),
        axes:wp.array(dtype=wp.vec3d),n:int):
    result=wp.vec3d(0.)
    if owner[node]>=0:
        for col in range(4):
            dof=n+4*owner[node]+col
            result+=weights[node][col]*projected(x[index[dof]],dof,n,axes)
    elif mass[node]>0.:
        result=x[index[node]]
    return result


@wp.kernel
def source_product(x:wp.array(dtype=wp.vec3d),ids:wp.array(dtype=wp.vec4i),
        source:wp.array(dtype=wp.vec2i),gradient:wp.array(dtype=vec12),coeff:wp.array(dtype=wp.vec2d),
        count:wp.array(dtype=int),owner:wp.array(dtype=int),weights:wp.array(dtype=wp.vec4d),
        index:wp.array(dtype=int),mass:wp.array(dtype=float),axes:wp.array(dtype=wp.vec3d),
        n:int,product:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if i<count[0]:
        value=wp.float64(0.);g=gradient[i]
        for j in range(4):
            d=mapped_direction(x,ids[i][j],owner,weights,index,mass,axes,n)
            value+=wp.dot(wp.vec3d(g[3*j],g[3*j+1],g[3*j+2]),d)
        for k in range(2):
            s=source[i][k]
            if s>=0:wp.atomic_add(product,s,coeff[i][k]*value)


@wp.kernel
def subtract_product(ids:wp.array(dtype=wp.vec4i),source:wp.array(dtype=wp.vec2i),
        gradient:wp.array(dtype=vec12),coeff:wp.array(dtype=wp.vec2d),count:wp.array(dtype=int),
        product:wp.array(dtype=wp.float64),rank:wp.array(dtype=wp.float64),
        owner:wp.array(dtype=int),weights:wp.array(dtype=wp.vec4d),index:wp.array(dtype=int),
        mass:wp.array(dtype=float),axes:wp.array(dtype=wp.vec3d),n:int,alpha:wp.float64,
        out:wp.array(dtype=wp.vec3d)):
    i=wp.tid()
    if i<count[0]:
        value=wp.float64(0.)
        for k in range(2):
            s=source[i][k]
            if s>=0:value-=rank[s]*coeff[i][k]*product[s]
        g=gradient[i]
        for j in range(4):
            node=ids[i][j];v=alpha*value*wp.vec3d(g[3*j],g[3*j+1],g[3*j+2])
            if owner[node]>=0:
                for col in range(4):
                    dof=n+4*owner[node]+col
                    wp.atomic_add(out,index[dof],weights[node][col]*projected(v,dof,n,axes))
            elif mass[node]>0.:wp.atomic_add(out,index[node],v)


@wp.kernel
def energy_difference(now:wp.array(dtype=wp.float64),old:wp.array(dtype=wp.float64),
        out:wp.array(dtype=wp.float64)):
    s=wp.tid();wp.atomic_add(out,0,now[s]-old[s])


class ContactAggregation:
    def __init__(self,mesh,power):
        self.mesh=mesh;self.device=mesh.device;self.capacity=mesh.batch.capacity;self.power=float(power)
        n=mesh._node_count;ns=n+len(mesh.edges)
        # Static material data only. No evolving geometry/forces are downloaded.
        self.areas=wp.array(.25*np.r_[mesh.vertex_area.numpy(),mesh.edge_area.numpy()],dtype=wp.float64,device=self.device)
        def zeros(size,dtype):return wp.zeros(size,dtype=dtype,device=self.device)
        self.source=zeros(self.capacity,wp.vec2i);self.unit_energy=zeros(self.capacity,wp.float64)
        self.unit_gradient=zeros(self.capacity,vec12);self.coeff=zeros(self.capacity,wp.vec2d)
        self.maximum=zeros(ns,wp.float64);self.total=zeros(ns,wp.float64);self.norm=zeros(ns,wp.float64)
        self.energy=zeros(ns,wp.float64);self.base_energy=zeros(ns,wp.float64);self.rank=zeros(ns,wp.float64)
        self.product=zeros(ns,wp.float64)
        self.identity=wp.array(np.arange(n,dtype=np.int32),dtype=int,device=self.device)
        self.owner=wp.full(n,-1,dtype=int,device=self.device)
        self.component=zeros(n,int);self.group_stride=0
        self.weights=zeros(n,wp.vec4d);self.axes=zeros(1,wp.vec3d)

    def launch(self,kernel,dim,args):
        if dim:wp.launch(kernel,dim,args,device=self.device)

    def build(self,x):
        m=self.mesh;b=m.batch
        self.maximum.zero_();self.total.zero_()
        self.launch(position_kernel(unit_terms,x),self.capacity,[x,b.ids,b.kinds,b.eps,b.count,m.edges,self.component,self.group_stride,
            m._node_count,m.dhat,m.dmin,m.stiffness,self.source,self.unit_energy,self.unit_gradient,self.maximum,m.invalid])
        self.launch(sums,self.capacity,[self.source,self.unit_energy,b.count,self.maximum,self.power,self.total])
        self.launch(source_norm,len(self.areas),[self.maximum,self.total,self.areas,self.power,self.norm,self.energy,self.rank])
        self.launch(coefficients,self.capacity,[self.source,self.unit_energy,b.count,self.norm,self.total,self.areas,
            self.power,self.coeff,b.areas])

    def add_hessian(self,mass):
        if self.power<0:return
        b=self.mesh.batch
        self.launch(positive_hessian,self.capacity,[self.unit_energy,self.unit_gradient,self.coeff,b.count,
            self.power,b.ids,mass,b.h])

    def mapping(self,mass,c=None,index=None):
        if c is None:return [self.owner,self.weights,self.identity,mass,self.axes,self.mesh._node_count]
        return [c.owner,c.weights,c.aggregation_index if index is None else index,mass,c.guide_axes,c.n]

    def prepare_product(self,x,mass,c=None,index=None):
        if self.power<0:return
        self.product.zero_();b=self.mesh.batch
        self.launch(source_product,self.capacity,[x,b.ids,self.source,self.unit_gradient,self.coeff,b.count,
            *self.mapping(mass,c,index),self.product])

    def add_product(self,out,alpha,mass,c=None,index=None):
        if self.power<0:return
        b=self.mesh.batch
        self.launch(subtract_product,self.capacity,[b.ids,self.source,self.unit_gradient,self.coeff,b.count,
            self.product,self.rank,*self.mapping(mass,c,index),alpha,out])

    def save_energy(self):wp.copy(self.base_energy,self.energy)

    def correct_energy(self,stats):
        # Lp/max are homogeneous: sum_i c_i*b_i already equals the energy.
        pass

    def energy_change(self,out):
        self.launch(energy_difference,len(self.energy),[self.energy,self.base_energy,out])
