"""Experimental area-measure soft maximum; no candidate-count normalization.

For source s, E_s=A_s*tau*log(1+sum_t w_st*expm1(b_st/tau)).
Target weights are fixed rest areas divided by the opposing component's
total surface rest area; each opposing component has a separate source group.
missing/inactive target area carries zero potential. Unlike an unweighted Lp
norm, subdividing target measure does not itself multiply the potential.
This changes the contact potential; finite quadrature error still needs audit.
"""
import warp as wp
import numpy as np
from .ipc_contact_aggregation import ContactAggregation,unit_terms
from .ipc_precision import position_kernel
from .ipc_gpu_math import vec12,mat12
wp.set_module_options({'enable_backward':False})

@wp.func
def expm1_small(x:wp.float64):
    if wp.abs(x)<wp.float64(1e-4):
        return x*(wp.float64(1.)+x*(wp.float64(.5)+x*(wp.float64(1./6.)+x*(wp.float64(1./24.)+x*(wp.float64(1./120.)+x*wp.float64(1./720.))))))
    return wp.exp(x)-wp.float64(1.)

@wp.func
def log1p_small(x:wp.float64):
    if wp.abs(x)<wp.float64(1e-4):
        return x*(wp.float64(1.)+x*(wp.float64(-.5)+x*(wp.float64(1./3.)+x*(wp.float64(-.25)+x*(wp.float64(.2)-x/wp.float64(6.))))))
    return wp.log(wp.float64(1.)+x)

@wp.kernel
def weighted_sums(rest:wp.array(dtype=wp.vec3d),ids:wp.array(dtype=wp.vec4i),
        source:wp.array(dtype=wp.vec2i),energies:wp.array(dtype=wp.float64),count:wp.array(dtype=int),
        areas:wp.array(dtype=wp.float64),component:wp.array(dtype=int),component_area:wp.array(dtype=wp.float64),
        stride:int,maximum:wp.array(dtype=wp.float64),
        tau:wp.float64,target:wp.array(dtype=wp.vec2d),total:wp.array(dtype=wp.float64),
        measure:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if i<count[0] and count[0]<=ids.shape[0]:
        w=wp.vec2d(0.);sources=source[i];e=energies[i]
        if sources[1]<0:
            v=ids[i]
            w[0]=wp.length(wp.cross(rest[v[2]]-rest[v[1]],rest[v[3]]-rest[v[1]]))/(wp.float64(2.)*component_area[component[v[1]]])
        else:
            v=ids[i]
            w=wp.vec2d(wp.float64(4.)*areas[sources[1]%stride]/component_area[component[v[2]]],
                       wp.float64(4.)*areas[sources[0]%stride]/component_area[component[v[0]]])
        target[i]=w
        if e>wp.float64(0.) and wp.isfinite(e):
            for k in range(2):
                s=sources[k]
                if s>=0:
                    value=w[k]*wp.exp((e-maximum[s])/tau)
                    if maximum[s]<wp.float64(.5)*tau:value=w[k]*expm1_small(e/tau)
                    wp.atomic_add(total,s,value);wp.atomic_add(measure,s,w[k])

@wp.kernel
def source_values(maximum:wp.array(dtype=wp.float64),total:wp.array(dtype=wp.float64),
        measure:wp.array(dtype=wp.float64),areas:wp.array(dtype=wp.float64),tau:wp.float64,
        norm:wp.array(dtype=wp.float64),energy:wp.array(dtype=wp.float64),rank:wp.array(dtype=wp.float64),
        invalid:wp.array(dtype=int)):
    s=wp.tid();z=wp.float64(1.);value=wp.float64(0.)
    if maximum[s]<wp.float64(.5)*tau:
        z+=total[s];value=tau*log1p_small(total[s])
    else:
        z=wp.exp(-maximum[s]/tau)*(wp.float64(1.)-measure[s])+total[s]
        value=maximum[s]+tau*wp.log(z)
    if not wp.isfinite(value) or value<wp.float64(0.) or measure[s]>wp.float64(1.+1e-12):
        wp.atomic_max(invalid,0,1)
    norm[s]=z;energy[s]=areas[s]*value;rank[s]=wp.float64(0.)
    if areas[s]>wp.float64(0.):rank[s]=wp.float64(1.)/(tau*areas[s])

@wp.kernel
def coefficients(source:wp.array(dtype=wp.vec2i),energies:wp.array(dtype=wp.float64),
        target:wp.array(dtype=wp.vec2d),count:wp.array(dtype=int),areas:wp.array(dtype=wp.float64),
        maximum:wp.array(dtype=wp.float64),norm:wp.array(dtype=wp.float64),tau:wp.float64,
        coeff:wp.array(dtype=wp.vec2d),effective:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if i<count[0] and count[0]<=source.shape[0]:
        c=wp.vec2d(0.);e=energies[i]
        if e>wp.float64(0.) and wp.isfinite(e):
            for k in range(2):
                s=source[i][k]
                if s>=0:
                    shift=maximum[s]
                    if shift<wp.float64(.5)*tau:shift=wp.float64(0.)
                    c[k]=areas[s]*target[i][k]*wp.exp((e-shift)/tau)/norm[s]
        coeff[i]=c;effective[i]=c[0]+c[1]

@wp.kernel
def positive_hessian(gradient:wp.array(dtype=vec12),coeff:wp.array(dtype=wp.vec2d),
        count:wp.array(dtype=int),tau:wp.float64,ids:wp.array(dtype=wp.vec4i),
        mass:wp.array(dtype=float),h:wp.array(dtype=mat12)):
    i=wp.tid()
    if i<count[0] and count[0]<=ids.shape[0]:
        free=False
        for j in range(4):free=free or mass[ids[i][j]]>0.
        if free:h[i]+=(coeff[i][0]+coeff[i][1])/tau*wp.outer(gradient[i],gradient[i])

@wp.kernel
def remove_linear_energy(energies:wp.array(dtype=wp.float64),area:wp.array(dtype=wp.float64),
                         count:wp.array(dtype=int),stats:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if i<count[0] and count[0]<=area.shape[0]:wp.atomic_add(stats,0,-energies[i]*area[i])

@wp.kernel
def add_source_energy(energy:wp.array(dtype=wp.float64),stats:wp.array(dtype=wp.float64)):
    wp.atomic_add(stats,0,energy[wp.tid()])

class AreaSoftmax(ContactAggregation):
    def __init__(self,mesh):
        super().__init__(mesh,4.)
        self.reference=wp.array(mesh._rest_reference,dtype=wp.vec3d,device=self.device)
        # Separate each opposing surface/body. Adding a remote object must not
        # dilute an existing contact, nor may contacts on opposite sides compete.
        faces=mesh.faces.numpy();n=mesh._node_count;parents=np.arange(n)
        def root(i):
            while parents[i]!=i:
                parents[i]=parents[parents[i]];i=parents[i]
            return int(i)
        def join(a,b):parents[root(b)]=root(a)
        for a,b,c in faces:join(a,b);join(a,c)
        first={}
        for i,g in enumerate(mesh.groups.numpy()):
            if g>=0:
                if int(g) in first:join(first[int(g)],i)
                else:first[int(g)]=i
        vertices=np.unique(faces);roots={root(int(i)) for i in vertices}
        mapping={r:i for i,r in enumerate(sorted(roots))}
        components=np.zeros(n,np.int32)
        for i in vertices:components[i]=mapping[root(int(i))]
        sizes=np.zeros(max(1,len(mapping)))
        np.add.at(sizes,components[vertices],mesh.vertex_area.numpy()[vertices])
        self.component=wp.array(components,dtype=int,device=self.device)
        self.component_area=wp.array(sizes,dtype=wp.float64,device=self.device)
        self.group_stride=len(self.areas)
        self.areas=wp.array(np.tile(self.areas.numpy(),len(sizes)),dtype=wp.float64,device=self.device)
        for name in ('maximum','total','norm','energy','base_energy','rank','product'):
            setattr(self,name,wp.zeros(len(self.areas),dtype=wp.float64,device=self.device))
        self.target=wp.zeros(self.capacity,dtype=wp.vec2d,device=self.device)
        self.measure=wp.zeros(len(self.areas),dtype=wp.float64,device=self.device)

    @property
    def temperature(self):
        return .25*self.mesh.stiffness*self.mesh.dhat

    def build(self,x):
        m=self.mesh;b=m.batch;tau=self.temperature
        self.maximum.zero_();self.total.zero_();self.measure.zero_()
        self.launch(position_kernel(unit_terms,x),self.capacity,[x,b.ids,b.kinds,b.eps,b.count,m.edges,self.component,self.group_stride,
            m._node_count,m.dhat,m.dmin,m.stiffness,self.source,self.unit_energy,self.unit_gradient,self.maximum,m.invalid])
        self.launch(weighted_sums,self.capacity,[self.reference,b.ids,self.source,self.unit_energy,b.count,
            self.areas,self.component,self.component_area,self.group_stride,self.maximum,tau,self.target,self.total,self.measure])
        self.launch(source_values,len(self.areas),[self.maximum,self.total,self.measure,self.areas,tau,
            self.norm,self.energy,self.rank,m.invalid])
        self.launch(coefficients,self.capacity,[self.source,self.unit_energy,self.target,b.count,self.areas,
            self.maximum,self.norm,tau,self.coeff,b.areas])

    def add_hessian(self,mass):
        b=self.mesh.batch
        self.launch(positive_hessian,self.capacity,[self.unit_gradient,self.coeff,b.count,self.temperature,b.ids,mass,b.h])

    def correct_energy(self,stats):
        b=self.mesh.batch
        self.launch(remove_linear_energy,self.capacity,[self.unit_energy,b.areas,b.count,stats])
        self.launch(add_source_energy,len(self.energy),[self.energy,stats])
