"""Opt-in local area quadrature with a regularized fourth-power norm.

For each source/opposing-component group:
    N = (tau**4 + sum_i (A_target_i / (pi*dhat**2)) * b_i**4)**.25
    E = A_source * (N - tau), tau = .25 * stiffness * dhat.
All areas are material reference areas. Noncontacting geometry has b=0 and
cannot dilute local contact. Splitting equal-potential target measure preserves
its contribution; general curved/underresolved quadrature is NOT exact.

This is an experimental contact law, not the official convergent IPC formula.
It keeps the full cross-contact Hessian. Its regularization makes the norm
smooth at contact birth and its aggregation curvature positive semidefinite.
The Revo3 frozen-surface refinement audit failed (9.435 -> 3.458 N); this
locality repair must NOT be promoted as a mesh-consistent production model.
"""
import math
import warp as wp

from .ipc_contact_aggregation import ContactAggregation, unit_terms
from .ipc_contact_softmax import AreaSoftmax
from .ipc_precision import position_kernel

wp.set_module_options({'enable_backward': False})


@wp.kernel
def weighted_sums(rest:wp.array(dtype=wp.vec3d), ids:wp.array(dtype=wp.vec4i),
        source:wp.array(dtype=wp.vec2i), energies:wp.array(dtype=wp.float64),
        count:wp.array(dtype=int), areas:wp.array(dtype=wp.float64), stride:int,
        maximum:wp.array(dtype=wp.float64), tau:wp.float64, reference_area:wp.float64,
        target:wp.array(dtype=wp.vec2d), total:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if i<count[0] and count[0]<=ids.shape[0]:
        v=ids[i];sources=source[i];e=energies[i];w=wp.vec2d(0.)
        if sources[1]<0:
            w[0]=wp.length(wp.cross(rest[v[2]]-rest[v[1]],rest[v[3]]-rest[v[1]]))/(wp.float64(2.)*reference_area)
        else:
            w=wp.vec2d(wp.float64(4.)*areas[sources[1]%stride]/reference_area,
                       wp.float64(4.)*areas[sources[0]%stride]/reference_area)
        target[i]=w
        if e>wp.float64(0.) and wp.isfinite(e):
            for k in range(2):
                s=sources[k]
                if s>=0:
                    r=e/wp.max(maximum[s],tau);r2=r*r
                    wp.atomic_add(total,s,w[k]*r2*r2)


@wp.kernel
def source_values(maximum:wp.array(dtype=wp.float64), total:wp.array(dtype=wp.float64),
        areas:wp.array(dtype=wp.float64), tau:wp.float64, norm:wp.array(dtype=wp.float64),
        energy:wp.array(dtype=wp.float64), rank:wp.array(dtype=wp.float64), invalid:wp.array(dtype=int)):
    s=wp.tid();scale=wp.max(maximum[s],tau);t=tau/scale;t2=t*t
    r=wp.sqrt(wp.sqrt(t2*t2+total[s]));n=scale*r
    # Rationalize N-tau: near activation direct subtraction loses all digits.
    value=scale*total[s]/((r+t)*(r*r+t2))
    if not wp.isfinite(value) or value<wp.float64(0.) or not wp.isfinite(n) or n<=wp.float64(0.):
        wp.atomic_max(invalid,0,1)
    norm[s]=n;energy[s]=areas[s]*value;rank[s]=wp.float64(0.)
    if areas[s]>wp.float64(0.):rank[s]=wp.float64(3.)/(areas[s]*n)


@wp.kernel
def coefficients(source:wp.array(dtype=wp.vec2i), energies:wp.array(dtype=wp.float64),
        target:wp.array(dtype=wp.vec2d), count:wp.array(dtype=int), areas:wp.array(dtype=wp.float64),
        norm:wp.array(dtype=wp.float64), coeff:wp.array(dtype=wp.vec2d), effective:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if i<count[0] and count[0]<=source.shape[0]:
        c=wp.vec2d(0.);e=energies[i]
        if e>wp.float64(0.) and wp.isfinite(e):
            for k in range(2):
                s=source[i][k]
                if s>=0:
                    r=e/norm[s];c[k]=areas[s]*target[i][k]*r*r*r
        coeff[i]=c;effective[i]=c[0]+c[1]


class LocalAreaNorm(AreaSoftmax):
    """Reuse static source/component mapping and nonhomogeneous energy support."""

    def build(self,x):
        m=self.mesh;b=m.batch;tau=self.temperature
        self.maximum.zero_();self.total.zero_()
        self.launch(position_kernel(unit_terms,x),self.capacity,[x,b.ids,b.kinds,b.eps,b.count,m.edges,self.component,self.group_stride,
            m._node_count,m.dhat,m.dmin,m.stiffness,self.source,self.unit_energy,self.unit_gradient,self.maximum,m.invalid])
        self.launch(weighted_sums,self.capacity,[self.reference,b.ids,self.source,self.unit_energy,b.count,
            self.areas,self.group_stride,self.maximum,tau,math.pi*m.dhat*m.dhat,self.target,self.total])
        self.launch(source_values,len(self.areas),[self.maximum,self.total,self.areas,tau,
            self.norm,self.energy,self.rank,m.invalid])
        self.launch(coefficients,self.capacity,[self.source,self.unit_energy,self.target,b.count,self.areas,
            self.norm,self.coeff,b.areas])

    def add_hessian(self,mass):
        # The positive term is 3*c/b, not softmax's c/tau. The negative group
        # term uses 3/(A_source*N), not 3/E (E includes the regularizing offset).
        ContactAggregation.add_hessian(self,mass)
