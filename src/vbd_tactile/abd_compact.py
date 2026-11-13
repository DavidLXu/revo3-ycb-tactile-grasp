"""Restricted PCG coordinates; full geometry, Hessian and final residual survive.

Only fixed particle slots are eliminated. All twelve affine coordinates remain,
including locked guide coordinates. Static maps are prepared once; current
matrix values, contact tails and vectors never leave the GPU.
"""
import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator
from . import ipc_gpu_multilevel as ml
from .abd_schwarz import cached_local_product,restrict_warp

wp.set_module_options({'enable_backward':False})


@wp.kernel
def gather(source:wp.array(dtype=wp.vec3d),rows:wp.array(dtype=int),target:wp.array(dtype=wp.vec3d)):
    i=wp.tid();target[i]=source[rows[i]]


@wp.kernel
def scatter(source:wp.array(dtype=wp.vec3d),rows:wp.array(dtype=int),target:wp.array(dtype=wp.vec3d)):
    i=wp.tid();target[rows[i]]=source[i]


@wp.kernel
def validate_selection(mass:wp.array(dtype=float),index:wp.array(dtype=int),stats:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if (mass[i]>0.)!=(index[i]>=0):wp.atomic_add(stats,2,wp.float64(1.))


@wp.kernel
def refresh_counts(rows:wp.array(dtype=int),source:wp.array(dtype=int),target:wp.array(dtype=int)):
    i=wp.tid();target[i]=source[rows[i]]


@wp.kernel
def refresh_columns(row_indices:wp.array(dtype=int),index:wp.array(dtype=int),offsets:wp.array(dtype=int),
                    counts:wp.array(dtype=int),columns:wp.array(dtype=int),target:wp.array(dtype=int)):
    p=wp.tid();row=row_indices[p]
    if index[row]>=0 and p<offsets[row]+counts[row]:target[p]=index[columns[p]]


@wp.kernel
def mapped_product(offsets:wp.array(dtype=int),counts:wp.array(dtype=int),columns:wp.array(dtype=int),
            values:wp.array3d(dtype=wp.float64),x:wp.array(dtype=wp.vec3d),y:wp.array(dtype=wp.vec3d),
            alpha:wp.float64,beta:wp.float64,out:wp.array2d(dtype=wp.float64)):
    row,k=wp.tid();value=wp.float64(0.)
    for p in range(offsets[row],offsets[row]+counts[row]):
        col=columns[p]
        if col>=0:
            xx=x[col]
            value+=values[p,k,0]*xx[0]+values[p,k,1]*xx[1]+values[p,k,2]*xx[2]
    out[row,k]=alpha*value+beta*y[row][k]


@wp.kernel
def product(rows:wp.array(dtype=int),index:wp.array(dtype=int),offsets:wp.array(dtype=int),
            counts:wp.array(dtype=int),columns:wp.array(dtype=int),values:wp.array3d(dtype=wp.float64),
            x:wp.array(dtype=wp.vec3d),y:wp.array(dtype=wp.vec3d),alpha:wp.float64,beta:wp.float64,
            out:wp.array2d(dtype=wp.float64)):
    i,k=wp.tid();row=rows[i];value=wp.float64(0.)
    for p in range(offsets[row],offsets[row]+counts[row]):
        col=index[columns[p]]
        if col>=0:
            xx=x[col]
            value+=values[p,k,0]*xx[0]+values[p,k,1]*xx[1]+values[p,k,2]*xx[2]
    out[i,k]=alpha*value+beta*y[i][k]


@wp.kernel
def remaining_product(x:wp.array(dtype=wp.vec3d),index:wp.array(dtype=int),ids:wp.array(dtype=wp.vec4i),
        indices:wp.array(dtype=int),count:wp.array(dtype=int),h:wp.array3d(dtype=wp.float64),
        missing:wp.array(dtype=int),alpha:wp.float64,out:wp.array2d(dtype=wp.float64),lanes:int):
    lane,local=wp.tid()
    for slot in range(lane,count[0],lanes):
        c=indices[slot];nodes=ids[c];a=local//3;row=int(-1)
        if nodes[a]>=0:row=index[nodes[a]]
        if row>=0:
            value=wp.float64(0.)
            for b in range(4):
                if missing[c] & (1<<(4*a+b)):
                    col=int(-1)
                    if nodes[b]>=0:col=index[nodes[b]]
                    if col>=0:
                        for j in range(3):value+=h[c,local,3*b+j]*x[col][j]
            wp.atomic_add(out,row,local%3,alpha*value)


class CompactJointSystem:
    def __init__(self,c,matrix,schwarz,batches):
        self.c=c;self.matrix=matrix;self.schwarz=schwarz;self.batches=batches;self.device=c.device
        mass=c.s.model.particle_mass.numpy()
        rows=np.r_[np.flatnonzero(mass>0),np.arange(c.n,c.nd)].astype(np.int32)
        index=np.full(c.nd,-1,np.int32);index[rows]=np.arange(len(rows))
        self.size=len(rows)
        def arr(v):return wp.array(v,dtype=int,device=self.device)
        self.rows=arr(rows);self.index=arr(index)
        self.offsets=arr(matrix.offsets.numpy()[rows])
        self.counts=wp.zeros(self.size,dtype=int,device=self.device)
        self.columns=wp.empty_like(matrix.packed_columns)
        self.maps=arr(schwarz.host_maps[:,rows])
        keys,offsets,indices=schwarz.restriction_plan
        remapped=index[indices.numpy()]
        if np.any(remapped<0):raise ValueError('Schwarz contains an eliminated fixed coordinate')
        self.restriction_plan=(keys,offsets,arr(remapped))
        self.rhs=wp.zeros(self.size,dtype=wp.vec3d,device=self.device)
        self.direction=wp.zeros_like(self.rhs)
        shape=(3*self.size,3*self.size)
        self.operator=LinearOperator(shape,wp.float64,self.device,self.matvec)
        self.assembled_operator=LinearOperator(shape,wp.float64,self.device,self.assembled_matvec)
        self.preconditioner=LinearOperator(shape,wp.float64,self.device,self.precondition)

    def launch(self,kernel,dim,args):
        if dim and (not isinstance(dim,tuple) or all(dim)):
            wp.launch(kernel,dim,args,device=self.device)

    def assembled_matvec(self,x,y,out,alpha,beta):
        m=self.matrix
        aggregation=self.c.s.energy.mesh.aggregation
        if aggregation is not None:aggregation.prepare_product(x,self.c.mobile_mass,self.c,self.index)
        if getattr(m,'cooperative_product',False):
            from .abd_warp_product import launch
            launch(self.size,self.offsets,self.counts,self.columns,
                m.packed_values.view(wp.float64).reshape((len(m.values),3,3)),x,y,alpha,beta,
                out.view(wp.float64).reshape((self.size,3)),self.device,removed_columns=True)
        else:
            self.launch(mapped_product,(self.size,3),[self.offsets,self.counts,self.columns,
                m.packed_values.view(wp.float64).reshape((len(m.values),3,3)),x,y,alpha,beta,
                out.view(wp.float64).reshape((self.size,3))])
        if aggregation is not None:aggregation.add_product(out,alpha,self.c.mobile_mass,self.c,self.index)

    def refresh(self):
        # Packed column positions can change at EVERY Hessian. Remap once here,
        # not during each PCG multiply, and keep the fresh physical block values.
        m=self.matrix
        self.launch(refresh_counts,self.size,[self.rows,m.counts,self.counts])
        self.launch(refresh_columns,len(m.values),[m.row_indices,self.index,m.offsets,m.counts,m.packed_columns,self.columns])

    def matvec(self,x,y,out,alpha,beta):
        self.assembled_matvec(x,y,out,alpha,beta)
        for b in self.batches:
            missing,indices,count=self.matrix.remaining[b];lanes=min(b.capacity,128)
            self.launch(remaining_product,(lanes,12),[x,self.index,b.ids,indices,count,
                b.h.view(wp.float64).reshape((b.capacity,12,12)),missing,alpha,
                out.view(wp.float64).reshape((self.size,3)),lanes])

    def restrict_and_solve(self,x):
        s=self.schwarz
        if s.cooperative_restriction:
            wp.launch(restrict_warp,(len(s.restriction_offsets)-1,32),
                [x,s.restriction_offsets,self.restriction_plan[2],s.rhs],block_dim=32,device=self.device)
        else:
            s.rhs.zero_()
            self.launch(ml.restrict,len(self.restriction_plan[0]),[x,*self.restriction_plan,s.rhs])
        self.launch(cached_local_product if s.cache_symmetry else ml.local_product,(len(s.factors),24),
                    [s.symmetric_inverse if s.cache_symmetry else s.inverse,s.rhs,s.solution])

    def precondition(self,x,y,out,alpha,beta):
        self.restrict_and_solve(x)
        s=self.schwarz
        self.launch(ml.prolong,self.size,[x,y,self.maps,s.offsets,s.weights,s.solution,alpha,beta,out])

    def begin(self):
        self.launch(gather,self.size,[self.c.rhs,self.rows,self.rhs]);self.direction.zero_()

    def finish(self):
        # The caller clears the full direction, so removed coordinates stay 0.
        self.launch(scatter,self.size,[self.direction,self.rows,self.c.direction])
