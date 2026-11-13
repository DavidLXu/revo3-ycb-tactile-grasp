"""GPU assembly of a joint affine/FEM operator, reused by every PCG iteration.

The static pattern contains FEM adjacency and all affine coordinate couplings.
Nonlocal FEM/FEM contact blocks absent from that pattern remain matrix-free.
No contact, row, small coefficient or physical acceptance check is discarded.
Only immutable index maps are prepared on the host at construction.
"""
import numpy as np
import warp as wp
from .ipc_gpu_math import mat12
from . import abd_block_rows as abr
from . import abd_block_scan as absn
from . import abd_guides as ag

wp.set_module_options({'enable_backward':False})


@wp.kernel
def no_remaining_contacts(normal:wp.array(dtype=int),friction:wp.array(dtype=int),empty:wp.array(dtype=int)):
    # Counts are rebuilt by every assembly and stay immutable during PCG.
    empty[0]=int(normal[0]==0 and friction[0]==0)


@wp.func
def slot_of(row:int,col:int,offsets:wp.array(dtype=int),columns:wp.array(dtype=int)):
    lo=offsets[row]; hi=offsets[row+1]
    while lo<hi:
        mid=(lo+hi)//2
        if columns[mid]<col:lo=mid+1
        else:hi=mid
    found=int(-1)
    if lo<offsets[row+1]:
        if columns[lo]==col:found=lo
    return found


@wp.kernel
def copy_base(source:wp.array(dtype=wp.mat33d),slots:wp.array(dtype=int),out:wp.array(dtype=wp.mat33d)):
    i=wp.tid(); out[slots[i]]=source[i]


@wp.kernel
def copy_bodies(h:wp.array(dtype=mat12),slots:wp.array3d(dtype=int),out:wp.array(dtype=wp.mat33d)):
    b,i,j=wp.tid(); value=wp.mat33d(0.)
    for k in range(3):
        for l in range(3):value[k,l]=h[b][3*i+k,3*j+l]
    out[slots[b,i,j]]=value


@wp.func
def coordinate(node:int,column:int,owner:wp.array(dtype=int),n:int):
    index=node
    if owner[node]>=0:index=n+4*owner[node]+column
    return index


@wp.func
def coefficient(node:int,column:int,owner:wp.array(dtype=int),weights:wp.array(dtype=wp.vec4d)):
    value=wp.float64(1.)
    if owner[node]>=0:value=weights[node][column]
    return value


@wp.kernel
def assemble_contact(ids:wp.array(dtype=wp.vec4i),active:wp.array(dtype=int),count:wp.array(dtype=int),
        h:wp.array3d(dtype=wp.float64),owner:wp.array(dtype=int),weights:wp.array(dtype=wp.vec4d),
        mass:wp.array(dtype=float),n:int,offsets:wp.array(dtype=int),columns:wp.array(dtype=int),
        values:wp.array(dtype=wp.mat33d),missing:wp.array(dtype=int),lanes:int):
    lane,a,b=wp.tid()
    for index in range(lane,count[0],lanes):
        contact=active[index]; nodes=ids[contact]; row=nodes[a]; col=nodes[b]
        if row>=0 and col>=0:
            if (owner[row]>=0 or mass[row]>0.) and (owner[col]>=0 or mass[col]>0.):
                nr=int(1); nc=int(1)
                if owner[row]>=0:nr=4
                if owner[col]>=0:nc=4
                block=wp.mat33d(0.)
                nonzero=bool(False)
                for k in range(3):
                    for l in range(3):
                        block[k,l]=h[contact,3*a+k,3*b+l]
                        nonzero=nonzero or block[k,l]!=wp.float64(0.)
                # Inactive closest-feature vertices have exactly zero blocks.
                # Do not let these create a false matrix-free remainder that
                # disables the assembled-only PCG path. No magnitude cutoff.
                if nonzero:
                    for i in range(nr):
                        rr=coordinate(row,i,owner,n); rw=coefficient(row,i,owner,weights)
                        for j in range(nc):
                            cc=coordinate(col,j,owner,n); cw=coefficient(col,j,owner,weights)
                            slot=slot_of(rr,cc,offsets,columns)
                            if slot>=0:wp.atomic_add(values,slot,(rw*cw)*block)
                            else:wp.atomic_or(missing,contact,1<<(4*a+b))


@wp.kernel
def compact_missing(active:wp.array(dtype=int),count:wp.array(dtype=int),missing:wp.array(dtype=int),
                    indices:wp.array(dtype=int),remaining:wp.array(dtype=int)):
    i=wp.tid()
    if i<count[0]:
        contact=active[i]
        if missing[contact]!=0:
            slot=wp.atomic_add(remaining,0,1); indices[slot]=contact


@wp.kernel
def pack(offsets:wp.array(dtype=int),columns:wp.array(dtype=int),values:wp.array(dtype=wp.mat33d),
         counts:wp.array(dtype=int),packed_columns:wp.array(dtype=int),packed_values:wp.array(dtype=wp.mat33d)):
    row=wp.tid(); start=offsets[row]; size=int(0)
    for p in range(start,offsets[row+1]):
        block=values[p]; used=columns[p]==row
        for i in range(3):
            for j in range(3):used=used or block[i,j]!=wp.float64(0.)
        if used:
            packed_columns[start+size]=columns[p]; packed_values[start+size]=block; size+=1
    counts[row]=size


@wp.kernel
def product(offsets:wp.array(dtype=int),counts:wp.array(dtype=int),columns:wp.array(dtype=int),
            values:wp.array3d(dtype=wp.float64),x:wp.array(dtype=wp.vec3d),y:wp.array(dtype=wp.vec3d),
            alpha:wp.float64,beta:wp.float64,out:wp.array2d(dtype=wp.float64)):
    row,k=wp.tid(); value=wp.float64(0.)
    for p in range(offsets[row],offsets[row]+counts[row]):
        xx=x[columns[p]]
        value+=values[p,k,0]*xx[0]+values[p,k,1]*xx[1]+values[p,k,2]*xx[2]
    out[row,k]=alpha*value+beta*y[row][k]


@wp.kernel
def remaining_product(x:wp.array(dtype=wp.vec3d),ids:wp.array(dtype=wp.vec4i),
        indices:wp.array(dtype=int),count:wp.array(dtype=int),h:wp.array3d(dtype=wp.float64),
        missing:wp.array(dtype=int),alpha:wp.float64,out:wp.array2d(dtype=wp.float64),lanes:int):
    lane,local=wp.tid()
    for index in range(lane,count[0],lanes):
        c=indices[index]; nodes=ids[c]; a=local//3; row=nodes[a]; value=wp.float64(0.)
        for b in range(4):
            if missing[c] & (1<<(4*a+b)):
                col=nodes[b]
                for j in range(3):value+=h[c,local,3*b+j]*x[col][j]
        if row>=0:wp.atomic_add(out,row,local%3,alpha*value)


@wp.kernel
def precondition(x:wp.array(dtype=wp.vec3d),y:wp.array(dtype=wp.vec3d),soft:wp.array(dtype=wp.mat33d),
        body:wp.array(dtype=mat12),mass:wp.array(dtype=float),n:int,alpha:wp.float64,beta:wp.float64,
        out:wp.array(dtype=wp.vec3d)):
    i=wp.tid(); value=wp.vec3d(0.)
    if i<n:
        value=x[i]
        if mass[i]>0.:value=soft[i]*x[i]
    else:
        b=(i-n)//4; row=(i-n)%4
        for k in range(3):
            for col in range(4):
                for j in range(3):value[k]+=body[b][3*row+k,3*col+j]*x[n+4*b+col][j]
    out[i]=alpha*value+beta*y[i]


class JointMatrix:
    def __init__(self,n,nb,base,device,*,surface_faces=None):
        self.n=n; self.nb=nb; self.nd=n+4*nb; self.device=wp.get_device(device)
        old_offsets=base.offsets.numpy(); old_columns=base.columns.numpy()
        rows=np.repeat(np.arange(n,dtype=np.int64),np.diff(old_offsets))
        base_keys=rows*self.nd+old_columns
        affine=np.arange(n,self.nd,dtype=np.int64); all_rows=np.arange(self.nd,dtype=np.int64)
        surface_keys=np.empty(0,np.int64)
        if surface_faces is not None:
            faces=np.asarray(surface_faces.numpy(),dtype=np.int64)
            if faces.ndim!=2 or faces.shape[1]!=3 or np.any(faces<0) or np.any(faces>=n):
                raise ValueError('invalid surface triangle topology for joint matrix')
            # Surface triangles need not be faces of the FEM tetrahedra.
            surface_keys=(faces[:,:,None]*self.nd+faces[:,None,:]).ravel()
        keys=np.unique(np.concatenate([base_keys,surface_keys,(all_rows[:,None]*self.nd+affine).ravel(),
                                       (affine[:,None]*self.nd+all_rows).ravel()]))
        if len(keys)>np.iinfo(np.int32).max:raise ValueError('joint matrix exceeds int32 storage')
        offsets=np.r_[0,np.cumsum(np.bincount(keys//self.nd,minlength=self.nd))].astype(np.int32)
        def array(value,dtype):return wp.array(value,dtype=dtype,device=self.device)
        self.offsets=array(offsets,int); self.columns=array((keys%self.nd).astype(np.int32),int)
        self.values=wp.zeros(len(keys),dtype=wp.mat33d,device=self.device)
        self.row_indices=array((keys//self.nd).astype(np.int32),int)
        self.used=wp.zeros(len(keys),dtype=int,device=self.device)
        self.block_scan=absn.IntegerScan(len(keys),self.device)
        self.base_slots=array(np.searchsorted(keys,base_keys).astype(np.int32),int)
        body_rows=affine.reshape(nb,4)
        self.body_slots=array(np.searchsorted(keys,body_rows[:,:,None]*self.nd+body_rows[:,None,:]).astype(np.int32),int)
        self.counts=wp.zeros(self.nd,dtype=int,device=self.device)
        self.packed_columns=wp.empty_like(self.columns); self.packed_values=wp.empty_like(self.values)
        self.remaining={}
        self.surface_error=wp.zeros(1,dtype=int,device=self.device)

    def prepare(self,batches):
        # Called outside capture; retain the batch objects alongside buffers.
        # Contact capacity growth rebuilds only scratch for the changed batch.
        self.remaining={b:self.remaining.get(b) or (
            wp.zeros(b.capacity,dtype=int,device=self.device),
            wp.empty(b.capacity,dtype=int,device=self.device),
            wp.zeros(1,dtype=int,device=self.device)) for b in batches}

    def assemble(self,base,c,batches):
        def launch(kernel,dim,args):wp.launch(kernel,dim,args,device=self.device)
        self.values.zero_();self.surface_error.zero_()
        launch(copy_base,len(base.values),[base.values,self.base_slots,self.values])
        if getattr(c,'rowwise_blocks',False):
            launch(abr.copy_bodies,(self.nb,4,4),[c.body_h.view(wp.float64).reshape((self.nb,12,12)),self.body_slots,self.values])
        else:
            launch(copy_bodies,(self.nb,4,4),[c.body_h,self.body_slots,self.values])
        for b in batches:
            missing,indices,count=self.remaining[b]; missing.zero_(); count.zero_()
            lanes=min(b.capacity,128)
            launch(assemble_contact,(lanes,4,4),[b.ids,b.active_indices,b.active_count,
                b.h.view(wp.float64).reshape((b.capacity,12,12)),c.owner,c.weights,c.s.model.particle_mass,
                self.n,self.offsets,self.columns,self.values,missing,lanes])
            launch(compact_missing,b.capacity,[b.active_indices,b.active_count,missing,indices,count])
        surface=getattr(getattr(c.s,'energy',None),'surface_contact',None)
        if surface is not None:
            from .abd_surface_operator import add_surface
            add_surface(self,surface,c)
        if getattr(c,'has_guides',False) and c.reduced_guides:
            # Every affine row/column is in this static pattern. Missing
            # contact blocks therefore involve FEM coordinates only, on
            # which the guide projector is the identity.
            launch(ag.project_blocks,len(self.values),[self.row_indices,self.columns,self.values,c.guide_axes,self.n])
        if getattr(c,'rowwise_blocks',False):
            # Ordered integer compaction: retain every nonzero block without
            # serially traversing the long affine-coordinate rows.
            launch(abr.mark_blocks,len(self.values),[self.row_indices,self.columns,
                self.values.view(wp.float64).reshape((len(self.values),9)),self.used])
            prefix=self.block_scan.run(self.used)
            launch(absn.row_counts,self.nd,[self.offsets,prefix,self.counts])
            launch(absn.scatter,len(self.values),[self.row_indices,self.offsets,self.columns,self.values,
                self.used,prefix,self.packed_columns,self.packed_values])
        else:
            launch(pack,self.nd,[self.offsets,self.columns,self.values,self.counts,self.packed_columns,self.packed_values])

    def assembled_matvec(self,x,y,out,alpha,beta):
        result=out.view(wp.float64).reshape((self.nd,3))
        if getattr(self,'cooperative_product',False):
            from .abd_warp_product import launch
            launch(self.nd,self.offsets,self.counts,self.packed_columns,
                self.packed_values.view(wp.float64).reshape((len(self.values),3,3)),x,y,alpha,beta,result,self.device)
            return
        wp.launch(product,(self.nd,3),[self.offsets,self.counts,self.packed_columns,
            self.packed_values.view(wp.float64).reshape((len(self.values),3,3)),x,y,alpha,beta,result],device=self.device)

    def matvec(self,x,y,out,alpha,beta,batches):
        self.assembled_matvec(x,y,out,alpha,beta)
        result=out.view(wp.float64).reshape((self.nd,3))
        for b in batches:
            missing,indices,count=self.remaining[b]; lanes=min(b.capacity,128)
            wp.launch(remaining_product,(lanes,12),[x,b.ids,indices,count,
                b.h.view(wp.float64).reshape((b.capacity,12,12)),missing,alpha,result,lanes],device=self.device)
