"""One CUDA warp cooperates on each scalar sparse row; no contact truncation."""
import warp as wp
from functools import lru_cache
wp.set_module_options({'enable_backward':False})


@wp.func_native('''
#ifdef __CUDA_ARCH__
for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_down_sync(0xffffffff, value, offset);
#endif
return value;
''')
def sum_warp(value:wp.float64)->wp.float64: ...


@wp.kernel
def product(offsets:wp.array(dtype=int),counts:wp.array(dtype=int),columns:wp.array(dtype=int),
            values:wp.array3d(dtype=wp.float64),x:wp.array(dtype=wp.vec3d),y:wp.array(dtype=wp.vec3d),
            alpha:wp.float64,beta:wp.float64,out:wp.array2d(dtype=wp.float64),removed_columns:bool):
    # Last launch dimension MUST be 32 and block_dim MUST be 32. Every lane
    # reaches the shuffle even for empty rows or a short final chunk.
    row,k,lane=wp.tid();value=wp.float64(0.)
    for p in range(offsets[row]+lane,offsets[row]+counts[row],32):
        col=columns[p]
        if not removed_columns or col>=0:
            xx=x[col]
            value+=values[p,k,0]*xx[0]+values[p,k,1]*xx[1]+values[p,k,2]*xx[2]
    value=sum_warp(value)
    if lane==0:out[row,k]=alpha*value+beta*y[row][k]


def launch(n,offsets,counts,columns,values,x,y,alpha,beta,out,device,*,removed_columns=False):
    if not wp.get_device(device).is_cuda:raise ValueError('cooperative sparse product requires CUDA')
    wp.launch(product,(n,3,32),[offsets,counts,columns,values,x,y,alpha,beta,out,removed_columns],
              block_dim=32,device=device)


@wp.kernel
def product_dot_terms(offsets:wp.array(dtype=int),counts:wp.array(dtype=int),columns:wp.array(dtype=int),
            values:wp.array3d(dtype=wp.float64),x:wp.array(dtype=wp.vec3d),
            out:wp.array2d(dtype=wp.float64),terms:wp.array(dtype=wp.float64)):
    # Same warp reduction as product; the scalar dot term is written while
    # the result is still in a register. No atomic/global unordered dot sum.
    row,k,lane=wp.tid();value=wp.float64(0.)
    for p in range(offsets[row]+lane,offsets[row]+counts[row],32):
        col=columns[p]
        if col>=0:
            xx=x[col]
            value+=values[p,k,0]*xx[0]+values[p,k,1]*xx[1]+values[p,k,2]*xx[2]
    value=sum_warp(value)
    if lane==0:
        out[row,k]=value
        terms[3*row+k]=x[row][k]*value


@lru_cache(None)
def sum_kernel(tile_size):
    @wp.kernel(module='unique')
    def reduce(source:wp.array(dtype=wp.float64),target:wp.array(dtype=wp.float64)):
        block,lane=wp.tid()
        data=wp.tile_load(source,shape=tile_size,offset=block*tile_size)
        total=wp.tile_sum(data)
        wp.tile_store(target,total,offset=block)
    return reduce


class FusedProductDot:
    """Unbatched compact assembled Ap and p.Ap; no contact tails allowed.

    Uses a bounded deterministic tree. The usual fingertip's 2811 scalars
    reduce in one block; larger systems use preallocated additional levels.
    The final linear residual is still checked using the complete operator.
    """
    def __init__(self,compact):
        self.compact=compact
        length=3*compact.size
        self.terms=wp.empty(length,dtype=wp.float64,device=compact.device)
        self.levels=[]
        while True:
            tile=min(4096,1 << (max(length,2)-1).bit_length())
            blocks=(length+tile-1)//tile
            output=wp.empty(blocks,dtype=wp.float64,device=compact.device)
            self.levels.append((sum_kernel(tile),blocks,output))
            if blocks==1:break
            length=blocks

    def __call__(self,x,out,dot):
        c=self.compact;m=c.matrix
        wp.launch(product_dot_terms,(c.size,3,32),[c.offsets,c.counts,c.columns,
            m.packed_values.view(wp.float64).reshape((len(m.values),3,3)),x,
            out.view(wp.float64).reshape((c.size,3)),self.terms],block_dim=32,device=c.device)
        source=self.terms
        for i,(kernel,blocks,scratch) in enumerate(self.levels):
            target=dot if i==len(self.levels)-1 else scratch
            wp.launch(kernel,(blocks,512),[source,target],block_dim=512,device=c.device)
            source=target
