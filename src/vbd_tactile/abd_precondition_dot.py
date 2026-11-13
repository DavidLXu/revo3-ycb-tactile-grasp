"""Fuse Schwarz gathers/local solves and prolongation/residual dot products.

Keep Warp 1.16's 512-scalar reduction tree, including the zero-padded final
block and subsequent sum rounds. The cooperative gather keeps its original
warp sums. Neither fusion changes the preconditioner or linear operator.
"""
import warp as wp

wp.set_module_options({'enable_backward': False})


# Each block owns eight gather warps and one 24-vector. All lanes reach the
# barrier, including unused/empty aggregate slots. The inverse is contiguous
# and already symmetrized (or symmetrized with the original expression).
@wp.func_native(r'''
#ifdef __CUDA_ARCH__
__shared__ double rhs[24];
int tid = threadIdx.x, lane = tid % 32, slot = tid / 32;
int key = 8 * block + slot;
double a = 0.0, b = 0.0, c = 0.0;
for (int p = offsets.data[key] + lane; p < offsets.data[key + 1]; p += 32) {
    auto v = x.data[indices.data[p]];
    a += v[0]; b += v[1]; c += v[2];
}
for (int k = 16; k > 0; k /= 2) {
    a += __shfl_down_sync(0xffffffff, a, k);
    b += __shfl_down_sync(0xffffffff, b, k);
    c += __shfl_down_sync(0xffffffff, c, k);
}
if (lane == 0) { rhs[3*slot] = a; rhs[3*slot+1] = b; rhs[3*slot+2] = c; }
__syncthreads();
if (tid < 24) {
    double value = 0.0;
    for (int col = 0; col < 24; ++col) {
        double h = inverse.data[576*block + 24*tid + col];
        if (!cached) h = 0.5 * (h + inverse.data[576*block + 24*col + tid]);
        value += h * rhs[col];
    }
    solution.data[24*block + tid] = value;
}
#endif
return 0;
''')
def restrict_solve_native(x:wp.array(dtype=wp.vec3d),offsets:wp.array(dtype=int),indices:wp.array(dtype=int),
                         inverse:wp.array3d(dtype=wp.float64),solution:wp.array2d(dtype=wp.float64),
                         block:int,cached:bool)->int: ...


@wp.kernel
def restrict_solve(x:wp.array(dtype=wp.vec3d),offsets:wp.array(dtype=int),indices:wp.array(dtype=int),
                   inverse:wp.array3d(dtype=wp.float64),solution:wp.array2d(dtype=wp.float64),cached:bool):
    block,lane=wp.tid()
    unused=restrict_solve_native(x,offsets,indices,inverse,solution,block,cached)


@wp.kernel
def prolong_dots(r: wp.array(dtype=wp.float64), maps: wp.array2d(dtype=int),
                 offsets: wp.array(dtype=int), weights: wp.array(dtype=wp.float64),
                 solution: wp.array2d(dtype=wp.float64), z: wp.array(dtype=wp.float64),
                 partial: wp.array2d(dtype=wp.float64)):
    block, lane = wp.tid()
    scalar = block * 512 + lane
    rr = wp.float64(0.)
    rz = wp.float64(0.)
    if scalar < r.shape[0]:
        node = scalar // 3
        axis = scalar % 3
        value = wp.float64(0.)
        fixed = bool(True)
        for level in range(maps.shape[0]):
            slot = maps[level, node]
            if slot >= 0:
                fixed = False
                value += weights[level] * solution[offsets[level] + slot // 8, 3 * (slot % 8) + axis]
        if fixed:
            value = r[scalar]
        z[scalar] = value
        rr = r[scalar] * r[scalar]
        rz = r[scalar] * value
    # No early return: padded lanes must participate in both reductions.
    wp.tile_store(partial[0], wp.tile_sum(wp.tile(rr)), offset=block)
    wp.tile_store(partial[1], wp.tile_sum(wp.tile(rz)), offset=block)


class FusedPreconditionDot:
    def __init__(self, compact):
        if not wp.get_device(compact.device).is_cuda:
            raise ValueError('fused preconditioner requires CUDA')
        self.compact = compact

    def __call__(self, r, z, dot):
        c = self.compact
        s = c.schwarz
        if dot.batch_offsets is not None or dot.tile_size != 512 or dot.max_column_count < 2:
            raise ValueError('fused preconditioner requires unbatched 512-scalar two-column TiledDot')
        if (r.dtype != wp.vec3d or z.dtype != wp.vec3d or r.shape != (c.size,) or z.shape != r.shape
                or not r.is_contiguous or not z.is_contiguous):
            raise ValueError('fused preconditioner requires matching compact vec3d vectors')
        inverse = s.symmetric_inverse if getattr(s, 'cache_symmetry', False) else getattr(s, 'inverse', None)
        if (getattr(s, 'cooperative_restriction', False) and inverse is not None
                and inverse.is_contiguous and s.solution.is_contiguous):
            wp.launch(restrict_solve, (len(s.factors), 256), [r, s.restriction_offsets,
                c.restriction_plan[2], inverse, s.solution, s.cache_symmetry],
                block_dim=256, device=c.device)
        else:
            c.restrict_and_solve(r)
        blocks = (3 * c.size + 511) // 512
        source, target = dot.partial_sums_a[:2], dot.partial_sums_b[:2]
        wp.launch(prolong_dots, (blocks, 512), [r.view(wp.float64).reshape((-1,)), c.maps, s.offsets,
            s.weights, s.solution, z.view(wp.float64).reshape((-1,)), source], block_dim=512, device=c.device)
        for _ in range(dot.rounds):
            length = blocks
            blocks = (length + 511) // 512
            wp.launch(dot.sum_kernel, (2, blocks, 512), [source[:, :length], target],
                block_dim=512, device=c.device)
            source, target = target, source
