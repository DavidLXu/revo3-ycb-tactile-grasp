"""Preallocated integer prefix scan for ordered sparse-block compaction.

All levels run on the owning device. There is no allocation, readback, floating
point reduction or atomic reservation during a captured Newton iteration.
"""
import warp as wp

wp.set_module_options({'enable_backward': False})
TILE = wp.constant(128)


@wp.kernel
def scan_tiles(source: wp.array(dtype=int), prefix: wp.array(dtype=int),
               totals: wp.array(dtype=int)):
    block, lane = wp.tid()
    values = wp.tile_load(source, shape=TILE, offset=block*TILE)
    result = wp.tile_scan_inclusive(values)
    wp.tile_store(prefix, result, offset=block*TILE)
    total = wp.tile_extract(result, TILE-1)
    if lane == 0:
        totals[block] = total


@wp.kernel
def add_offsets(prefix: wp.array(dtype=int), parent: wp.array(dtype=int)):
    i = wp.tid(); block = i//TILE
    if block > 0:
        prefix[i] += parent[block-1]


@wp.kernel
def row_counts(offsets: wp.array(dtype=int), prefix: wp.array(dtype=int),
               counts: wp.array(dtype=int)):
    row = wp.tid(); start = offsets[row]; end = offsets[row+1]
    count = int(0)
    if end > start:
        count = prefix[end-1]
        if start > 0:
            count -= prefix[start-1]
    counts[row] = count


@wp.kernel
def scatter(rows: wp.array(dtype=int), offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int), values: wp.array(dtype=wp.mat33d),
        used: wp.array(dtype=int), prefix: wp.array(dtype=int),
        out_columns: wp.array(dtype=int), out_values: wp.array(dtype=wp.mat33d)):
    i = wp.tid()
    if used[i] != 0:
        start = offsets[rows[i]]; before = int(0)
        if start > 0:
            before = prefix[start-1]
        target = start+prefix[i]-before-1
        out_columns[target] = columns[i]
        out_values[target] = values[i]


class IntegerScan:
    def __init__(self, size, device):
        self.device = wp.get_device(device)
        self.prefixes = []; self.totals = []
        while size > 0:
            groups = (size+127)//128
            self.prefixes.append(wp.empty(size, dtype=int, device=self.device))
            self.totals.append(wp.empty(groups, dtype=int, device=self.device))
            if groups == 1:
                break
            size = groups

    def run(self, values):
        source = values
        for prefix, totals in zip(self.prefixes, self.totals):
            wp.launch_tiled(scan_tiles, dim=len(totals), inputs=[source, prefix, totals],
                            block_dim=32, device=self.device)
            source = totals
        for level in range(len(self.prefixes)-2, -1, -1):
            wp.launch(add_offsets, dim=len(self.prefixes[level]),
                      inputs=[self.prefixes[level], self.prefixes[level+1]], device=self.device)
        return self.prefixes[0] if self.prefixes else values
