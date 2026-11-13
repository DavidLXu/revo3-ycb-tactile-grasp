"""Constraint-aware normal Hessians; the objective and contact set are unchanged.

The reduced projection removes only the three exact common-translation null
directions. It projects before imposing fixed-node constraints, as psd12 does.
The opt-in constrained projection instead eliminates fixed DOFs first. This
changes the Newton metric, while keeping the objective and acceptance tests.
"""
from typing import Any
import warp as wp

from .ipc_precision import position_overloads, pose_value
from .ipc_gpu_kernels import point
from .ipc_gpu_math import mat9, mat12, contact_terms, squared_distance, psd9, psd12
from .ipc_gpu_material import eigen3

wp.set_module_options({"enable_backward": False})
mat9x12 = wp.types.matrix(shape=(9, 12), dtype=wp.float64)


@wp.func
def relative_basis(vertex: int, column: int):
    # First three columns of a Hadamard matrix, excluding its constant column.
    result = wp.float64(-0.5)
    if vertex == 0 or vertex == column + 1:
        result = wp.float64(0.5)
    return result


@wp.func
def psd_relative(h: mat12):
    """T psd(T^T H T) T^T, with orthonormal T = Q (Kronecker) I3."""
    left = mat9x12(0.)
    for i in range(9):
        for j in range(12):
            for a in range(4):
                left[i, j] += relative_basis(a, i // 3) * h[3*a + i % 3, j]
    reduced = mat9(0.)
    for i in range(9):
        for j in range(9):
            for b in range(4):
                reduced[i, j] += left[i, 3*b + j % 3] * relative_basis(b, j // 3)
    projected = psd9(reduced)
    for i in range(9):
        for j in range(12):
            left[i, j] = wp.float64(0.)
            for b in range(3):
                left[i, j] += projected[i, 3*b + j % 3] * relative_basis(j // 3, b)
    result = mat12(0.)
    for i in range(12):
        for j in range(12):
            for a in range(3):
                result[i, j] += relative_basis(i // 3, a) * left[3*a + i % 3, j]
    return result


@wp.func
def has_free_node(ind: wp.vec4i, mass: wp.array(dtype=float)):
    result = False
    for a in range(4):
        if ind[a] >= 0:
            result = result or mass[ind[a]] > 0.
    return result


@wp.func
def psd_constrained(h: mat12, ind: wp.vec4i, mass: wp.array(dtype=float)):
    """Project the free principal block, then lift; a different Newton metric.

    Fixed coordinates are eliminated before PSD projection. The energy and
    derivatives used for convergence remain unchanged. Do not use these blocks
    for a rigid Schur tangent: its fixed/free cross derivatives are absent.
    """
    slots=wp.vec4i(-1)
    count=int(0)
    for a in range(4):
        if ind[a]>=0:
            if mass[ind[a]]>0.:
                slots[count]=a
                count+=1
    result=mat12(0.)
    if count==4:
        result=psd12(h)
    elif count==1:
        block=wp.mat33d(0.)
        for i in range(3):
            for j in range(3):block[i,j]=h[3*slots[0]+i,3*slots[0]+j]
        values,vectors=eigen3(block)
        for i in range(3):
            for j in range(3):
                value=wp.float64(0.)
                for k in range(3):value+=wp.max(values[k],wp.float64(0.))*vectors[i,k]*vectors[j,k]
                result[3*slots[0]+i,3*slots[0]+j]=value
    elif count>1:
        block9=mat9(0.)
        for i in range(3*count):
            for j in range(3*count):block9[i,j]=h[3*slots[i//3]+i%3,3*slots[j//3]+j%3]
        projected=psd9(block9)
        for i in range(3*count):
            for j in range(3*count):result[3*slots[i//3]+i%3,3*slots[j//3]+j%3]=projected[i,j]
    return result


@position_overloads("x", poses=("center",))
@wp.kernel
def normals_constrained(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
                 kinds: wp.array(dtype=int), areas: wp.array(dtype=wp.float64),
                 moll_eps: wp.array(dtype=wp.float64), count: wp.array(dtype=int),
                 center: Any, dhat: wp.float64, dmin: wp.float64,
                 stiffness: wp.float64, offset: wp.float64, hessians: wp.array(dtype=mat12),
                 mass: wp.array(dtype=float)):
    i=wp.tid()
    if i<count[0]:
        ind=ids[i]
        h=mat12(0.)
        if has_free_node(ind,mass):
            c0=pose_value(center)
            a,b=point(x,ind[0],c0),point(x,ind[1],c0)
            c,d=point(x,ind[2],c0),point(x,ind[3],c0)
            q,gq,hq,w=squared_distance(a,b,c,d,kinds[i],False)
            if wp.sqrt(q)<offset+dmin+dhat:
                e,g,h,w=contact_terms(a,b,c,d,kinds[i],dhat,dmin,stiffness,areas[i],offset,moll_eps[i],True)
                h=psd_constrained(h,ind,mass)
        hessians[i]=h


@position_overloads("x", poses=("center",))
@wp.kernel
def normals_free(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
                 kinds: wp.array(dtype=int), areas: wp.array(dtype=wp.float64),
                 moll_eps: wp.array(dtype=wp.float64), count: wp.array(dtype=int),
                 center: Any, dhat: wp.float64, dmin: wp.float64,
                 stiffness: wp.float64, offset: wp.float64, hessians: wp.array(dtype=mat12),
                 mass: wp.array(dtype=float)):
    i = wp.tid()
    if i < count[0]:
        ind = ids[i]
        h = mat12(0.)
        if has_free_node(ind, mass):
            c0 = pose_value(center)
            a, b = point(x, ind[0], c0), point(x, ind[1], c0)
            c, d = point(x, ind[2], c0), point(x, ind[3], c0)
            q, gq, hq, w = squared_distance(a, b, c, d, kinds[i], False)
            if wp.sqrt(q) < offset+dmin+dhat:
                e, g, h, w = contact_terms(a, b, c, d, kinds[i], dhat, dmin,
                                          stiffness, areas[i], offset, moll_eps[i], True)
                h = psd12(h)
        # Explicitly clear previously active stencils when they become fixed.
        hessians[i] = h


@position_overloads("x", poses=("center",))
@wp.kernel
def normals_reduced(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
                    kinds: wp.array(dtype=int), areas: wp.array(dtype=wp.float64),
                    moll_eps: wp.array(dtype=wp.float64), count: wp.array(dtype=int),
                    center: Any, dhat: wp.float64, dmin: wp.float64,
                    stiffness: wp.float64, offset: wp.float64, hessians: wp.array(dtype=mat12),
                    mass: wp.array(dtype=float)):
    i = wp.tid()
    if i < count[0]:
        ind = ids[i]
        h = mat12(0.)
        if has_free_node(ind, mass):
            c0 = pose_value(center)
            a, b = point(x, ind[0], c0), point(x, ind[1], c0)
            c, d = point(x, ind[2], c0), point(x, ind[3], c0)
            q, gq, hq, w = squared_distance(a, b, c, d, kinds[i], False)
            if wp.sqrt(q) < offset+dmin+dhat:
                e, g, h, w = contact_terms(a, b, c, d, kinds[i], dhat, dmin,
                                          stiffness, areas[i], offset, moll_eps[i], True)
                h = psd_relative(h)
        hessians[i] = h
