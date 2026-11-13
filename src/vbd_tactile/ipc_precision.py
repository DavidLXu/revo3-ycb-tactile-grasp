"""Explicit Warp specializations preserve float64 scalar launch arguments.

Launching a generic Warp kernel with Python floats otherwise infers float32
scalars, even when its non-generic scalar parameters require float64.
"""

from typing import Any
import warp as wp

_double_kernels = {}
_pose_kernels = {}
_device_scalar_kernels = {}


@wp.func
def scalar_value(value: wp.float32):
    return value


@wp.func
def scalar_value(value: wp.float64):
    return value


@wp.func
def scalar_value(value: wp.array(dtype=wp.float64)):
    return value[0]


@wp.func
def pose_value(value: wp.vec3d):
    return value


@wp.func
def pose_value(value: wp.array(dtype=wp.vec3d)):
    return value[0]


def position_overloads(*names, poses=(), scalars=None):
    def specialize(kernel):
        scalar_poses = {**{n: wp.vec3d for n in poses}, **(scalars or {})}
        single = wp.overload(kernel, {**{n: wp.array(dtype=wp.vec3) for n in names}, **scalar_poses})
        double = wp.overload(kernel, {**{n: wp.array(dtype=wp.vec3d) for n in names}, **scalar_poses})
        _double_kernels[single] = double
        if poses:
            array_poses = {n: wp.array(dtype=wp.vec3d) for n in poses}
            for original, dtype in ((single, wp.vec3), (double, wp.vec3d)):
                _pose_kernels[original] = wp.overload(
                    kernel, {**{n: wp.array(dtype=dtype) for n in names}, **array_poses})
        if scalars:
            device_scalars = {n: wp.array(dtype=wp.float64) for n in scalars}
            for original,dtype in ((single,wp.vec3),(double,wp.vec3d)):
                _device_scalar_kernels[original]=wp.overload(kernel,
                    {**{n: wp.array(dtype=dtype) for n in names},**scalar_poses,**device_scalars})
        return single

    return specialize


def position_kernel(kernel, positions):
    if positions.dtype == wp.vec3d:
        return _double_kernels.get(kernel, kernel)
    return kernel


def pose_kernel(kernel):
    """Use device-resident sphere inputs while retaining identical arithmetic."""
    return _pose_kernels.get(kernel, kernel)


def device_scalar_kernel(kernel):
    return _device_scalar_kernels.get(kernel,kernel)


@wp.kernel
def _convert(source: wp.array(dtype=Any), target: wp.array(dtype=Any)):
    i = wp.tid()
    target[i] = type(target[i])(source[i])


_converters = {
    (src, dst): wp.overload(
        _convert, {"source": wp.array(dtype=src), "target": wp.array(dtype=dst)}
    )
    for src, dst in ((wp.vec3, wp.vec3d), (wp.vec3d, wp.vec3))
}


def copy_positions(source, target):
    if source.dtype == target.dtype:
        wp.copy(target, source)
    else:
        wp.launch(
            _converters[source.dtype, target.dtype],
            source.size,
            inputs=[source, target],
            device=source.device,
        )
