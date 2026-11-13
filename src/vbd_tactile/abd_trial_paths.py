"""Skip surface CCD only when an independent required path check already failed."""
import warp as wp
from .ipc_gpu_control import safe_path
wp.set_module_options({'enable_backward': False})

@wp.kernel
def needs_surface_check(diagnostic: wp.array(dtype=wp.float64), gap: wp.float64,
                        jacobian: wp.float64, enabled: wp.array(dtype=int)):
    enabled[0] = int(safe_path(diagnostic, gap, jacobian))
