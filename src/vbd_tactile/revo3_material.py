"""Material provenance for Revo3 checkpoint continuation (no solver imports)."""
from numbers import Real
import math

BASE_MU_PA = 43006.99300699301
BASE_LAMBDA_PA = 264185.8141858142
DEFAULT_STIFFNESS_SCALE = 2.0


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


def checkpoint_material(report, *, legacy_scale=None, legacy_poisson_ratio=None):
    """Recover the saved material; never reinterpret an archive using defaults.

    Archives without material metadata require an explicit legacy choice.
    Unsupported density/damping or inconsistent elastic constants are rejected.
    The returned kwargs work with Revo3TactileHand before any GPU allocation.
    """
    material = report.get('gel_material')
    model = report.get('gel_material_model', 'stable_neo_hookean')
    if model not in ('stable_neo_hookean', 'log_neo_hookean'):
        raise ValueError('unsupported checkpoint gel_material_model')
    model_args = {} if 'gel_material_model' not in report else {'gel_material_model': model}
    if material is None:
        if legacy_scale is None:
            raise ValueError('checkpoint lacks gel_material; specify --legacy-gel-stiffness-scale '
                             '(1 for the original 123 kPa material) and its Poisson ratio if changed')
        scale = _positive(legacy_scale, 'legacy stiffness scale')
        nu = .43 if legacy_poisson_ratio is None else legacy_poisson_ratio
        if isinstance(nu, bool) or not isinstance(nu, Real) or not math.isfinite(nu) or not 0 <= nu < .5:
            raise ValueError('legacy Poisson ratio must be in [0, 0.5)')
        return dict(gel_stiffness_scale=scale, gel_poisson_ratio=None if nu == .43 else float(nu), **model_args)
    if legacy_scale is not None or legacy_poisson_ratio is not None:
        raise ValueError('legacy material overrides are only allowed when checkpoint metadata is missing')
    if not isinstance(material, dict) or not {'mu_pa', 'lambda_pa'} <= material.keys():
        raise ValueError('checkpoint gel_material requires mu_pa and lambda_pa')
    mu = _positive(material['mu_pa'], 'saved shear modulus')
    lam = material['lambda_pa']
    if isinstance(lam, bool) or not isinstance(lam, Real) or not math.isfinite(lam) or lam < 0:
        raise ValueError('saved lambda must be finite and nonnegative')
    nu = lam / (2 * (lam + mu))
    if not math.isfinite(nu) or not 0 <= nu < .5:
        raise ValueError('saved elastic constants imply an unsupported Poisson ratio')
    scale = mu / BASE_MU_PA
    expected = dict(stiffness_scale=scale, poisson_ratio=nu, young_modulus_pa=2*mu*(1+nu),
                    bulk_modulus_pa=lam+2*mu/3, density_kg_m3=1010., damping=2.)
    for key, value in expected.items():
        if key in material:
            saved = material[key]
            if (isinstance(saved, bool) or not isinstance(saved, Real) or not math.isfinite(saved)
                    or not math.isclose(saved, value, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f'inconsistent or unsupported checkpoint material: {key}')
    # Retain the original constants bit-for-bit for the standard .43 material.
    ratio = None if math.isclose(lam, BASE_LAMBDA_PA*scale, rel_tol=1e-14, abs_tol=0.) else nu
    return dict(gel_stiffness_scale=scale, gel_poisson_ratio=ratio, **model_args)


def verify_checkpoint_material(actual, report, *, actual_model='stable_neo_hookean'):
    """Check what reached the model, not only the arguments selected by the CLI."""
    expected_model = report.get('gel_material_model', 'stable_neo_hookean')
    supported = ('stable_neo_hookean', 'log_neo_hookean')
    if expected_model not in supported or actual_model not in supported:
        raise ValueError('unsupported checkpoint or loaded gel_material_model')
    if actual_model != expected_model:
        raise ValueError('loaded material differs from checkpoint: gel_material_model')
    saved = report.get('gel_material')
    if saved is not None:
        for key in ('mu_pa', 'lambda_pa', 'density_kg_m3', 'damping'):
            if key in saved and not math.isclose(actual[key], saved[key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'loaded material differs from checkpoint: {key}')
