"""Opt-in Taccel-style PT/EE contact in our Newton solver, not native Taccel.

Normal energy already matches the released Taccel cb23bc2 generated CUDA:
  A * dhat * [-kappa * ((q-xi^2-H)/H)^2 * log((q-xi^2)/H)],
  H=dhat*(dhat+2*xi), A=vertex_area/4 or (edge_area_i+edge_area_j)/4.
EE uses the reference-edge mollifier. Here xi is our positive minimum gap.
Native detection enumerates BOTH ordered EE pairs. We store one unordered
pair with twice its weight, preserving energy/force without duplicate work.
Taccel's frozen friction load uses sqrt(q-xi^2), rather than sqrt(q).
Its friction potential differs by a frozen additive constant only.

This option keeps our CCD, FEM, PSD projection, solver and minimum-gap policy.
It does not add convergent-IPC inclusion/exclusion or surface quadrature.
"""
from typing import Any
import numpy as np
import warp as wp
from .ipc_precision import position_overloads
from .ipc_gpu_math import squared_distance

wp.set_module_options({'enable_backward': False})


@position_overloads('x')
@wp.kernel
def adjust_frozen_load(x: wp.array(dtype=Any), ids: wp.array(dtype=wp.vec4i),
        kinds: wp.array(dtype=int), count: wp.array(dtype=int), xi: wp.float64,
        loads: wp.array(dtype=wp.float64)):
    i=wp.tid()
    if i<count[0] and count[0]<=ids.shape[0]:
        v=ids[i]
        q,g,h,w=squared_distance(wp.vec3d(x[v[0]]),wp.vec3d(x[v[1]]),
            wp.vec3d(x[v[2]]),wp.vec3d(x[v[3]]),kinds[i],False)
        # Invalid states stay invalid; never turn a penetrated pair into a
        # finite zero friction load. The normal energy/CCD rejects them too.
        if q<=xi*xi or not wp.isfinite(q):
            loads[i]=wp.float64(wp.nan)
        else:
            loads[i]*=wp.sqrt((q-xi*xi)/q)


def adjust_friction(mesh, x, frame, launch):
    if mesh.taccel_discrete:
        launch(adjust_frozen_load,frame.capacity,
            [x,frame.ids,frame.kinds,frame.count,mesh.dmin,frame.loads])


def rigid_edge_weights(mesh, rule):
    """Reference-topology setup only. Retain the pinned release as an ablation.

    The release assigns each closed affine body's edge 2/3 of that body's
    TOTAL area. This is empirically verified on box/scaled box/tetra/subdivided
    box, not endorsed as a local area quadrature or a general mesh guarantee.
    """
    if rule not in ('local','released'):raise ValueError('unknown rigid area rule')
    faces=mesh.faces.numpy();edges=mesh.edges.numpy();groups=mesh.groups.numpy()
    x=mesh._rest_reference;tri=x[faces]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    lookup={tuple(e):i for i,e in enumerate(edges)};values=np.zeros(len(edges))
    degree=np.zeros(len(edges),int)
    for f,a in zip(faces,area):
        for i in range(3):
            k=lookup[tuple(sorted((int(f[i]),int(f[(i+1)%3]))))]
            values[k]+=a/3;degree[k]+=1
    if rule=='released':
        for group in np.unique(groups[groups>=0]):
            edge_mask=np.all(groups[edges]==group,axis=1)
            face_mask=np.all(groups[faces]==group,axis=1)
            if not np.any(face_mask) or np.any(degree[edge_mask]!=2):
                raise ValueError('released rigid area rule requires closed manifold rigid meshes')
            values[edge_mask]=area[face_mask].sum()*2/3
    return values


def configure(energy, *, dhat=None, stiffness=None, rigid_area='local'):
    """Configure before stepping; existing candidate/graph keys track changes."""
    m=energy.mesh
    if energy.surface_contact is not None or m.convergent_contact or m.aggregation_power:
        raise ValueError('Taccel discrete contact is exclusive with surface/aggregated/convergent contact')
    values={'dhat':m.dhat if dhat is None else dhat,
            'stiffness':m.stiffness if stiffness is None else stiffness}
    for name,value in values.items():
        if isinstance(value,bool) or not np.isfinite(value) or value<=0:
            raise ValueError('positive finite contact '+name+' required')
    if values['dhat']<=m.dmin:raise ValueError('contact dhat must exceed the minimum gap')
    weights=rigid_edge_weights(m,rigid_area)
    m.dhat=float(values['dhat']);m.stiffness=float(values['stiffness'])
    m.taccel_discrete=True
    m.taccel_rigid_area=rigid_area
    m.mollifier_coefficient=float(np.float32(1e-3))
    m.ee_multiplicity=2.
    m.edge_area.assign(weights);m.revision+=1
    if m.candidate_cache is not None:m.candidate_cache.valid.zero_()
    m.validate_parameters()
    return metadata(energy)


def metadata(energy):
    m=energy.mesh
    return dict(model='taccel_discrete_pt_ee',dhat=m.dhat,min_gap=m.dmin,
        stiffness=m.stiffness,mu=m.mu,friction_epsilon=energy.epsilon,
        native_taccel=False,upstream_commit='cb23bc251b531ba6908a3788c2f91423cd543149',
        rigid_area_rule=m.taccel_rigid_area,
        mollifier_coefficient=m.mollifier_coefficient,
        ee_multiplicity=m.ee_multiplicity,
        friction_load='sqrt_squared_distance_minus_gap_squared')


def restore(energy, report):
    record=report.get('discrete_contact_parameters')
    if not report.get('taccel_discrete_contact',False):return
    if not isinstance(record,dict) or record.get('model')!='taccel_discrete_pt_ee':
        raise ValueError('missing or unsupported discrete contact checkpoint metadata')
    expected={'upstream_commit':'cb23bc251b531ba6908a3788c2f91423cd543149',
        'mollifier_coefficient':float(np.float32(1e-3)), 'ee_multiplicity':2.,
        'friction_load':'sqrt_squared_distance_minus_gap_squared'}
    for name,value in expected.items():
        if record.get(name)!=value:
            raise ValueError('discrete contact checkpoint mismatch: '+name)
    m=energy.mesh
    for name,actual in [('min_gap',m.dmin),('mu',m.mu),('friction_epsilon',energy.epsilon)]:
        if not np.isclose(record[name],actual,rtol=0,atol=1e-15):
            raise ValueError('discrete contact checkpoint mismatch: '+name)
    configure(energy,dhat=record['dhat'],stiffness=record['stiffness'],rigid_area=record['rigid_area_rule'])
