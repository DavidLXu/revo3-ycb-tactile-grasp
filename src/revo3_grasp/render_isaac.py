"""Usage (Isaac Sim 5.1 Python, repo src on PYTHONPATH):

    PYTHONPATH=src <isaac-python> -m revo3_grasp.render_isaac --usd <revo3_right_bionic_silver.usda> \\
        --states outputs/013_apple/states.npz --model assets/ycb/013_apple --output outputs/013_apple/render \\
        --title "Revo3 right hand grasping a YCB apple"

Path-traced Isaac Sim replay of a saved old-solver grasp on the textured Revo3 right hand.

Every hand link is posed per frame from the same forward kinematics and joint curls the physics used.
The rigid visual DIP rubber tips are hidden and replaced by the simulated FEM gel surfaces (tip_pad
material). The textured YCB model is placed at the fixed object pose used by the physics (its
decimated closed copy was the contact surface). Gel vertices are written verbatim; no smoothing,
interpolation or exaggeration. Two cameras: grasp overview (slow orbit) and a fingertip close-up.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

p = argparse.ArgumentParser()
p.add_argument('--usd', required=True)
p.add_argument('--states', required=True)
p.add_argument('--output', required=True)
p.add_argument('--width', type=int, default=960)
p.add_argument('--height', type=int, default=1080)
p.add_argument('--spp', type=int, default=48)
p.add_argument('--updates', type=int, default=3)
p.add_argument('--frames', type=int, default=0, help='0 = all saved steps')
p.add_argument('--model', required=True, help='YCB google_16k directory (textured.obj, texture_map.png)')
p.add_argument('--title', required=True)
p.add_argument('--close-finger', default='index')
p.add_argument('--close-view', default='0,1,0.35', help='close camera direction in hand axes: palm-normal, side, up')
p.add_argument('--hold', type=int, default=8, help='extra still frames of the initial pose')
p.add_argument('--wide-view', default='0.5,1,0.25', help='wide camera direction in hand axes: palm-normal, side, up')
p.add_argument('--wide-dist', type=float, default=0.42, help='wide camera distance in metres')
p.add_argument('--close-dist', type=float, default=0.10)
p.add_argument('--close-shift', default='0,0,0', help='close target offset toward the object centre (fraction), side, up in metres')
p.add_argument('--pick', default='', help='comma list of saved-step indices to render (testing only)')
a = p.parse_args()

app = SimulationApp({'headless': True, 'renderer': 'PathTracing', 'width': a.width, 'height': a.height})
try:
    import carb
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

    import trimesh
    from revo3_grasp.hand_fk import HandFK
    from revo3_grasp.paths import FINGERTIPS
    from revo3_grasp.synergy import hand_angles
    FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
    out = Path(a.output)
    (out / 'frames').mkdir(parents=True, exist_ok=False)
    S = np.load(a.states)
    X = S['positions']
    F = S['finger_forces']
    C = S['curls']
    thumb = tuple(S['thumb'])
    profiles = {f: str(p) for f, p in zip(FINGERS, S['profiles'])}
    Tobj = S['object_to_world']
    ranges = S['ranges']
    tris = S['gel_triangles'].reshape(-1, 3)
    initial = S['initial']
    fk = HandFK()
    tips = FINGERTIPS
    sensor_to_link = {f: np.load(tips / f'{f}.npz')['sensor_to_link'] for f in FINGERS}
    zero = fk.poses()
    sensors = np.array([zero[f'right_{f}_DIP_link'] @ sensor_to_link[f] for f in FINGERS])
    root_inv = np.linalg.inv(fk.root)
    poses = [fk.poses(hand_angles(dict(zip(FINGERS, c)), thumb, profiles)) for c in C]
    nframes = len(X) if a.frames <= 0 else min(a.frames, len(X))

    st = carb.settings.get_settings()
    st.set_int('/rtx/pathtracing/spp', a.spp)
    st.set_int('/rtx/pathtracing/totalSpp', a.spp * a.updates)
    st.set_int('/rtx/pathtracing/maxBounces', 6)
    st.set_bool('/rtx/pathtracing/optixDenoiser/enabled', True)

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    hand = stage.DefinePrim('/World/Hand', 'Xform')
    hand.GetReferences().AddReference(str(Path(a.usd).resolve()))
    for _ in range(5):
        app.update()
    for f in FINGERS:
        rubber = stage.GetPrimAtPath(f'/World/Hand/right_{f}_DIP_rubber_link')
        assert rubber.IsValid(), f
        UsdGeom.Imageable(rubber).MakeInvisible()
    tip_pad = UsdShade.Material(stage.GetPrimAtPath('/World/Hand/Looks/tip_pad'))
    assert tip_pad, 'tip_pad material missing'

    # Simulated gel surfaces, one mesh per finger.
    gels = []
    for k, f in enumerate(FINGERS):
        lo, hi = ranges[k]
        sel = tris[(tris >= lo).all(1) & (tris < hi).all(1)]
        used = np.unique(sel)
        remap = np.full(hi, -1)
        remap[used] = np.arange(len(used))
        faces = remap[sel]
        M = sensors[k]
        local = (initial[used] - M[:3, 3]) @ M[:3, :3]
        uv = local[:, :2] / 0.0081
        mesh = UsdGeom.Mesh.Define(stage, f'/World/Gels/{f}')
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.ravel().tolist()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
        mesh.CreateNormalsAttr()
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(tip_pad)
        gels.append((mesh, used, faces))

    # Textured YCB object (render copy), at the fixed physics pose.
    tm = trimesh.load(Path(a.model) / 'textured.obj', force='mesh', process=False)
    obj_xf = UsdGeom.Xform.Define(stage, '/World/Object')
    om = UsdGeom.Mesh.Define(stage, '/World/Object/mesh')
    om.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(tm.vertices, np.float32)))
    om.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tm.faces)))
    om.CreateFaceVertexIndicesAttr(Vt.IntArray(np.asarray(tm.faces).ravel().tolist()))
    om.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(tm.vertex_normals, np.float32)))
    om.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    om.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdGeom.PrimvarsAPI(om).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray.FromNumpy(np.asarray(tm.visual.uv, np.float32)))
    obj_xf.AddTransformOp().Set(Gf.Matrix4d(*np.asarray(Tobj).T.ravel().tolist()))
    omat = UsdShade.Material.Define(stage, '/World/Looks/ycb')
    psh = UsdShade.Shader.Define(stage, '/World/Looks/ycb/Surface'); psh.CreateIdAttr('UsdPreviewSurface')
    psh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.55)
    rd = UsdShade.Shader.Define(stage, '/World/Looks/ycb/st'); rd.CreateIdAttr('UsdPrimvarReader_float2')
    rd.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    tx = UsdShade.Shader.Define(stage, '/World/Looks/ycb/tex'); tx.CreateIdAttr('UsdUVTexture')
    tx.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str((Path(a.model) / 'texture_map.png').resolve()))
    tx.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    tx.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), 'result')
    psh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), 'rgb')
    omat.CreateSurfaceOutput().ConnectToSource(psh.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(om.GetPrim()).Bind(omat)

    # Hand links: one transform op per link, overriding the referenced rest pose.
    link_ops = {}
    for n in fk.rest:
        prim = stage.GetPrimAtPath(f'/World/Hand/{n}')
        if not prim.IsValid() or n == 'world':
            continue
        xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
        link_ops[n] = xf.AddTransformOp(opSuffix='fk')

    def pose_matrix(r):
        r = np.asarray(r, float)
        if r.shape == (4, 4):
            return r
        if r.shape == (12,):  # affine q: translation + 3x3
            T = np.eye(4)
            T[:3, 3] = r[:3]
            T[:3, :3] = r[3:].reshape(3, 3)
            return T
        raise ValueError(f'unsupported rigid pose shape {r.shape}')

    def vertex_normals(x, faces):
        n = np.zeros_like(x)
        fn = np.cross(x[faces[:, 1]] - x[faces[:, 0]], x[faces[:, 2]] - x[faces[:, 0]])
        for c in range(3):
            np.add.at(n, faces[:, c], fn)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)

    def set_frame(i):
        x = X[i]
        for mesh, used, faces in gels:
            q = x[used]
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(q.astype(np.float32)))
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(vertex_normals(q, faces).astype(np.float32)))
        for n, op in link_ops.items():
            op.Set(Gf.Matrix4d(*(root_inv @ poses[i][n]).T.ravel().tolist()))

    # Camera frame from the actual sensor geometry.
    centres = sensors[:, :3, 3]
    normal = sensors[:, :3, 2].mean(0)
    normal /= np.linalg.norm(normal)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bb = cache.ComputeWorldBound(hand).ComputeAlignedBox()
    lo, hi = np.array(bb.GetMin()), np.array(bb.GetMax())
    hand_centre = (lo + hi) / 2
    base = cache.ComputeWorldBound(stage.GetPrimAtPath('/World/Hand/right_hand_base_link')).ComputeAlignedBox()
    base_c = (np.array(base.GetMin()) + np.array(base.GetMax())) / 2
    up = centres[1:].mean(0) - base_c
    up -= up.dot(normal) * normal
    up /= np.linalg.norm(up)
    side = np.cross(up, normal)
    size = float(np.max(hi - lo))

    def look(prim, eye, target):
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        op = ops[0] if ops else xf.AddTransformOp()
        op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(*map(float, up))).GetInverse())

    dome = UsdLux.DomeLight.Define(stage, '/World/Lights/Dome')
    dome.CreateIntensityAttr(520)
    for name, d, w, hgt, power in [('Key', normal + .8 * side + .6 * up, .35, .55, 3200),
                                   ('Rim', -normal - .5 * side + .4 * up, .2, .6, 3600),
                                   ('Fill', normal - 1.1 * side - .2 * up, .5, .5, 1000)]:
        L = UsdLux.RectLight.Define(stage, f'/World/Lights/{name}')
        L.CreateWidthAttr(w)
        L.CreateHeightAttr(hgt)
        L.CreateIntensityAttr(power)
        L.CreateNormalizeAttr(False)
        d = d / np.linalg.norm(d)
        look(L.GetPrim(), hand_centre + .55 * d, hand_centre)

    def camera(name, focal):
        c = UsdGeom.Camera.Define(stage, f'/World/{name}')
        c.CreateFocalLengthAttr(focal)
        c.CreateHorizontalApertureAttr(20.25)
        c.CreateVerticalApertureAttr(20.25 * a.height / a.width)
        c.CreateClippingRangeAttr(Gf.Vec2f(.001, 20))
        return c

    wide = camera('Wide', 28)
    close = camera('Close', 42)
    obj_c = (np.asarray(Tobj) @ np.r_[np.asarray(trimesh.load(Path(a.model) / 'textured.obj', force='mesh').centroid), 1])[:3]
    k = FINGERS.index(a.close_finger); lo_, hi_ = ranges[k]
    close_target = X[len(X) // 2][lo_:hi_].astype(float).mean(0)
    cv = [float(v) for v in a.close_view.split(',')]
    close_dir = cv[0] * normal + cv[1] * side + cv[2] * up
    close_dir /= np.linalg.norm(close_dir)
    cs = [float(v) for v in a.close_shift.split(',')]
    close_target = close_target + cs[0] * (obj_c - close_target) + cs[1] * side + cs[2] * up
    close_dir = close_dir / np.linalg.norm(close_dir)
    look(close.GetPrim(), close_target + a.close_dist * close_dir, close_target)
    # Link prims have no pose until the first frame is set, so frame the shot from the physics positions.
    gel_c = np.array([X[len(X) // 2][r0:r1].astype(float).mean(0) for r0, r1 in ranges])
    wide_target = .55 * obj_c + .45 * gel_c.mean(0)

    products = []
    for cam in (wide, close):
        rp = rep.create.render_product(str(cam.GetPath()), (a.width, a.height))
        an = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
        an.attach(rp)
        products.append((rp, an))

    font_dir = Path('/usr/share/fonts/truetype/dejavu')
    big = ImageFont.truetype(str(font_dir / 'DejaVuSans-Bold.ttf'), 30)
    mid = ImageFont.truetype(str(font_dir / 'DejaVuSans.ttf'), 22)
    small = ImageFont.truetype(str(font_dir / 'DejaVuSans.ttf'), 18)

    def phase(t):
        s = (t - 1e-9) % 2.0
        return 'press' if s < .5 else 'hold 10 N' if s < 1 else 'release' if s < 1.5 else 'idle'

    rows = []
    rep_rows = json.loads((Path(a.states).parent / 'report.json').read_text())['rows']
    PH = [r['phase'] for r in rep_rows]
    FMAX = max(2., float(np.ceil(np.linalg.norm(F, axis=2).max())))
    schedule = [int(v) for v in a.pick.split(',')] if a.pick else [0] * a.hold + list(range(nframes))
    wv = [float(v) for v in a.wide_view.split(',')]
    wbase = wv[0] * normal + wv[1] * side
    wbase = wbase / np.linalg.norm(wbase)
    for fi, i in enumerate(schedule):
        set_frame(i)
        ang = math.radians(25) * math.sin(2 * math.pi * fi / max(len(schedule) - 1, 1))
        d = math.cos(ang) * wbase + math.sin(ang) * np.cross(up, wbase)
        d = d + wv[2] * up
        d /= np.linalg.norm(d)
        look(wide.GetPrim(), wide_target + a.wide_dist * d, wide_target)
        rep.orchestrator.step(rt_subframes=a.updates * (8 if fi == 0 else 1), delta_time=0.0, pause_timeline=False)
        for _ in range(40):
            imgs = [np.asarray(an.get_data()) for _, an in products]
            imgs = [im[..., :3] if im.ndim == 3 else im for im in imgs]
            if all(im.shape == (a.height, a.width, 3) and im.std() > 1 for im in imgs):
                break
            rep.orchestrator.step(rt_subframes=a.updates, delta_time=0.0, pause_timeline=False)
        else:
            for j, im in enumerate(imgs):
                if im.size:
                    Image.fromarray(im).save(out / f'debug_{j}.png')
            (out / 'debug.json').write_text(json.dumps(dict(
                shapes=[list(im.shape) for im in imgs], stds=[float(im.std()) if im.size else None for im in imgs],
                means=[float(im.mean()) if im.size else None for im in imgs],
                hand_lo=lo.tolist(), hand_hi=hi.tolist(), size=size, up=up.tolist(), normal=normal.tolist(),
                side=side.tolist(), centres=centres.tolist(), base_c=base_c.tolist(),
                wide_eye=(wide_target + 1.55 * size * d).tolist(), wide_target=wide_target.tolist()), indent=1))
            raise RuntimeError('empty render after 40 extra updates')
        canvas = Image.new('RGB', (2 * a.width, a.height))
        canvas.paste(Image.fromarray(imgs[0]), (0, 0))
        canvas.paste(Image.fromarray(imgs[1]), (a.width, 0))
        dr = ImageDraw.Draw(canvas)
        dr.rectangle((14, 14, 640, 100), fill=(30, 33, 40))
        dr.rectangle((14, a.height - 58, 380, a.height - 14), fill=(30, 33, 40))
        dr.rectangle((a.width + 14, 14, a.width + 560, 64), fill=(30, 33, 40))
        t = (i + 1) / 30
        dr.rectangle((14, 14, max(820, 42 + int(dr.textlength(a.title, font=big))), 100), fill=(30, 33, 40))
        dr.text((28, 24), a.title, font=big, fill=(245, 245, 245))
        dr.text((28, 64), 'FEM fingertip gels + GPU IPC · finger motion prescribed · object held fixed', font=small, fill=(225, 225, 225))
        dr.text((28, a.height - 48), f't = {t:4.2f} s   {PH[i]}', font=mid, fill=(245, 245, 245))
        dr.text((a.width + 28, 24), f'{a.close_finger.capitalize()} fingertip, close-up (same frame)', font=mid, fill=(245, 245, 245))
        y = a.height - 48 - 5 * 30
        dr.rectangle((a.width + 14, y - 44, a.width + 520, y + 5 * 30 + 4), fill=(30, 33, 40))
        dr.text((a.width + 28, y - 34), 'Gel contact force', font=small, fill=(230, 230, 230))
        mags = np.linalg.norm(F[i], axis=1)
        for k, f in enumerate(FINGERS):
            yy = y + 30 * k
            dr.text((a.width + 28, yy), f, font=small, fill=(230, 230, 230))
            dr.rectangle((a.width + 110, yy + 4, a.width + 110 + 300, yy + 20), outline=(200, 200, 200))
            dr.rectangle((a.width + 110, yy + 4, a.width + 110 + int(300 * min(mags[k] / FMAX, 1)), yy + 20), fill=(90, 160, 255))
            dr.text((a.width + 420, yy), f'{mags[k]:5.2f} N', font=small, fill=(230, 230, 230))
        canvas.save(out / 'frames' / f'{fi:04d}.png')
        rows.append(dict(frame=fi, step=i + 1, time_s=t, contact_N=mags.tolist(), wide_orbit_deg=math.degrees(ang)))
        if fi % 10 == 0:
            print('FRAME', fi, '/', len(schedule), flush=True)
    (out / 'render_report.json').write_text(json.dumps(dict(
        usd=str(Path(a.usd).resolve()), states=str(Path(a.states).resolve()), frames=nframes,
        resolution=[2 * a.width, a.height], renderer='PathTracing', spp_per_update=a.spp, updates_per_frame=a.updates,
        hidden_visual_prims=[f'right_{f}_DIP_rubber_link' for f in FINGERS], model=str(Path(a.model).resolve()),
        held_initial_frames=a.hold, gel_material='/Revo3/Looks/tip_pad',
        gel_uv='planar sensor-XY rest coordinates / 8.1 mm', interpolation=False, deformation_scale=1,
        rows=rows), indent=2))
    for rp, an in products:
        try:  # Replicator 1.12 teardown can raise after all frames are saved.
            an.detach(rp)
            rp.destroy()
        except Exception as exc:
            print('TEARDOWN_IGNORED', type(exc).__name__, flush=True)
    print('RENDER_COMPLETE', flush=True)
except BaseException:
    import os
    import traceback
    traceback.print_exc()
    os._exit(1)
finally:
    app.close()
