"""Deterministic collision triangulation of flat rigid facets.

Only machine-precision coplanar faces are merged. No convex hull or curved
surface decimation is performed. Facets with holes retain their original
triangles. This removes arbitrary planar subdivisions; it is not a claim of
convergent contact quadrature on arbitrary curved meshes.
"""
import numpy as np


def canonical_planar_mesh(mesh):
    from .rigid_mesh import RigidTriangleMesh
    v,f=mesh.vertices,mesh.triangles
    scale=np.ptp(v,axis=0).max()
    tol=64*np.finfo(float).eps*scale
    triangles=v[f]; normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    edges={}
    for i,tri in enumerate(f):
        for a,b in zip(tri,np.roll(tri,-1)):edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
    adjacent=[[] for _ in f]
    for pair in edges.values():
        a,b=pair;adjacent[a].append(b);adjacent[b].append(a)
    labels=np.full(len(f),-1,int);groups=[]
    # Coordinate-based seed order makes the result independent of face order.
    order=sorted(range(len(f)),key=lambda i:tuple(sorted(map(tuple,triangles[i]))))
    for seed in order:
        if labels[seed]>=0:continue
        index=len(groups); labels[seed]=index; stack=[seed];group=[]
        while stack:
            i=stack.pop();group.append(i)
            for j in adjacent[i]:
                if labels[j]>=0:continue
                if np.dot(normals[seed],normals[j])>1-1e-12 and np.abs((triangles[j]-triangles[seed,0])@normals[seed]).max()<=tol:
                    labels[j]=index;stack.append(j)
        groups.append(group)
    owners=[set() for _ in v]
    for i,tri in enumerate(f):
        for vertex in tri:owners[vertex].add(labels[i])
    output=[]
    for group in groups:
        if len(group)==1:output.extend(f[group]);continue
        boundary=[]
        for i in group:
            for a,b in zip(f[i],np.roll(f[i],-1)):
                if any(labels[j]!=labels[i] for j in edges[tuple(sorted((int(a),int(b))))]):boundary.append((int(a),int(b)))
        nxt=dict(boundary)
        if not boundary or len(nxt)!=len(boundary):output.extend(f[group]);continue
        first=min(nxt,key=lambda i:tuple(v[i]));loop=[first]
        while nxt.get(loop[-1],first)!=first and len(loop)<=len(boundary):loop.append(nxt[loop[-1]])
        if len(loop)!=len(boundary) or len(set(loop))!=len(loop):output.extend(f[group]);continue
        changed=True
        while changed and len(loop)>3:
            changed=False
            for i,b in enumerate(loop):
                a,c=loop[i-1],loop[(i+1)%len(loop)]
                u,w=v[b]-v[a],v[c]-v[b]
                if len(owners[b])<=2 and np.linalg.norm(np.cross(u,w))<=tol*(np.linalg.norm(u)+np.linalg.norm(w)) and np.dot(u,w)>=0:
                    loop.pop(i);changed=True;break
        # Ear clipping supports concave facets, with a deterministic ear order.
        normal=normals[group[0]];basis=v[loop[1]]-v[loop[0]];basis/=np.linalg.norm(basis)
        uv=np.column_stack(((v-v[loop[0]])@basis,(v-v[loop[0]])@np.cross(normal,basis)))
        def cross(a,b):return a[0]*b[1]-a[1]*b[0]
        pending=loop.copy();part=[]
        while len(pending)>3:
            found=False
            for b in sorted(pending,key=lambda i:tuple(v[i])):
                i=pending.index(b);a,c=pending[i-1],pending[(i+1)%len(pending)]
                if cross(uv[b]-uv[a],uv[c]-uv[b])<=tol*scale:continue
                inside=False
                for p in pending:
                    if p in (a,b,c):continue
                    if min(cross(uv[b]-uv[a],uv[p]-uv[a]),cross(uv[c]-uv[b],uv[p]-uv[b]),cross(uv[a]-uv[c],uv[p]-uv[c]))>=-tol*scale:
                        inside=True;break
                if not inside:
                    part.append((a,b,c));pending.pop(i);found=True;break
            if not found:break
        if len(pending)==3:output.extend(part+[tuple(pending)])
        else:output.extend(f[group])
    faces=np.asarray(output,int);used=np.unique(faces)
    used=np.array(sorted(used,key=lambda i:tuple(v[i])),int)
    remap=np.full(len(v),-1,int);remap[used]=np.arange(len(used));faces=remap[faces]
    faces=np.array([np.roll(tri,-np.argmin(tri)) for tri in faces])
    faces=np.array(sorted(map(tuple,faces)),int)
    # Never silently accept cracks or partial simplification at facet junctions.
    return RigidTriangleMesh(v[used],faces,mesh.name)
