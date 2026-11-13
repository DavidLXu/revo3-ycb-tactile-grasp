"""Independent reference for area-integrated contact against convex rigid solids.

The surface is partitioned by the rigid solid's closest-feature Voronoi cells.
Faces, genuine edges and corners all integrate the SAME reference-area density;
there are no arbitrary line/point stiffness multipliers or counted edge areas.
The inverse-square compact barrier follows IPC Toolkit's scalar GCP kernel,
not its full geometric/contact discretization. This is a distinct formulation.
CPU routines here are setup/oracles only, never a physics fallback.
"""
from dataclasses import dataclass
import numpy as np
from .ipc_primitives import closest_feature


def inverse_density(distance, dhat=5e-5, min_gap=1e-6, stiffness=1000.):
    """kappa*dhat*spline(s)/s^2, s=(distance-min_gap)/dhat; units J/m^2."""
    if any(not np.isfinite(v) or v<=0 for v in (dhat,min_gap,stiffness)):
        raise ValueError('positive finite contact parameters required')
    s=(np.asarray(distance,dtype=float)-min_gap)/dhat
    if not np.isfinite(s).all() or np.any(s<=0):raise ValueError('contact clearance violated')
    t=np.minimum(s,1.);small=t<.5
    a=np.where(small,2/3-4*t*t*(1-t),4/3*(1-t)**3)
    b=np.where(small,4*t*(3*t-2),-4*(1-t)**2)
    c=np.where(small,8*(3*t-1),8*(1-t))
    return (stiffness*dhat*a/t**2,stiffness*(b-2*a/t)/t**2,
            stiffness/dhat*(c-4*b/t+6*a/t**2)/t**2)


def friction_density(displacement,normal,load,epsilon):
    tangent=displacement-normal*np.einsum('ij,ij->i',displacement,normal)[:,None]
    radius=np.linalg.norm(tangent,axis=1);small=radius<epsilon
    safe=np.maximum(radius,np.finfo(float).tiny)
    energy=np.where(small,radius**2/epsilon-radius**3/(3*epsilon**2),radius-epsilon/3)
    ratio=np.where(small,2/epsilon-radius/epsilon**2,1/safe)
    return load*energy,load[:,None]*ratio[:,None]*tangent


@dataclass
class ConvexFeatures:
    # Each plane record (kind,a,b,c,d): kind0 dot(b-a,p-a)>=0;
    # kind1 dot(cross(b-a,c-a),p-a)>=0; kind2 negative edge-orthogonal
    # projection of c-a; kind3 inward edge normal of oriented triangle.
    ids: np.ndarray
    kinds: np.ndarray
    offsets: np.ndarray
    planes: np.ndarray
    body: np.ndarray
    faces: np.ndarray

    @classmethod
    def build(cls,rest,faces,groups):
        rest=np.asarray(rest,float);faces=np.asarray(faces);groups=np.asarray(groups)
        if not np.issubdtype(faces.dtype,np.integer) or not np.issubdtype(groups.dtype,np.integer):
            raise ValueError('integer triangle indices and rigid groups required')
        if rest.ndim!=2 or rest.shape[1]!=3 or not np.isfinite(rest).all():raise ValueError('finite Nx3 rest positions required')
        if faces.ndim!=2 or faces.shape[1]!=3 or (faces.size and (faces.min()<0 or faces.max()>=len(rest))):raise ValueError('valid triangles required')
        if groups.shape!=(len(rest),):raise ValueError('one collision group per node required')
        limit=np.iinfo(np.int32)
        if (faces.size and faces.max()>limit.max) or (groups.size and (groups.min()<limit.min or groups.max()>limit.max)):
            raise ValueError('topology exceeds int32 capacity')
        faces=faces.astype(np.int32);groups=groups.astype(np.int32)
        if np.any(np.any(groups[faces]>=0,axis=1)&~np.all(groups[faces]==groups[faces[:,0],None],axis=1)):
            raise ValueError('triangles cannot mix rigid bodies or soft and rigid nodes')
        ids=[];kinds=[];offsets=[0];planes=[];body=[];oriented=[]
        for group in sorted(set(groups[faces.ravel()])):
            if group<0:continue
            ff=faces[np.all(groups[faces]==group,axis=1)].copy();nodes=np.unique(ff)
            if not len(ff):continue
            center=rest[nodes].mean(0);tri=rest[ff]-center
            volume=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
            if volume<0:ff=ff[:,::-1];tri=tri[:,::-1]
            if volume==0:raise ValueError('closed nonzero-volume rigid solid required')
            normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normals,axis=1)
            if np.any(length==0):raise ValueError('degenerate rigid face')
            normals/=length[:,None];size=float(np.ptp(rest[nodes],axis=0).max());tol=1e-10*size
            if np.max(np.einsum('fvi,fi->fv',rest[nodes][None,:,:]-rest[ff[:,0]][:,None,:],normals))>tol:
                raise ValueError('surface-integral backend currently requires convex rigid bodies')
            edges={};incident={int(v):[] for v in nodes};neighbors={int(v):set() for v in nodes}
            for i,f in enumerate(ff):
                for v in f:incident[int(v)].append(i)
                for a,b,c in ((f[0],f[1],f[2]),(f[1],f[2],f[0]),(f[2],f[0],f[1])):
                    edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append((i,int(c),int(a),int(b)))
                    neighbors[int(a)].add(int(b));neighbors[int(b)].add(int(a))
            if any(len(e)!=2 or e[0][2:]!=e[1][2:][::-1] for e in edges.values()):
                raise ValueError('consistently oriented closed manifold rigid surface required')
            def add(kind,ind,pp):
                ids.append(ind);kinds.append(kind);planes.extend(pp);offsets.append(len(planes));body.append(group)
            true_edges={key:adj for key,adj in edges.items()
                if np.linalg.norm(normals[adj[0][0]]-normals[adj[1][0]])>1e-10}
            corners={int(a) for a in nodes if np.linalg.matrix_rank(normals[incident[int(a)]],tol=1e-10)>=3}
            edge_at={int(a):[] for a in nodes}
            for key in true_edges:
                for a in key:edge_at[a].append(key)
            # Halfspace d marks an excluded all-zero (positive source-area)
            # tie. Prefer lower-dimensional physical features; on purely
            # artificial subdivisions choose one deterministic owner.
            for i,f in enumerate(ff):
                a,b,c=map(int,f);pp=[(1,a,b,c,0)]
                for aa,bb,cc in ((a,b,c),(b,c,a),(c,a,b)):
                    key=tuple(sorted((aa,bb)));adj=edges[key]
                    strict=int(key in true_edges or i!=min(t[0] for t in adj))
                    pp.append((3,aa,bb,cc,strict))
                add(2,(a,b,c),pp)
            for (a,b),adj in true_edges.items():
                add(1,(a,b,b),[(0,a,b,0,int(a in corners or (a,b)!=min(edge_at[a]))),(0,b,a,0,int(b in corners or (a,b)!=min(edge_at[b]))),
                    (2,a,b,adj[0][1],0),(2,a,b,adj[1][1],0)])
            for a in nodes:
                if a not in corners:continue
                # kind4 reverses the edge halfspace for an exposed corner.
                add(0,(int(a),int(a),int(a)),[(4,int(a),b,0,0) for b in sorted(neighbors[int(a)])])
            oriented.extend(ff)
        return cls(np.asarray(ids,np.int32).reshape(-1,3),np.asarray(kinds,np.int32),np.asarray(offsets,np.int32),
                   np.asarray(planes,np.int32).reshape(-1,5),np.asarray(body,np.int32),np.asarray(oriented,np.int32).reshape(-1,3))

    def halfspaces(self,x,index):
        out=[]
        for kind,a,b,c,d in self.planes[self.offsets[index]:self.offsets[index+1]]:
            e=x[b]-x[a]
            if kind==0:n=e
            elif kind==4:n=-e
            elif kind==1:n=np.cross(e,x[c]-x[a])
            elif kind==2:n=-(x[c]-x[a]-e*np.dot(e,x[c]-x[a])/np.dot(e,e))
            else:n=np.cross(np.cross(e,x[c]-x[a]),e)
            length=np.linalg.norm(n)
            if length==0:raise ValueError('degenerate moving rigid geometry')
            out.append((x[a],n/length))
        return out


def clip(poly,values):
    out=[]
    if not len(poly):return np.empty((0,3))
    a=poly[-1];fa=values[-1]
    for b,fb in zip(poly,values):
        if (fa>=0)!=(fb>=0):out.append(a+fa/(fa-fb)*(b-a))
        if fb>=0:out.append(b)
        a,fa=b,fb
    return np.asarray(out).reshape(-1,3)


def feature_values(points,target,kind):
    """Euclidean closest feature inside its Voronoi cell (no clamping)."""
    relative=points-target[0]
    alpha=np.zeros((len(points),3));alpha[:,0]=1
    if kind==1:
        e=target[1]-target[0];t=relative@e/(e@e);alpha[:,0]=1-t;alpha[:,1]=t
    elif kind==2:
        e1=target[1]-target[0];e2=target[2]-target[0];n=np.cross(e1,e2);den=n@n
        u=np.cross(relative,e2)@n/den;v=np.cross(e1,relative)@n/den
        alpha=np.column_stack((1-u-v,u,v))
    delta=relative-alpha[:,1,None]*(target[1]-target[0])-alpha[:,2,None]*(target[2]-target[0])
    distance=np.linalg.norm(delta,axis=1)
    return distance,delta,alpha


_BARY=np.array([[1/3,1/3,1/3],
 [.059715871789770,.470142064105115,.470142064105115],
 [.470142064105115,.059715871789770,.470142064105115],
 [.470142064105115,.470142064105115,.059715871789770],
 [.797426985353087,.101286507323456,.101286507323456],
 [.101286507323456,.797426985353087,.101286507323456],
 [.101286507323456,.101286507323456,.797426985353087]])
_WEIGHTS=np.array([.225]+[.132394152788506]*3+[.125939180544827]*3);_WEIGHTS/=_WEIGHTS.sum()


@dataclass
class SurfaceSamples:
    ids: np.ndarray
    weights: np.ndarray
    normals: np.ndarray
    area: np.ndarray
    distance: np.ndarray
    kinds: np.ndarray
    max_depth: int

    def friction(self,x,motion,mu,epsilon,**parameters):
        _,first,_=inverse_density(self.distance,**parameters)
        d=np.einsum('ki,kij->kj',self.weights,np.asarray(motion)[self.ids])
        e,g=friction_density(d,self.normals,-mu*self.area*first,epsilon)
        gradient=np.zeros_like(x,dtype=float)
        np.add.at(gradient,self.ids.ravel(),(self.weights[:,:,None]*g[:,None,:]).reshape(-1,3))
        return e.sum(),gradient

    def evaluate(self,x,**parameters):
        e,first,second=inverse_density(self.distance,**parameters)
        g=np.zeros_like(x,dtype=float)
        np.add.at(g,self.ids.ravel(),(self.area[:,None,None]*first[:,None,None]*self.weights[:,:,None]*self.normals[:,None,:]).reshape(-1,3))
        return float(self.area@e),g,self.area*second


def integrate_surface(rest,x,slave_faces,features,*,dhat=5e-5,min_gap=1e-6,stiffness=1000.,rtol=1e-5,max_depth=32,motion=None,mu=0.,epsilon=1e-5):
    """Offline adaptive integration of ALL closest-feature cells of convex solids."""
    rest=np.asarray(rest,float);x=np.asarray(x,float);sf=np.asarray(slave_faces)
    if not 0<rtol<1 or not 0<=max_depth<=40:raise ValueError('invalid quadrature controls')
    if x.shape!=rest.shape or x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all() or not np.isfinite(rest).all():raise ValueError('finite matching rest and current positions required')
    if not np.issubdtype(sf.dtype,np.integer):raise ValueError('integer slave triangles required')
    if sf.ndim!=2 or sf.shape[1]!=3 or (sf.size and (sf.min()<0 or sf.max()>=len(x))):raise ValueError('invalid slave triangles')
    if motion is not None and (np.asarray(motion).shape!=x.shape or not np.isfinite(motion).all()):raise ValueError('finite matching nodal motion required')
    if not np.isfinite(mu) or mu<0 or not np.isfinite(epsilon) or epsilon<=0:raise ValueError('invalid friction controls')
    extent=dhat+min_gap;ids=[];coeff=[];normals=[];areas=[];distances=[];kinds=[];deepest=0
    for j,ind in enumerate(features.ids):
        target=x[ind];lo=target.min(0)-extent;hi=target.max(0)+extent;kind=int(features.kinds[j])
        for face in sf:
            slave=x[face]
            if np.any(slave.max(0)<lo) or np.any(slave.min(0)>hi):continue
            poly=np.eye(3)
            for plane,(anchor,n) in zip(features.planes[features.offsets[j]:features.offsets[j+1]],features.halfspaces(x,j)):
                values=(slave-anchor)@n
                rounding=64*np.finfo(float).eps*(np.linalg.norm(slave,axis=1).max()+np.linalg.norm(anchor)+extent)
                if np.max(np.abs(values))<=rounding:
                    if plane[4]:poly=np.empty((0,3));break
                    values=np.zeros(3)
                poly=clip(poly,poly@values)
                if len(poly)<3:break
            if len(poly)<3:continue
            if kind==2:
                normal=np.cross(target[1]-target[0],target[2]-target[0]);normal/=np.linalg.norm(normal)
                poly=clip(poly,poly@(extent-(slave-target[0])@normal))
            if len(poly)<3:continue
            for axis in range(3):
                poly=clip(poly,poly@(slave[:,axis]-lo[axis]));poly=clip(poly,poly@(hi[axis]-slave[:,axis]))
            if len(poly)<3:continue
            rr=rest[face];reference_area=np.linalg.norm(np.cross(rr[1]-rr[0],rr[2]-rr[0]))/2
            def rule(tri,area):
                beta=_BARY@tri;points=slave[0]+beta[:,1,None]*(slave[1]-slave[0])+beta[:,2,None]*(slave[2]-slave[0])
                distance,delta,alpha=feature_values(points,target,kind)
                e,f,h=inverse_density(distance,dhat,min_gap,stiffness);w=np.column_stack((beta,-alpha));n=delta/distance[:,None]
                weight=area*_WEIGHTS
                v=np.r_[weight@e,np.einsum('k,ki,kj->ij',weight*f,w,n).ravel()]
                if motion is not None:
                    dis=w@np.asarray(motion)[np.r_[face,ind]]
                    fe,fg=friction_density(dis,n,-mu*weight*f,epsilon)
                    v=np.r_[v,fe.sum(),np.einsum('ki,kj->ij',w,fg).ravel()]
                return v,beta,w,n,distance,weight
            stack=[]
            for k in range(1,len(poly)-1):
                tri=np.stack((poly[0],poly[k],poly[k+1]));de=tri[1:,1:]-tri[0,1:]
                area=reference_area*abs(np.linalg.det(de))
                if area>0:stack.append((tri,area,0))
            while stack:
                tri,area,depth=stack.pop();vertices=slave[0]+tri[:,1,None]*(slave[1]-slave[0])+tri[:,2,None]*(slave[2]-slave[0])
                vd,delta,alpha=feature_values(vertices,target,kind)
                if kind==2:minimum=vd.min()
                else:
                    pts=np.vstack((np.zeros(3),delta));w,_=closest_feature(pts,'point_triangle');minimum=np.linalg.norm(w@pts)
                if minimum<=min_gap:raise ValueError('contact domain includes a clearance violation')
                if minimum>=extent:continue
                coarse=rule(tri,area)
                edge_scores=np.array([np.sum((delta[a]-delta[b])**2)/dhat**2 for a,b in ((0,1),(1,2),(2,0))])
                if motion is not None:
                    dis=np.column_stack((tri,-alpha))@np.asarray(motion)[np.r_[face,ind]]
                    edge_scores += [np.sum((dis[a]-dis[b])**2)/epsilon**2 for a,b in ((0,1),(1,2),(2,0))]
                edge=((0,1),(1,2),(2,0))[np.argmax(edge_scores)]
                a,b=edge;mid=(tri[a]+tri[b])/2;l=tri.copy();r=tri.copy();l[a]=mid;r[b]=mid
                left=rule(l,area/2);right=rule(r,area/2);fine=left[0]+right[0]
                atol=area*stiffness*1e-9
                de=abs(coarse[0][0]-fine[0])/(rtol*abs(fine[0])+atol*dhat)
                dg=np.linalg.norm((coarse[0][1:19]-fine[1:19]).reshape(6,3),axis=1)
                scale=rtol*np.linalg.norm(fine[1:19].reshape(6,3),axis=1)+atol
                error=max(de,float(np.max(dg/scale)))
                if motion is not None:
                    fe=abs(coarse[0][19]-fine[19])/(rtol*abs(fine[19])+atol*epsilon)
                    fg=np.linalg.norm((coarse[0][20:]-fine[20:]).reshape(6,3),axis=1)
                    fs=rtol*np.linalg.norm(fine[20:].reshape(6,3),axis=1)+atol
                    error=max(error,fe,float(np.max(fg/fs)))
                bound=inverse_density([minimum],dhat,min_gap,stiffness)
                negligible=area*abs(bound[1][0])*max(1.,mu)<=atol
                resolved=vd.max()-minimum<=2*(minimum-min_gap)
                nonzero=fine[0]>0
                if negligible or (resolved and nonzero and error<=1):
                    for _,beta,w,n,distance,weight in (left,right):
                        use=distance<extent;count=int(use.sum())
                        ids.extend([np.r_[face,ind]]*count);coeff.extend(w[use]);normals.extend(n[use]);distances.extend(distance[use]);areas.extend(weight[use]);kinds.extend([kind]*count)
                    deepest=max(deepest,depth)
                else:
                    if depth>=max_depth:raise RuntimeError(f'surface quadrature unresolved: depth={depth}, error={error}, gap={minimum}')
                    stack.extend(((l,area/2,depth+1),(r,area/2,depth+1)))
    return SurfaceSamples(np.asarray(ids,np.int32).reshape(-1,6),np.asarray(coeff,float).reshape(-1,6),np.asarray(normals,float).reshape(-1,3),np.asarray(areas),np.asarray(distances),np.asarray(kinds,np.int32),deepest)
