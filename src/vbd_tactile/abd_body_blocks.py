"""Parallel exact spectral assembly and scaled Cholesky for affine blocks."""
import numpy as np
import warp as wp
from .abd_gpu import affine

wp.set_module_options({'enable_backward':False,'enable_mathdx_solver':False})


@wp.kernel
def spectral_frame(u:wp.array(dtype=wp.vec3d),radius:wp.array(dtype=wp.float64),n:int,
                   left:wp.array(dtype=wp.mat33d),right:wp.array(dtype=wp.mat33d),sigma:wp.array(dtype=wp.vec3d)):
    b=wp.tid()
    ll,ss,rr=wp.svd3(affine(u,n+4*b,radius[b]))
    left[b]=ll;right[b]=rr;sigma[b]=ss


@wp.kernel
def spectral_rows(left:wp.array(dtype=wp.mat33d),right:wp.array(dtype=wp.mat33d),
        sigma:wp.array(dtype=wp.vec3d),radius:wp.array(dtype=wp.float64),
        mass:wp.array(dtype=wp.mat44d),kv:wp.array(dtype=wp.float64),dt:wp.float64,
        h:wp.array3d(dtype=wp.float64),diagonal:wp.array3d(dtype=wp.float64)):
    b,p,q=wp.tid();value=wp.float64(0.)
    if p%3==q%3:value=mass[b][p//3,q//3]/(dt*dt)
    if p>=3 and q>=3:
        lp=(p-3)%3;rp=(p-3)//3;lq=(q-3)%3;rq=(q-3)//3
        uu=left[b];vv=right[b];ss=sigma[b];shape=wp.float64(0.)
        # Same modal order and positive spectrum as shape_hessian_psd;
        # each thread assembles only one scalar instead of copying mat9.
        for i in range(3):
            eigenvalue=wp.max(wp.float64(4.)*(wp.float64(3.)*ss[i]*ss[i]-wp.float64(1.)),wp.float64(0.))
            shape+=eigenvalue*(uu[lp,i]*vv[rp,i])*(uu[lq,i]*vv[rq,i])
            for j in range(i+1,3):
                base=ss[i]*ss[i]+ss[j]*ss[j]-wp.float64(1.)
                cross=ss[i]*ss[j]
                ap=uu[lp,i]*vv[rp,j];bp=uu[lp,j]*vv[rp,i]
                aq=uu[lq,i]*vv[rq,j];bq=uu[lq,j]*vv[rq,i]
                norm=wp.float64(0.7071067811865475244)
                shape+=wp.max(wp.float64(4.)*(base+cross),wp.float64(0.))*((ap+bp)*norm)*((aq+bq)*norm)
                shape+=wp.max(wp.float64(4.)*(base-cross),wp.float64(0.))*((ap-bp)*norm)*((aq-bq)*norm)
        value+=shape*(kv[b]/(radius[b]*radius[b]))
    h[b,p,q]=value;diagonal[b,p,q]=value


@wp.kernel
def matrix_scale(h:wp.array3d(dtype=wp.float64),scale:wp.array(dtype=wp.float64),stats:wp.array(dtype=wp.float64)):
    b=wp.tid();maximum=wp.float64(0.);valid=True
    for i in range(12):
        for j in range(12):
            v=h[b,i,j];valid=valid and wp.isfinite(v);maximum=wp.max(maximum,wp.abs(v))
    if not valid or maximum<=wp.float64(0.):
        wp.atomic_add(stats,2,wp.float64(1.));maximum=wp.float64(1.)
    scale[b]=maximum


@wp.kernel
def prepare_inverse(h:wp.array3d(dtype=wp.float64),scale:wp.array(dtype=wp.float64),out:wp.array3d(dtype=wp.float64)):
    b,i,j=wp.tid();out[b,i,j]=(h[b,i,j]+h[b,j,i])/(wp.float64(2.)*scale[b])


@wp.kernel
def tiled_inverse(prepared:wp.array3d(dtype=wp.float64),identity:wp.array2d(dtype=wp.float64),
                  factor:wp.array3d(dtype=wp.float64),inverse:wp.array3d(dtype=wp.float64)):
    b=wp.tid()
    h=wp.tile_load(prepared[b],shape=(12,12));l=wp.tile_cholesky(h)
    eye=wp.tile_load(identity,shape=(12,12));inv=wp.tile_cholesky_solve(l,eye)
    wp.tile_store(factor[b],l);wp.tile_store(inverse[b],inv)


@wp.kernel
def finish_inverse(raw:wp.array3d(dtype=wp.float64),factor:wp.array3d(dtype=wp.float64),
        scale:wp.array(dtype=wp.float64),out:wp.array3d(dtype=wp.float64),stats:wp.array(dtype=wp.float64)):
    b,i,j=wp.tid()
    # Keep the preconditioner symmetric even if triangular solves round
    # transpose entries differently. Factor positivity is still mandatory.
    value=(raw[b,i,j]+raw[b,j,i])/(wp.float64(2.)*scale[b])
    out[b,i,j]=value
    if not wp.isfinite(value) or not wp.isfinite(factor[b,i,j]) or factor[b,i,i]<=wp.float64(0.):
        wp.atomic_add(stats,2,wp.float64(1.))


class BodyBlocks:
    def __init__(self,count,device):
        self.count=count;self.device=device
        self.left=wp.empty(count,dtype=wp.mat33d,device=device)
        self.right=wp.empty_like(self.left);self.sigma=wp.empty(count,dtype=wp.vec3d,device=device)
        self.scale=wp.empty(count,dtype=wp.float64,device=device)
        self.prepared=wp.empty((count,12,12),dtype=wp.float64,device=device)
        self.factor=wp.empty_like(self.prepared);self.raw_inverse=wp.empty_like(self.prepared)
        self.identity=wp.array(np.eye(12),dtype=wp.float64,device=device)

    def shape(self,c,dt):
        wp.launch(spectral_frame,self.count,[c.u,c.radius,c.n,self.left,self.right,self.sigma],device=self.device)
        wp.launch(spectral_rows,(self.count,12,12),[self.left,self.right,self.sigma,c.radius,c.mass_matrix,c.kv,dt,
            c.body_h.view(wp.float64).reshape((self.count,12,12)),
            c.body_diagonal.view(wp.float64).reshape((self.count,12,12))],device=self.device)

    def inverse(self,diagonal,out,stats):
        h=diagonal.view(wp.float64).reshape((self.count,12,12))
        wp.launch(matrix_scale,self.count,[h,self.scale,stats],device=self.device)
        wp.launch(prepare_inverse,(self.count,12,12),[h,self.scale,self.prepared],device=self.device)
        wp.launch_tiled(tiled_inverse,dim=self.count,inputs=[self.prepared,self.identity,self.factor,self.raw_inverse],block_dim=32,device=self.device)
        wp.launch(finish_inverse,(self.count,12,12),[self.raw_inverse,self.factor,self.scale,
            out.view(wp.float64).reshape((self.count,12,12)),stats],device=self.device)
