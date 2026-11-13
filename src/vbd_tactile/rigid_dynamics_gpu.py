"""GPU rigid momentum balance and small dense interface solve.

Translations are COM displacements. Rotation coordinates are world rotation
vectors scaled by a body's radius. Residuals therefore all have force units.
Angular balance uses world angular momentum, including changing inertia.
"""
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def exponential(theta: wp.vec3d):
    angle=wp.length(theta)
    a=wp.float64(1.); b=wp.float64(.5)
    if angle>wp.float64(1.e-7):
        a=wp.sin(angle)/angle
        b=(wp.float64(1.)-wp.cos(angle))/(angle*angle)
    else:
        a-=angle*angle/wp.float64(6.)
        b-=angle*angle/wp.float64(24.)
    x,y,z=theta[0],theta[1],theta[2]
    skew=wp.mat33d(0.,-z,y,z,0.,-x,-y,x,0.)
    return wp.identity(n=3,dtype=wp.float64)+a*skew+b*(skew*skew)


@wp.func
def rotation(pose: wp.mat44d):
    return wp.mat33d(pose[0,0],pose[0,1],pose[0,2],pose[1,0],pose[1,1],pose[1,2],pose[2,0],pose[2,1],pose[2,2])


@wp.func
def translation(pose: wp.mat44d):
    return wp.vec3d(pose[0,3],pose[1,3],pose[2,3])


@wp.kernel
def poses(q: wp.array(dtype=wp.float64), old: wp.array(dtype=wp.mat44d),
          com: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64),
          out: wp.array(dtype=wp.mat44d)):
    i=wp.tid(); k=6*i; r=rotation(old[i])
    center=translation(old[i])+r*com[i]+wp.vec3d(q[k],q[k+1],q[k+2])
    rnew=exponential(wp.vec3d(q[k+3],q[k+4],q[k+5])/radius[i])*r
    p=center-rnew*com[i]
    out[i]=wp.mat44d(rnew[0,0],rnew[0,1],rnew[0,2],p[0],rnew[1,0],rnew[1,1],rnew[1,2],p[1],
                    rnew[2,0],rnew[2,1],rnew[2,2],p[2],0.,0.,0.,1.)


@wp.kernel
def rotation_guard(q: wp.array(dtype=wp.float64), radius: wp.array(dtype=wp.float64),
                   origin_radius: wp.array(dtype=wp.float64), gap: wp.float64,
                   curved_path: bool, status: wp.array(dtype=wp.float64)):
    i=wp.tid();k=6*i
    theta=wp.length(wp.vec3d(q[k+3],q[k+4],q[k+5]))/radius[i]
    # Principal rotation coordinates must agree with the prescribed SLERP.
    # The legacy path also imposes a fixed sagitta limit; the curved GPU
    # certificate instead checks actual clearance along the whole arc.
    sine=wp.sin(theta/wp.float64(4.))
    guard=wp.float64(4.)*origin_radius[i]*sine*sine
    if not wp.isfinite(theta) or theta>wp.float64(wp.pi) or (not curved_path and guard>gap*wp.float64(.25)):
        wp.atomic_max(status,0,wp.float64(wp.inf))


@wp.kernel
def wrench(x: wp.array(dtype=wp.vec3d), force: wp.array(dtype=wp.vec3d), owners: wp.array(dtype=int),
           pose: wp.array(dtype=wp.mat44d), com: wp.array(dtype=wp.vec3d), out: wp.array2d(dtype=wp.float64)):
    i=wp.tid(); body=owners[i]
    if body>=0:
        arm=x[i]-translation(pose[body])-rotation(pose[body])*com[body]
        torque=wp.cross(arm,force[i])
        for j in range(3):
            wp.atomic_add(out,body,j,force[i][j])
            wp.atomic_add(out,body,j+3,torque[j])


@wp.kernel
def balance(q: wp.array(dtype=wp.float64), oldpose: wp.array(dtype=wp.mat44d),
            pose: wp.array(dtype=wp.mat44d), velocity: wp.array2d(dtype=wp.float64),
            mass: wp.array(dtype=wp.float64), inertia: wp.array(dtype=wp.mat33d),
            radius: wp.array(dtype=wp.float64), forces: wp.array2d(dtype=wp.float64),
            external: wp.array2d(dtype=wp.float64), gravity: wp.vec3d, dt: wp.float64,
            residual: wp.array(dtype=wp.float64), stats: wp.array(dtype=wp.float64)):
    i=wp.tid();k=6*i
    r0,r1=rotation(oldpose[i]),rotation(pose[i])
    oldw=wp.vec3d(velocity[i,3],velocity[i,4],velocity[i,5])
    w=wp.vec3d(q[k+3],q[k+4],q[k+5])/(dt*radius[i])
    angular=(r1*inertia[i]*wp.transpose(r1)*w-r0*inertia[i]*wp.transpose(r0)*oldw)/dt
    for j in range(6):
        value=wp.float64(0.)
        if j<3:value=mass[i]*(q[k+j]-dt*velocity[i,j])/(dt*dt)-mass[i]*gravity[j]-forces[i,j]-external[i,j]
        else:value=(angular[j-3]-forces[i,j]-external[i,j])/radius[i]
        residual[k+j]=value
        if not wp.isfinite(value):
            wp.atomic_max(stats,0,wp.float64(wp.inf))
        else:wp.atomic_max(stats,0,wp.abs(value))


@wp.kernel
def initial_inverse(oldpose: wp.array(dtype=wp.mat44d), mass: wp.array(dtype=wp.float64),
                    inertia: wp.array(dtype=wp.mat33d), radius: wp.array(dtype=wp.float64),
                    dt: wp.float64, inverse: wp.array2d(dtype=wp.float64)):
    i=wp.tid();k=6*i
    for j in range(3):inverse[k+j,k+j]=dt*dt/mass[i]
    r=rotation(oldpose[i]);a=wp.inverse(r*inertia[i]*wp.transpose(r))*dt*dt*radius[i]*radius[i]
    for j in range(3):
        for l in range(3):inverse[k+j+3,k+l+3]=a[j,l]


@wp.kernel
def inertial_matrix(oldpose: wp.array(dtype=wp.mat44d),mass: wp.array(dtype=wp.float64),
                    inertia: wp.array(dtype=wp.mat33d),radius: wp.array(dtype=wp.float64),
                    dt: wp.float64,out: wp.array2d(dtype=wp.float64)):
    i=wp.tid();k=6*i
    for j in range(3):out[k+j,k+j]=mass[i]/(dt*dt)
    r=rotation(oldpose[i]);a=r*inertia[i]*wp.transpose(r)/(dt*dt*radius[i]*radius[i])
    for j in range(3):
        for l in range(3):out[k+j+3,k+l+3]=a[j,l]


@wp.kernel
def cache_stiffness(jacobian: wp.array2d(dtype=wp.float64),mass: wp.array2d(dtype=wp.float64),
                    status: wp.array(dtype=wp.float64),valid: wp.array(dtype=int),
                    stiffness: wp.array2d(dtype=wp.float64)):
    i,j=wp.tid()
    value=jacobian[i,j]-mass[i,j]
    if not wp.isfinite(status[0]) or not wp.isfinite(value):wp.atomic_min(valid,0,0)
    stiffness[i,j]=value


@wp.kernel
def interface_matrix(stiffness: wp.array2d(dtype=wp.float64),mass: wp.array2d(dtype=wp.float64),
                     valid: wp.array(dtype=int),out: wp.array2d(dtype=wp.float64)):
    i,j=wp.tid();out[i,j]=mass[i,j]
    if valid[0]:out[i,j]+=stiffness[i,j]


@wp.kernel
def choose_inverse(candidate: wp.array2d(dtype=wp.float64),status: wp.array(dtype=wp.float64),
                   inverse: wp.array2d(dtype=wp.float64)):
    i,j=wp.tid()
    if wp.isfinite(status[0]):inverse[i,j]=candidate[i,j]


@wp.kernel
def direction(inverse: wp.array2d(dtype=wp.float64), residual: wp.array(dtype=wp.float64),
              out: wp.array(dtype=wp.float64)):
    i=wp.tid();value=wp.float64(0.)
    for j in range(residual.shape[0]):value-=inverse[i,j]*residual[j]
    out[i]=value


@wp.kernel
def trial(base: wp.array(dtype=wp.float64), delta: wp.array(dtype=wp.float64),
          alpha: wp.float64, out: wp.array(dtype=wp.float64)):
    i=wp.tid();out[i]=base[i]+alpha*delta[i]


@wp.kernel
def secant(q: wp.array(dtype=wp.float64), oldq: wp.array(dtype=wp.float64),
           f: wp.array(dtype=wp.float64), oldf: wp.array(dtype=wp.float64),
           inverse: wp.array2d(dtype=wp.float64), work: wp.array(dtype=wp.float64)):
    # The vector B*y is completed before the matrix update in a separate launch.
    i=wp.tid();value=wp.float64(0.)
    for j in range(f.shape[0]):value+=inverse[i,j]*(f[j]-oldf[j])
    work[i]=q[i]-oldq[i]-value


@wp.kernel
def update_inverse(f: wp.array(dtype=wp.float64), oldf: wp.array(dtype=wp.float64),
                   work: wp.array(dtype=wp.float64), inverse: wp.array2d(dtype=wp.float64)):
    i,j=wp.tid();norm=wp.float64(0.)
    for k in range(f.shape[0]):norm+=(f[k]-oldf[k])*(f[k]-oldf[k])
    if norm>wp.float64(1.e-30):inverse[i,j]+=work[i]*(f[j]-oldf[j])/norm


@wp.kernel
def update_inverse_bfgs(q: wp.array(dtype=wp.float64), oldq: wp.array(dtype=wp.float64),
                        f: wp.array(dtype=wp.float64), oldf: wp.array(dtype=wp.float64),
                        work: wp.array(dtype=wp.float64), inverse: wp.array2d(dtype=wp.float64)):
    # work = s - H*y was computed before this launch. No thread reads another
    # matrix entry while it is being updated. Reject nonpositive/degenerate
    # secants: this changes the search direction only, never final acceptance.
    i,j=wp.tid()
    sy,yhy,ss,yy=wp.float64(0.),wp.float64(0.),wp.float64(0.),wp.float64(0.)
    for k in range(f.shape[0]):
        s,y=q[k]-oldq[k],f[k]-oldf[k]
        sy+=s*y; yhy+=y*(s-work[k]); ss+=s*s; yy+=y*y
    if wp.isfinite(sy) and wp.isfinite(yhy) and sy>wp.float64(1.e-10)*wp.sqrt(ss*yy) and sy>wp.float64(1.e-30):
        si,sj=q[i]-oldq[i],q[j]-oldq[j]
        hi,hj=si-work[i],sj-work[j]
        rho=wp.float64(1.)/sy
        inverse[i,j]+=(rho+rho*rho*yhy)*si*sj-rho*(hi*sj+si*hj)


@wp.kernel
def perturb(base: wp.array(dtype=wp.float64), column: int, amount: wp.float64,
            out: wp.array(dtype=wp.float64)):
    i=wp.tid();out[i]=base[i]
    if i==column:out[i]+=amount


@wp.kernel
def derivative(value: wp.array(dtype=wp.float64), base: wp.array(dtype=wp.float64),
               column: int, amount: wp.float64, matrix: wp.array2d(dtype=wp.float64)):
    i=wp.tid();matrix[i,column]=(value[i]-base[i])/amount


@wp.kernel
def invert(matrix: wp.array2d(dtype=wp.float64), inverse: wp.array2d(dtype=wp.float64),
           status: wp.array(dtype=wp.float64)):
    # Partial-pivot Gauss-Jordan for a small 6N interface system, never FEM.
    n=matrix.shape[0]
    for i in range(n):
        for j in range(n):inverse[i,j]=wp.float64(i==j)
    for k in range(n):
        pivot=k
        for i in range(k+1,n):
            if wp.abs(matrix[i,k])>wp.abs(matrix[pivot,k]):pivot=i
        if not wp.isfinite(matrix[pivot,k]) or wp.abs(matrix[pivot,k])<wp.float64(1.e-20):
            status[0]=wp.float64(wp.inf)
            return
        for j in range(n):
            temp=matrix[k,j];matrix[k,j]=matrix[pivot,j];matrix[pivot,j]=temp
            temp=inverse[k,j];inverse[k,j]=inverse[pivot,j];inverse[pivot,j]=temp
        divisor=matrix[k,k]
        for j in range(n):matrix[k,j]/=divisor;inverse[k,j]/=divisor
        for i in range(n):
            if i!=k:
                factor=matrix[i,k]
                for j in range(n):
                    matrix[i,j]-=factor*matrix[k,j]
                    inverse[i,j]-=factor*inverse[k,j]


@wp.kernel
def velocities(q: wp.array(dtype=wp.float64), radius: wp.array(dtype=wp.float64), dt: wp.float64,
               out: wp.array2d(dtype=wp.float64)):
    i,j=wp.tid();out[i,j]=q[6*i+j]/dt
    if j>=3:out[i,j]/=radius[i]
