if(0==0): #%% Import libraries
    import numpy as np
    from numpy.random import seed
    import matplotlib.pyplot as plt
    import os
    from silence_tensorflow import silence_tensorflow
    silence_tensorflow()
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM
    from sklearn.preprocessing import MinMaxScaler
    import joblib

if(1==1):#%% Define functions
    def import_data(nx,ny,n): (modify)
        folder = 'data_'+ str(nx) + '_' + str(ny)
        filename = './Results/'+folder+'/data_' + str(int(n))+'.npz'
        data = np.load(filename)
        w = data['w']
        s = data['s']
        t = data['t']
        return w,s,t
    def PODproj_svd(u,Phi): #Projection
        a = np.dot(u.T,Phi)  # u = Phi * a.T
        return a
    def PODrec_svd(a,Phi): #Reconstruction
        u = np.dot(Phi,a.T)
        return u
    def jacobian(nx,ny,dx,dy,q,s): # arakawa scheme
    # computed at all internal physical domain points (1:nx-1,1:ny-1)
        gg = 1.0/(4.0*dx*dy)
        hh = 1.0/3.0
        #Arakawa 1:nx,1:ny
        j1 = gg*( (q[2:nx+1,1:ny]-q[0:nx-1,1:ny])*(s[1:nx,2:ny+1]-s[1:nx,0:ny-1]) \
                 -(q[1:nx,2:ny+1]-q[1:nx,0:ny-1])*(s[2:nx+1,1:ny]-s[0:nx-1,1:ny]))

        j2 = gg*( q[2:nx+1,1:ny]*(s[2:nx+1,2:ny+1]-s[2:nx+1,0:ny-1]) \
                - q[0:nx-1,1:ny]*(s[0:nx-1,2:ny+1]-s[0:nx-1,0:ny-1]) \
                - q[1:nx,2:ny+1]*(s[2:nx+1,2:ny+1]-s[0:nx-1,2:ny+1]) \
                + q[1:nx,0:ny-1]*(s[2:nx+1,0:ny-1]-s[0:nx-1,0:ny-1]))

        j3 = gg*( q[2:nx+1,2:ny+1]*(s[1:nx,2:ny+1]-s[2:nx+1,1:ny]) \
                - q[0:nx-1,0:ny-1]*(s[0:nx-1,1:ny]-s[1:nx,0:ny-1]) \
                - q[0:nx-1,2:ny+1]*(s[1:nx,2:ny+1]-s[0:nx-1,1:ny]) \
                + q[2:nx+1,0:ny-1]*(s[2:nx+1,1:ny]-s[1:nx,0:ny-1]) )
        jac = (j1+j2+j3)*hh
        return jac
    def laplacian(nx,ny,dx,dy,w):
        aa = 1.0/(dx*dx)
        bb = 1.0/(dy*dy)
        lap = aa*(w[2:nx+1,1:ny]-2.0*w[1:nx,1:ny]+w[0:nx-1,1:ny]) \
            + bb*(w[1:nx,2:ny+1]-2.0*w[1:nx,1:ny]+w[1:nx,0:ny-1])
        return lap
    def RK3t(rhs,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt):
    # time integration using third-order Runge Kutta method
        aa = 1.0/3.0
        bb = 2.0/3.0

        tt = np.zeros([nx+1,ny+1])
        tt = np.copy(t)

        #stage-1
        rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,w,s,t)
        tt[1:nx,1:ny] = t[1:nx,1:ny] + dt*rt
        tt = tbc(tt)

        #stage-2
        rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,w,s,tt)
        tt[1:nx,1:ny] = 0.75*t[1:nx,1:ny] + 0.25*tt[1:nx,1:ny] + 0.25*dt*rt
        tt = tbc(tt)

        #stage-3
        rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,w,s,tt)
        t[1:nx,1:ny] = aa*t[1:nx,1:ny] + bb*tt[1:nx,1:ny] + bb*dt*rt
        t = tbc(t)
        return t
    def tbc(t):
        t[0,:] = t[1,:]
        t[-1,:] = t[-2,:]
        t[:,0] = t[:,1]
        t[:,-1] = t[:,-2]
        return t
    def BoussRHS_t(nx,ny,dx,dy,Re,Pr,Ri,w,s,t):
        rt = np.zeros([nx-1,ny-1]) #define
        Lt = laplacian(nx,ny,dx,dy,t) #laplacian terms
        Jt = jacobian(nx,ny,dx,dy,t,s) #Jacobian terms
        rt = -Jt + (1/(Re*Pr))*Lt # t-equation
        return rt
    def velocity(nx,ny,dx,dy,s):
    #compute velocity components from streamfunction (internal points)
        u =  np.zeros([nx-1,ny-1])
        u = (s[1:nx,2:ny+1] - s[1:nx,0:ny-1])/(2*dy) # u = ds/dy
        v =  np.zeros([nx-1,ny-1])
        v = -(s[2:nx+1,1:ny] - s[0:nx-1,1:ny])/(2*dx) # v = -ds/dx
        return u,v

#%% Main program
if(2==2): # Inputs
    lx = 8 #length in x direction
    ly = 1 #length in y direction
    nx = 128 #number of meshes in x direction
    ny = int(nx/8) #number of meshes in y direction

    Re = 1e4 #Reynolds Number: inertial/viscous
    Ri = 4 #Richardson Number: Buoyancy/flow_shear
    Pr = 1 #Prandtl Number: momentum_diffusivity/thermal_diffusivity

    Tm = 8 #maximum time
    dt = 5e-4 #time_step_size
    nt = np.int(np.round(Tm/dt)) #number of time_steps

    ns = 800 #number of snapshots
    freq = np.int(nt/ns) #every freq time_stap we export data

if(3==3):#%% grid
    dx = lx/nx
    dy = ly/ny
    x = np.linspace(0.0,lx,nx+1)
    y = np.linspace(0.0,ly,ny+1)
    X, Y = np.meshgrid(x, y, indexing='ij')
if(4==4):# Load DPI data (modify)
    folder = 'data_'+ str(nx) + '_' + str(ny)
    filename = './DPI/'+folder+'/DPI_data_nr='+str(nr)+'.npz'
    data = np.load(filename)
    aDPI = data['aDPI']
    bDPI = data['bDPI']
if(5==5):#%% Initialize
    time=0
    nstep = 1
    ns = int(nt/nstep)
    alpha = np.zeros([ns+1,nr])
    beta = np.zeros([ns+1,nr])

    w,s,t = import_data(nx,ny,0)
    tmp = w.reshape([-1,])-wm
    alpha[0,:] = PODproj_svd(tmp,Phiw)
    tmp = t.reshape([-1,])-tm
    beta[0,:] = PODproj_svd(tmp,Phit)
    xtest = np.zeros((1,lookback,features.shape[1]))
    tmp = np.copy(alpha[0,:])
    xtest[0,0,:] = scalerIn.transform(tmp.reshape((1,-1)))

for i in range(1,lookback):
    w,s,t = RK3(BoussRHS,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)
    tmp = w.reshape([-1,])-wm
    aT = PODproj_svd(tmp,Phiw)
    alpha[i,:] = aT
    tmp = t.reshape([-1,])-tm
    beta[i,:] = PODproj_svd(tmp,Phit)

    tmp = np.copy(aT)
    xtest[0,i,:] = scalerIn.transform(tmp.reshape((1,-1)))

for i in range(lookback-1,ns):
    time = time+dt
    tmp = t.reshape([-1,])-tm
    beta[i,:] = PODproj_svd(tmp,Phit)

    # update xtest
    for ii in range(lookback-1):
        xtest[0,ii,:] = xtest[0,ii+1,:]

    tmp = np.copy(aDPI[i+1,:])
    xtest[0,lookback-1,:] = scalerIn.transform(tmp.reshape((1,-1)))

    ytest = model.predict(xtest)
    ytest = scalerOut.inverse_transform(ytest) # rescale
    alpha[i+1,:] = aDPI[i+1,:] + ytest

    w = PODrec_svd(alpha[i+1,:],Phiw) + wm
    s = PODrec_svd(alpha[i+1,:],Phis) + sm

    w = w.reshape([nx+1,ny+1])
    s = s.reshape([nx+1,ny+1])

    t = RK3t(BoussRHS_t,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)

    if (i+1)%freq==0:
        export_data_CPI(nx,ny,i+1,w,s,t)

    u,v = velocity(nx,ny,dx,dy,s)
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    cfl = np.max([umax*dt/dx, vmax*dt/dy])

    if cfl >= 0.8:
        print('CFL exceeds maximum value')
        break

    if (i+1)%100==0:
        print(i+1, " time: ", time, " max(w): ", np.max(np.abs(w)), " cfl: ", cfl)

tmp = t.reshape([-1,])-tm
beta[i+1,:] = PODproj_svd(tmp,Phit)

#%% Save DATA
folder = 'data_'+ str(nx) + '_' + str(ny)
filename = './CPI/'+folder+'/CPI_data_nr='+str(nr)+'.npz'
np.savez(filename, alpha = alpha, beta = beta)

