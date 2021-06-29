#%% Load POD data (the Phiw, Phis, and Phit are the first nr from the 2_Bous)
folder = 'data_'+ str(nx) + '_' + str(ny)
filename = './POD/'+folder+'/POD_data.npz'
data = np.load(filename)

wm = data['wm']
Phiw = data['Phiw']
sm = data['sm']
Phis = data['Phis']
tm = data['tm']
Phit = data['Phit']
aTrue = data['aTrue']
bTrue = data['bTrue']

#%% Select the first nr
nr = 20
Phiw = Phiw[:,:nr]
Phis = Phis[:,:nr]
Phit = Phit[:,:nr]
aTrue = aTrue[:,:nr]
bTrue = bTrue[:,:nr]

#%% Training
#Scaling data
m,n = ytrain.shape # m is number of training samples, n is number of output features
scalerOut = MinMaxScaler(feature_range=(-1,1))
scalerOut = scalerOut.fit(ytrain)
ytrain = scalerOut.transform(ytrain)

for k in range(lookback):
    if k == 0:
        tmp = xtrain[:,k,:]
    else:
        tmp = np.vstack([tmp,xtrain[:,k,:]])

scalerIn = MinMaxScaler(feature_range=(-1,1))
scalerIn = scalerIn.fit(tmp)
for i in range(m):
    xtrain[i,:,:] = scalerIn.transform(xtrain[i,:,:])

#Shuffling data
perm = np.random.permutation(m)
xtrain = xtrain[perm,:,:]
ytrain = ytrain[perm,:]

#create folder
if os.path.isdir("./LSTM Model"):
    print('LSTM models folder already exists')
else:
    print('Creating LSTM models folder')
    os.makedirs("./LSTM Model")

#Removing old models
model_name = 'LSTM Model/LSTM_CPI_' + str(nr) + '.h5'
if os.path.isfile(model_name):
   os.remove(model_name)


#create the LSTM architecture
model = Sequential()
model.add(LSTM(20, input_shape=(lookback, features.shape[1]), return_sequences=True, activation='tanh'))
model.add(LSTM(20, input_shape=(lookback, features.shape[1]), activation='tanh'))
model.add(Dense(labels.shape[1]))

#compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

#run the model
history = model.fit(xtrain, ytrain, epochs=200, batch_size=64, validation_split=0.20)

#evaluate the model
scores = model.evaluate(xtrain, ytrain, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
epochs = range(1, len(loss) + 1)
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
filename = 'LSTM Model/loss_CPI_' + str(nr) + '.png'
plt.savefig(filename, dpi = 200)
plt.show()

#Save the model
model.save(model_name)
#Save the scales
filename = 'LSTM Model/input_scaler_CPI_' + str(nr) + '.save'
joblib.dump(scalerIn,filename)
filename = 'LSTM Model/output_scaler_CPI_' + str(nr) + '.save'
joblib.dump(scalerOut,filename)


#%% Testing
# Load DPI data
folder = 'data_'+ str(nx) + '_' + str(ny)
filename = './DPI/'+folder+'/DPI_data_nr='+str(nr)+'.npz'
data = np.load(filename)
aDPI = data['aDPI']
bDPI = data['bDPI']

#%% Initialize
nstart= 0
nend = nt
nstep = 1

ns = int((nend-nstart)/nstep)
print(ns)

w,s,t = import_data(nx,ny,nstart)

aCPI = np.zeros([ns+1,nr])
bCPI = np.zeros([ns+1,nr])

tmp = w.reshape([-1,])-wm
aCPI[0,:] = PODproj_svd(tmp,Phiw)

tmp = t.reshape([-1,])-tm
bCPI[0,:] = PODproj_svd(tmp,Phit)

time=0
xtest = np.zeros((1,lookback,features.shape[1]))
tmp = np.copy(aCPI[0,:])
xtest[0,0,:] = scalerIn.transform(tmp.reshape((1,-1)))

for i in range(1,lookback):
    w,s,t = RK3(BoussRHS,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)
    tmp = w.reshape([-1,])-wm
    aT = PODproj_svd(tmp,Phiw)
    aCPI[i,:] = aT
    tmp = t.reshape([-1,])-tm
    bCPI[i,:] = PODproj_svd(tmp,Phit)

    tmp = np.copy(aT)
    xtest[0,i,:] = scalerIn.transform(tmp.reshape((1,-1)))

for i in range(lookback-1,ns):

    time = time+dt

    tmp = t.reshape([-1,])-tm
    bCPI[i,:] = PODproj_svd(tmp,Phit)

    # update xtest
    for ii in range(lookback-1):
        xtest[0,ii,:] = xtest[0,ii+1,:]

    tmp = np.copy(aDPI[i+1,:])
    xtest[0,lookback-1,:] = scalerIn.transform(tmp.reshape((1,-1)))

    ytest = model.predict(xtest)
    ytest = scalerOut.inverse_transform(ytest) # rescale
    aCPI[i+1,:] = aDPI[i+1,:] + ytest

    w = PODrec_svd(aCPI[i+1,:],Phiw) + wm
    s = PODrec_svd(aCPI[i+1,:],Phis) + sm

    w = w.reshape([nx+1,ny+1])
    s = s.reshape([nx+1,ny+1])

    t = RK3t(BoussRHS_t,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)

    if (i+1)%freq==0:
        export_data_CPI(nx,ny,nstart+i+1,w,s,t)

    u,v = velocity(nx,ny,dx,dy,s)
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    cfl = np.max([umax*dt/dx, vmax*dt/dy])

    if cfl >= 0.8:
        print('CFL exceeds maximum value')
        break

    if (i+1)%100==0:
        print(i+1, " ", time, " ", np.max(np.abs(w)), " ", cfl)

tmp = t.reshape([-1,])-tm
bCPI[i+1,:] = PODproj_svd(tmp,Phit)

#%% Save DATA
folder = 'data_'+ str(nx) + '_' + str(ny)
filename = './CPI/'+folder+'/CPI_data_nr='+str(nr)+'.npz'
np.savez(filename, aCPI = aCPI, bCPI = bCPI)
