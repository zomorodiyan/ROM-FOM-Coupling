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

#%% Define functions
#-----------------------------------------------------------------------------#
# Neural network Methods
#-----------------------------------------------------------------------------#
def create_training_data_lstm(serie, window_size):
    n_snapshots = serie.shape[0]
    n_states = serie.shape[1]
    ytrain = serie[window_size:,:]
    xtrain = np.zeros((n_snapshots-window_size, window_size, n_states))
    for i in range(n_snapshots-window_size):
        tmp = serie[i,:]
        for j in range(1,window_size):
            tmp = np.vstack((tmp,serie[i+j,:]))
        xtrain[i,:,:] = tmp
    return xtrain , ytrain

#%% Main program
if(0==0): #Load POD data
    nx = 16 # set in file 1*
    ny = int(nx/8) # set in file 1*
    #number_of_snapshots = 8 set in file 1* + 1 (the initial condition)
    #number_of_states(basis) = 5 set in file 1*
    folder = 'data_'+ str(nx) + '_' + str(ny)
    filename = './POD/'+folder+'/POD_data.npz'
    data = np.load(filename)
    aTrue = data['aTrue']
    print('aTrue.shape: ', aTrue.shape)

#%% Training
    n_states = aTrue.shape[1]
    window_size = 3
    xtrain, ytrain = create_training_data_lstm(serie=aTrue,window_size=window_size)

if(1==2): #Scaling data
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

if(2==2):
    #Shuffling data
    seed(1)
    tf.random.set_seed(0)
    perm = np.random.permutation(xtrain.shape[0]) # xtarin.shape[0] is (n_snapshots - window_size)
    xtrain = xtrain[perm,:,:]
    ytrain = ytrain[perm,:]
    #Create folder
    if os.path.isdir("./LSTM Model"):
        print('LSTM models folder already exists')
    else:
        print('Creating LSTM models folder')
        os.makedirs("./LSTM Model")
    #Removing old models
    model_name = 'LSTM Model/LSTM_Me_' + str(n_states) + '.h5'
    if os.path.isfile(model_name):
       os.remove(model_name)
    #create the LSTM architecture
    model = Sequential()
    model.add(LSTM(20, input_shape=(window_size, n_states), return_sequences=True, activation='tanh'))
    model.add(LSTM(20, input_shape=(window_size, n_states), activation='tanh'))
    model.add(Dense(n_states))
    #compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    #run the model
    history = model.fit(xtrain, ytrain, epochs=200, batch_size=64,
            validation_split=0.20, verbose = 0)
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
    filename = 'LSTM Model/loss_Me_' + str(n_states) + '.png'
    plt.savefig(filename, dpi = 200)
    plt.show()
    #Save the model
    model.save(model_name)
