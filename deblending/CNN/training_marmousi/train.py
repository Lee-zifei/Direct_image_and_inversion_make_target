#======================================================================================
#    Copyright (C) 2019 Chengdu University of Technology.
#    Copyright (C) 2019 Shaohuan Zu.
#
#    Filename:     ztrain3.py
#    Author：      Shaohuan Zu
#    Institute：   Chengdu University of Technology
#    Email：       zushaohuan19@cdut.edu.cnmodule named 'keras'
#    Date：        2019/12/13    10:16:25
#    Function：
#
#    This program is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation, either version 3 of the License, or any later version.
#===================================================================================
import numpy as np
import os
import sys, os, platform
if 'macos' in platform.platform().lower(): 
    myprog_path='/Users/lzf/Documents/cdut_zsh_group/python' 
elif 'linux' in platform.platform().lower(): 
    myprog_path='/media/lzf/Work/code/python' 
    myprog_path_survey='/home/lzf/code/python' 
else: 
    myprog_path='L:\data\code\python' 
sys.path.append(myprog_path)
sys.path.append(myprog_path_survey)
from subfunctions import normalize_patches,denormalize_patches,seis

########################load module#################################
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers import Input
from scipy import io
from keras.utils import plot_model
import time
import random
from sklearn.model_selection import train_test_split
import numpy as np
from  tensorflow.python.client import device_lib
from myprog import *
from keras import backend as K
import numpy as np
import time
import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF #
import matplotlib.pyplot as plt
import time
import os
import argparse
from pynvml import *
import tensorflow as tf
from subfunctions import normalize_input_and_label

########################obtain parameters#################################
parser = argparse.ArgumentParser(description='read parameters')
parser.add_argument ('--model', default=3, type=int,help='Setting model')
parser.add_argument ('--epoch', default=100, type=int,help='Training epoch')
parser.add_argument ('--abs', default=10, type=int,help='Noise abs')
args = parser.parse_args()
kk = args.model
epochs = args.epoch
# Abs = args.abs

########################set CPU information#################################

# nvmlInit()
# GpuCount = nvmlDeviceGetCount()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# KTF.set_session(sess )
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
    # tf.config.experimental.set_memory_growth(device, True)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.snrs = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.snrs.append(logs.get('snr'))

def snr(y_true, y_pred):
    loss1 = K.sum(K.sum(K.square(y_pred - y_true),axis=0),axis=1)
    loss2 = K.sum(K.sum(K.square(y_true),axis=0),axis=1)
    snr = 10*K.log(loss2/loss1)
    return snr



########################set data parameters ###############################
batch_size = 8       # batch size for training in each step
n1=330;                # the training volumn size   nz
n2=2000;                # the training volumn size   nt
n3=300;                # the training volumn size   nx
patch_rows = 128        # patch size
patch_cols= 128         # patch size
tslide=128              # time stride
xslide=128              # space stride
abs_list = ['10']
for Abs in abs_list:
        
    #filename1 = 'Data/obser.dat';
    filename1 = f'../Data_marmousi/obser_{Abs}.dat';
    inp1 = np.fromfile(filename1, dtype='float32')
    inp1 = inp1.reshape(n1*n3,n2).T

    #filename1 = 'Data/obser2.dat';
    filename2 = f'../Data_marmousi/obser2_{Abs}.dat';
    inp2 = np.fromfile(filename2, dtype='float32')
    inp2 = inp2.reshape(n1*n3,n2).T

    #filename1 = 'Data/hyper.dat';
    filename3 = '../Data_marmousi/hyper.dat';
    out1 = np.fromfile(filename3, dtype='float32')
    out1 = out1.reshape(n1*n3,n2).T

    #filename1 = 'Data/hyper2.dat';
    filename4 = '../Data_marmousi/hyper2.dat';
    out2 = np.fromfile(filename4, dtype='float32')
    out2 = out2.reshape(n1*n3,n2).T

    inputs = np.concatenate((inp1,inp2), axis=1)
    outputs = np.concatenate((out1,out2), axis=1)
    


    train1 = myimtocol(inputs, patch_rows, patch_rows,  n2, n1*n3, tslide,xslide,1);
    dev1 = myimtocol(outputs, patch_rows, patch_rows,  n2, n1*n3, tslide,xslide,1);

    train1, dev1, norm_stats = normalize_input_and_label(train1, dev1)
    print(np.max(train1))
    print(np.min(dev1))
    
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(train1[1,:,:])
    # plt.subplot(122)
    # plt.imshow(dev1[1,:,:])
    # plt.show()
    # #
    ########################splite the training data into two part #################################
    ######################## one is for training, the other is for testing#################################

    trainsize = 0.9
    seed = 20250511
    X_train, X_dev, Y_train, Y_dev = train_test_split(train1, dev1, train_size=trainsize, random_state=seed)
    
    train_datasize = int(trainsize*train1.shape[0])
    dev_datasize =  train1.shape[0] - train_datasize

    X_train = X_train.reshape(train_datasize, patch_rows, patch_cols, 1)
    X_dev = X_dev.reshape(dev_datasize, patch_rows, patch_cols, 1)
    Y_train = Y_train.reshape(train_datasize, patch_rows, patch_cols, 1)
    Y_dev = Y_dev.reshape(dev_datasize, patch_rows, patch_cols, 1)

    ########################create the model#################################
    ## snr is to evaluate the training process


    model = Sequential()
    input_data = Input(shape=(patch_rows, patch_cols, 1))

    conv2dlayers= 4
    deconv2dlayers= 2
    for i in range(conv2dlayers):
        if i==0:
            conlayer = Conv2D(32,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='glorot_uniform')(input_data)
        else:
            conlayer = Conv2D(32,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='glorot_uniform')(conlayer)
        conlayer = BatchNormalization()(conlayer)


    for j in range(deconv2dlayers):
        if j ==0:
            delayer = Conv2DTranspose(32,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='glorot_uniform')(conlayer)
        else:
            delayer = Conv2DTranspose(32,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='glorot_uniform')(delayer)


    output_data = Conv2DTranspose(1,
                kernel_size=(1, 1),
                padding='same',
                activation='tanh',
                kernel_initializer='glorot_uniform')(delayer)

    model = Model(inputs=input_data, outputs=output_data)
    print(model.summary())
    print ('##################################################################################\n')
    print ('########################Training model is {}######################################\n'.format(Abs))
    print ('##################################################################################\n')



    loss = keras.losses.mean_squared_error    # loss function
    optimizers = keras.optimizers.Adadelta()  # optimization algorithm,
                                            # an Adaptive Learning Rate Method is used here, so no need to set training rate ;-)
    metrics=['mean_squared_error']            # metrics:    a function that is used to judge the performance of your model,
                                            # mean squared error is used here

    ######## specify the inversion algorithm ########
    model.compile(loss=loss,
                optimizer=optimizers,
                metrics= [snr])

    ########################training the model#################################
    history = LossHistory()
    start = time.time()
    model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_dev, Y_dev),
            callbacks=[history])
    end = time.time()



    ########################save the trained model and error#################################
    plot_model(model, to_file='modelccn%d.png'%kk, show_shapes=True)
    model.save(f"model_{Abs}_%d.h5"%kk)
    print("Total time=", end-start)

    io.savemat(f"./loss_{Abs}_%d.mat"%kk, {'loss':history.losses})
    io.savemat(f"./snrs_{Abs}_%d.mat"%kk, {'snrs':history.snrs})
    io.savemat(f"./time_{Abs}_%d.mat"%kk, {'time':end-start})




