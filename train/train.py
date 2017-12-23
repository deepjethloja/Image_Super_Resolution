

import numpy as np
import math
import os
import keras
from PIL import Image
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.initializers import RandomNormal

#%%
train_l=np.load('train_input.npy')
train_h=np.load('train_output.npy')
test_l=np.load('test_input.npy')
test_h=np.load('test_output.npy')

#%%


def PSNR_loss(y_true , y_pred):
    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)
    

#%% 
filters_1 = 64
filters_2 = 32
filters_3 = 3

strides_1 = 9
strides_2 = 1
strides_3 = 5

adam =keras.optimizers.Adam(lr=0.0003)

init_filter=RandomNormal(mean=0.0,stddev=0.001)
 
model=Sequential()

model.add(Convolution2D(filters=filters_1,kernel_size=strides_1,padding="same",activation = "relu",use_bias=True,kernel_initializer=init_filter,bias_initializer='zeros', input_shape=(66,66,3)));

model.add(Convolution2D(filters=filters_2,kernel_size=strides_2,padding ="same",activation = "relu",use_bias=True,kernel_initializer=init_filter,bias_initializer='zeros'))

model.add(Convolution2D(filters=filters_3,kernel_size=strides_3,padding = "same",use_bias=True,kernel_initializer=init_filter,bias_initializer='zeros'))

model.compile(loss='mean_squared_error', optimizer=adam,metrics =[PSNR_loss])

#%%
backprops = 10

model.fit(train_l,train_h,epochs=backprops,batch_size=64,validation_data=(test_l,test_h))






