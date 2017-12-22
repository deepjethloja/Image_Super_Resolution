import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Convolution2D , MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
import numpy as np
import os
import math

from PIL import Image
import tensorflow as tf

os.chdir('D:\data');

path1="training"
classes=os.listdir(path1)
train_l=[]
train_h=[]
test_l=[]
test_h=[]

import scipy.misc
for i in range(0,48362):
    im=Image.open('D:/data/training/IMG_HF-%s.png'%i)
    im=im.convert(mode="RGB")
    im=im.resize((28,28))
    imrs=img_to_array(im);
    #imrs=imrs.transpose(2,0,1);
    imrs=imrs.reshape(28,28,3);
    #imrs_crop = imrs[:,6:22,6:22]
    train_h.append(imrs)
    
    im=Image.open('D:/data/training/IMG_LF%s.png'%i)    
    im=im.convert(mode="RGB")
    im=scipy.misc.imresize(im,(28,28),interp="bicubic")
    imrs1=img_to_array(im);
    #imrs1=imrs1.transpose(2,0,1);
    imrs1=imrs1.reshape(28,28,3);
    train_l.append(imrs1)
    
for i in range(0,4332):
    im=Image.open('D:/data/validation//IMG_HF-%s.png'%i)
    im=im.convert(mode="RGB")
    im=im.resize((28,28))
    imrs=img_to_array(im);
    #imrs=imrs.transpose(2,0,1);
    imrs=imrs.reshape(28,28,3);
    #imrs_crop = imrs[:,6:22,6:22]
    test_h.append(imrs)
    
    im=Image.open('D:/data/validation//IMG_LF%s.png'%i)    
    im=im.convert(mode="RGB")
    im=scipy.misc.imresize(im,(28,28),interp="bicubic")
    imrs1=img_to_array(im);
    #imrs1=imrs1.transpose(2,0,1);
    imrs1=imrs1.reshape(28,28,3);
    test_l.append(imrs1)
  
test_l=np.array(test_l)
test_h=np.array(test_h)
train_l=np.array(train_l)
train_h=np.array(train_h)

from math import log10
from keras import backend as K

def PSNR_loss(y_true , y_pred):
    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)
    
adam =keras.optimizers.Adam(lr=0.001 , beta_1 = 0.9 , beta_2 =0.999 , epsilon = 1e-8)

smodel=Sequential()
model.add(Convolution2D(filters=64,kernel_size=(9,9),strides=(1,1),padding="same",input_shape=(28,28,3)));
model.add(Activation('relu'));
model.add(Convolution2D(filters=32,kernel_size=1,padding ="same"))
model.add(Activation('relu'));
model.add(Convolution2D(filters=3,kernel_size=5,padding = "same"))
model.compile(loss='mean_squared_error', optimizer=adam,metrics =[PSNR_loss])

model.fit(train_l,train_h,epochs=1000000,verbose=1,validation_data=(test_l,test_h),batch_size=32)