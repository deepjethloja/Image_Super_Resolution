from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Convolution2D , MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
import numpy as np
import os

from PIL import Image
import tensorflow as tf

os.chdir('D:\data');

path1="BSD100"
path2="image_SRF_2"
k2=0
k1=0

clas = os.listdir(path1+'\\'+path2)
i=0
k=0

def crop(Path, input, height, width,area):
    im = Image.open(path1+'\\'+path2+'\\'+input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            global k1
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            o = a.crop(area)
            o.save(os.path.join("D:/data/training","IMG_LF%s.png" % k1))
            k1 +=1


def cro(Path, input, height, width,area):
    im = Image.open(path1+'\\'+path2+'\\'+input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            global k2
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            o = a.crop(area)
            o.save(os.path.join("D:/data/training","IMG_HF-%s.png" % k2))
            k2 = k2+1
            

for img in clas:
    if i%2==0:
        cro(clas,img,28,28,(0,0,28,28))
    if i%2==1:
        crop(clas,img,14,14,(0,0,14,14))
    i +=1
    
path1="Urban100"

for img in clas:
    if i%2==0:
        cro(clas,img,28,28,(0,0,28,28))
    if i%2==1:
        crop(clas,img,14,14,(0,0,14,14))
    i +=1

path1="Set14"
path2="image_SRF_2"
k2=0
k1=0

clas = os.listdir(path1+'\\'+path2)
i=0
k=0

def crop(Path, input, height, width,area):
    im = Image.open(path1+'\\'+path2+'\\'+input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            global k1
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            o = a.crop(area)
            o.save(os.path.join("D:/data/validation","IMG_LF%s.png" % k1))
            k1 +=1


def cro(Path, input, height, width,area):
    im = Image.open(path1+'\\'+path2+'\\'+input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            global k2
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            o = a.crop(area)
            o.save(os.path.join("D:/data/validation","IMG_HF-%s.png" % k2))
            k2 = k2+1
    for img in clas:
    if i%2==0:
        cro(clas,img,28,28,(0,0,28,28))
    if i%2==1:
        crop(clas,img,14,14,(0,0,14,14))
    i +=1