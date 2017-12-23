from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
import numpy as np
import os
import inspect
from PIL import Image

dir = os.path.dirname(os.path.realpath(os.path.abspath(inspect.stack()[0][1])))
os.chdir(dir);

path = "data"
path1 = "BSD100"
path2= "image_SRF_2"

classes = os.listdir(path+'\\'+path1+'\\'+path2)

import scipy.misc
i=0
sub_input_image_size=33
sub_output_image_size=66
input_stride=14
output_stride=28
train_input=[]
train_output=[]
def crop_input(Path, input, height, width, area,stride,test):
    im = Image.open(path+'\\'+path1+'\\'+path2+'\\'+input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,stride):
        for j in range(0,imgwidth,stride):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            o = a.crop(area)
            imr=o.convert(mode ="RGB")
            imr=scipy.misc.imresize(imr,(sub_output_image_size,sub_output_image_size),interp="bicubic")
            imrs=img_to_array(imr)/255;
            imrs=imrs.reshape(sub_output_image_size,sub_output_image_size,3);
            test.append(imrs)
            
def crop_output(Path, input, height, width, area,stride,test):
    im = Image.open(path+'\\'+path1+'\\'+path2+'\\'+input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,stride):
        for j in range(0,imgwidth,stride):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            o = a.crop(area)
            im=o.convert(mode ="RGB")
            im=im.resize((sub_output_image_size,sub_output_image_size))
            imrs=img_to_array(im)/255;
            imrs=imrs.reshape(sub_output_image_size,sub_output_image_size,3);
            test.append(imrs)
            
for img in classes:
    if i%2==0:
        crop_output(classes,img,sub_output_image_size,sub_output_image_size,(0,0,sub_output_image_size,sub_output_image_size),output_stride,train_output)
    if i%2==1:
        crop_input(classes,img,sub_input_image_size,sub_input_image_size,(0,0,sub_input_image_size,sub_input_image_size),input_stride,train_input)
    i=i+1
train_input=np.array(train_input)
train_output=np.array(train_output)

path1="Set14"
classes = os.listdir(path+'\\'+path1+'\\'+path2)
test_input=[]
test_output=[]
for img in classes:
    if i%2==0:
        crop_output(classes,img,sub_output_image_size,sub_output_image_size,(0,0,sub_output_image_size,sub_output_image_size),output_stride,test_output)
    if i%2==1:
        crop_input(classes,img,sub_input_image_size,sub_input_image_size,(0,0,sub_input_image_size,sub_input_image_size),input_stride,test_input)
    i=i+1
test_input=np.array(test_input)
test_output=np.array(test_output)

np.save('train/train_input.npy',train_input)
np.save('train/train_output.npy',train_output)
np.save('train/test_input.npy',test_input)
np.save('train/test_output.npy',test_output)