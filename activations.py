#!/usr/bin/python
#
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import sys
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib.pyplot import imsave, imread, imshow
import cv2
#
colormap = cm.get_cmap('viridis')
#
model = load_model("models/unpooling")
#
input_img_name = sys.argv[1]
img = image.load_img(input_img_name, target_size=(224,224))
input_img_data = np.array([img_to_array(img)]).astype('float32')/255
#
func = K.function([model.input] , [model.get_layer("conv3").output])
prediction = K.function([model.input] , [model.output])
#
layer_output = np.moveaxis(func([input_img_data])[0][0], [-1],[0])
#
print(layer_output.shape)
#
activations = np.zeros((layer_output.shape[1],layer_output.shape[2],4))
i_float = np.zeros((layer_output.shape[1],layer_output.shape[2]))
#
for i in range(layer_output.shape[0]):
    i_float += layer_output[i].astype("float32")/layer_output.shape[0]
print(input_img_data.shape)

resized = cv2.resize(colormap((i_float-i_float.min())/(i_float.max()-i_float.min()))[:,:,:3], dsize=(input_img_data.shape[1],input_img_data.shape[2]), interpolation=cv2.INTER_NEAREST)
alpha = 0.7
tmp_img = (resized*alpha)+(input_img_data[0,:,:,:]*(1-alpha))
#
#
#
print(tmp_img.shape)
imsave("teste.png", tmp_img)
side_by_side = np.zeros(((input_img_data.shape[1],input_img_data.shape[2]*3,3)))
imsave("teste_side.png", side_by_side)
print(side_by_side.shape)
side_by_side[0:input_img_data.shape[1],0:input_img_data.shape[2],:] = input_img_data[0,:,:,:]
side_by_side[0:input_img_data.shape[1],input_img_data.shape[2]:input_img_data.shape[2]*2,:] = tmp_img
side_by_side[0:input_img_data.shape[1],input_img_data.shape[2]*2:input_img_data.shape[2]*3,:] = resized
imsave("teste_side.png", side_by_side)
print(prediction([input_img_data])[0])
#
#
#
exit(0)
