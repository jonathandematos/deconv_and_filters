from __future__ import print_function
import keras
from keras import applications
from keras import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Conv2D, Conv2DTranspose
from scipy.misc import imsave
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import numpy as np
import time
from keras import backend as K
import sys
from keras.initializers import Initializer
from keras.engine.topology import Layer

import tensorflow as tf

if(len(sys.argv) != 2):
	exit(0)


# layer_activations
def Unpool2D(ksize=(3,3), stride=(2,2), input_data=None):
    
    # fix kernel size problem
    
    xk = ksize[0]
    yk = ksize[1]
    
    xst = stride[0]
    yst = stride[1]
    
    j = int(xk/2)
    # stride on X
    idx = list()
    value = list()
    while( j < layer_activations.shape[0] and (j + int(xk/2)) < layer_activations.shape[0] ):
        k = int(yk/2)
        # stride on Y
        idx_line = list() 
        value_line = list()
        while( k < layer_activations.shape[1] and (k + int(yk/2)) < layer_activations.shape[1] ):
            #print("{:.5f} ".format(layer_activations[j,k,i]), end="")
            maxval = -1000000
            maxpos = [0,0]
            jj = 0
            posjj = j-int(3/2)
            poskk = k-int(3/2)
            # kernel convolution X
            while(jj < xk):
                kk = 0
                # kernel convolution Y
                while(kk < yk):
                    #print("[{} {}] {} ".format(posjj+jj, poskk+kk, input_data[posjj+jj][poskk+kk]), end="")
                    if(input_data[posjj+jj][poskk+kk] > maxval):
                        maxpos[0] = posjj+jj
                        maxpos[1] = poskk+kk
                        maxval = input_data[posjj+jj][poskk+kk]
                    kk += 1
                #print()
                jj += 1
            #print("{} {} {}".format(maxpos[0], maxpos[1], maxval))
            idx_line.append(maxpos)
            value_line.append(maxval)
            k += xst
        j += yst
        idx.append(idx_line)
        value.append(value_line)
    return np.array(idx), np.array(value)
    


def Reconstruct(idx, value):
    
    img = np.zeros((idx.shape[0]*2,idx.shape[1]*2))
    
    print(idx.shape)
    
    for i in range(idx.shape[0]-1):
        for j in range(idx.shape[1]-1):
            img[idx[i][j][0]][idx[i][j][1]] = value[i][j]
    
    return img


class WeightCopy(Initializer):

    def __init__(self, model=None, layer=None):
        self.model = model
        self.layer = layer

    def __call__(self, shape, dtype=None):
        return self.model.get_layer(self.layer).get_weights()[0]

    def get_config(self):
        return {
            'model': self.model,
            'layer': self.layer
        }

class BiasCopy(Initializer):

    def __init__(self, model=None, layer=None):
        self.model = model
        self.layer = layer

    def __call__(self, shape, dtype=None):
        #return self.model.get_layer(self.layer).get_weights()[1]
        return np.array([1,1,1])

    def get_config(self):
        return {
            'model': self.model,
            'layer': self.layer
        }


input_img_name = sys.argv[1]

resnet = keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)
x = resnet.get_layer(name="avg_pool").output #output
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(50, activation="relu")(x)
x = Dense(2, activation='softmax', name="dense2")(x)
model = Model(inputs=resnet.inputs, outputs=x)
model.load_weights("lala.hdf5")

for i in model.layers:
    print("{} -> {} -> {}".format(i.name, i.__class__.__name__, i.output.shape))
    print()
    print(i.get_config())
    print(i.output_shape)
    print(i.input_shape)
    print("----------------------------------------")
    
print("----------------------------")    

conv_layers = list()

model2 = Sequential()
wi = WeightCopy(model=model, layer="conv1")
bi = BiasCopy(model=model, layer="conv1")

y = resnet.get_layer(name="max_pooling2d_1").output
y = Conv2DTranspose(3, (7,7), kernel_initializer=wi, bias_initializer=bi, input_shape=(55,55,64))(y)
model2 = Model(inputs=resnet.inputs, outputs=y)

input_img = model.input


'''
for i in model.layers:
    print("{} -> {} -> {}".format(i.name, i.__class__.__name__, i.output.shape))
    
print("----------------------------")    
'''    
for i in model2.layers:
    #if( i.__class__.__name__ == "Conv2D" or i.__class__.__name__ == "Conv2DTranspose" ):
    if( i.__class__.__name__ == ""):
#        print("{} -> {} -> {}".format(i.name, i.__class__.__name__, i.output.shape))
        conv_layers.append(i.name)
#        print(model.get_layer(i.name).get_weights()[1].shape)
#        print(model.get_layer(i.name).get_config())


conv_layers = ["activation_1"]

model2 = model

#model2.add(Conv2DTranspose(3, (7,7), kernel_initializer=wi, bias_initializer=bi, input_shape=(55,55,64)))

ct = 0
for layer_name in conv_layers:
    
    ct += 1
    func = K.function([input_img], [model2.get_layer(layer_name).output])
    #func = K.function([input_img], [model2.model.get_layer(layer_name).output])

    img = image.load_img(input_img_name, target_size=(224,224))
    input_img_data = np.array([img_to_array(img)]).astype('float32')/255

    layer_outputs = func([input_img_data])[0]




    func = K.function([input_img], [model2.get_layer("max_pooling2d_1").output])
    #func = K.function([input_img], [model2.model.get_layer(layer_name).output])

    img = image.load_img(input_img_name, target_size=(224,224))
    input_img_data = np.array([img_to_array(img)]).astype('float32')/255

    layer_outputs_max = func([input_img_data])[0]





    activations = list()

    print_shape_only = True

    std_width = 400
    
    print("{}/{} ".format(ct, len(conv_layers)), end="")
    
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            ct += 1
            qtd_width = std_width / layer_activations.shape[1]
            final_width = ((qtd_width-1) * 3) + (layer_activations.shape[1]) * qtd_width
            qtd_height = layer_activations.shape[2] / qtd_width
            final_height = ((qtd_height-1) * 3) + qtd_height * layer_activations.shape[0]

            #print(layer_activations.shape)
            #print("{}x{}   {}x{}".format(final_width, final_height, qtd_width, qtd_height))
            big_image = np.zeros((final_height,final_width))
            img_width = layer_activations.shape[1]
            img_height = layer_activations.shape[0]
            t = 0
            t=0
            for i in range(qtd_width):
                for j in range(qtd_height):
                    img = layer_activations[:,:,t]
                    img2 = layer_outputs_max[0,:,:,0]
                    #img += img.min()*(-1)
                    img = np.clip(img,0,img.max())
                    if(img.max() > 0):
                        img /= img.max()
                    img *= 255
                    img = np.clip(img,0,255).astype('uint8')


                    img2 = np.clip(img2,0,img2.max())
                    if(img2.max() > 0):
                        img2 /= img2.max()
                    img2 *= 255
                    img2 = np.clip(img2,0,255).astype('uint8')


                    #print(img.max(), img.min())
                    #print((img_height + 3) * j, (img_height + 3) * j + img_height)
                    big_image[(img_height + 3) * j: (img_height + 3) * j + img_height,
                            (img_width + 3) * i: (img_width + 3) * i + img_width] = img
                    t+=1
                    idx, value = Unpool2D(input_data = img)
                    img_rec = Reconstruct(idx, value)
                    img_rec2 = Reconstruct(idx, img2)
                    imsave('convolucao_maxpool_{}.png'.format(layer_name), img2)
                    imsave('convolucao_pospool_{}.png'.format(layer_name), value)
                    imsave('convolucao_reconstruct_{}.png'.format(layer_name), img_rec)
                    imsave('convolucao_reconstruct2_{}.png'.format(layer_name), img_rec2)
                    imsave('convolucao_{}.png'.format(layer_name), img)
                    exit(0)

        else:
            print("output")
#            for i in range(layer_activations.shape[2]):
#                print(tf.nn.max_pool_with_argmax(input=np.array([layer_activations[:,:,:]]), ksize=(1,3,3,1), strides=(1,2,2,1), padding="VALID"))
                
                
            
