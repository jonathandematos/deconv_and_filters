from __future__ import print_function
import keras
from keras import applications
from keras import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D
from keras.optimizers import SGD
from scipy.misc import imsave
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import numpy as np
import time
from keras.models import load_model
from keras import backend as K
import sys
from keras.initializers import Initializer
from keras.engine.topology import Layer
from breakhis_generator_validation import LoadBreakhisList, Generator, GeneratorImgs, ReadImgs
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau, TensorBoard
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report



import tensorflow as tf

if(len(sys.argv) != 2):
	exit(0)


print(K.image_data_format())



def FindSwitches(ksize=(2,2), stride=None, input_data=None):
    
    if(stride == None):
        stride = ksize
    
    xk = ksize[0]
    yk = ksize[1]
    
    xst = stride[0]
    yst = stride[1]
   
    idx = list()
    value = list()
    
    j = 0    
    while(j+xk-1 < input_data.shape[0] ):
        idx_line = list() 
        value_line = list()
        
        k=0
        while(k+yk-1 < input_data.shape[1] ):    
            maxval = -1000000
            maxpos = [0,0]
            jj = 0
            posjj = j
            poskk = k
            # kernel convolution X
            #print(posjj+xk-1)
            while(jj < xk and posjj+xk-1 < input_data.shape[0]):
                kk = 0
                #print(poskk+yk-1)
                while(kk < yk and poskk+yk-1 < input_data.shape[1]):
                    if(input_data[posjj+jj][poskk+kk] > maxval):
                        maxpos[0] = posjj+jj
                        maxpos[1] = poskk+kk
                        maxval = input_data[posjj+jj][poskk+kk]
                    kk += 1
                    poskk += 1
                posjj += 1
                jj += 1
            idx_line.append(maxpos)
            value_line.append(maxval)
            k += yst
        j += xst
        idx.append(idx_line)
        value.append(value_line)
    return np.array(idx), np.array(value)
    


def Unpool(idx, value, ksize=(2,2), stride=(2,2)):

    img = np.zeros((idx.shape[0]*2,idx.shape[1]*2))
     
    for i in range(idx.shape[0]-1):
        for j in range(idx.shape[1]-1):
            img[idx[i][j][0]][idx[i][j][1]] = value[i][j]
    
    return img



def build_cnn(nr_convs):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), name="conv1", activation='relu', input_shape=(224,224,3)))
    model.add(BatchNormalization(axis=3, name="batch1"))
    model.add(MaxPooling2D(pool_size=(2,2), name="pool1"))
    
    if(nr_convs > 1):
        model.add(Conv2D(16, (5, 5), name="conv2", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch2"))
        model.add(MaxPooling2D(pool_size=(2,2), name="pool2"))

    if(nr_convs > 2):
        model.add(Conv2D(64, (5, 5), name="conv3", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch3"))
        model.add(MaxPooling2D(pool_size=(2,2), name="pool3"))

    if(nr_convs > 3):
        model.add(Conv2D(32, (3, 3), name="conv4", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch4"))
        model.add(MaxPooling2D(pool_size=(2,2), name="pool4"))
        
    if(nr_convs > 4):
        model.add(Conv2D(32, (3, 3), name="conv5", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch5"))
        model.add(MaxPooling2D(pool_size=(2,2), name="pool5"))

    if(nr_convs > 5):
        model.add(Conv2D(32, (3, 3), name="conv6", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch6"))
        model.add(MaxPooling2D(pool_size=(2,2), name="pool6"))

    if(nr_convs > 6):
        model.add(Conv2D(16, (3, 3), name="conv7", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch7"))
        model.add(MaxPooling2D(pool_size=(2,2), name="pool7"))

    if(nr_convs > 7):
        model.add(Conv2D(8, (3, 3), name="conv8", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch8"))
        model.add(MaxPooling2D(pool_size=(2,2), name="pool8"))

    if(nr_convs > 8):
        model.add(Conv2D(8, (3, 3), name="conv9", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch9"))
        model.add(MaxPooling2D(pool_size=(2,2), name="pool9"))

    #model.add(Dropout(0.25))
    #
    model.add(Flatten())

    model.add(Dense(64, activation='relu', name='dense1'))
    model.add(Dropout(0.25))

    model.add(Dense(32, activation='relu', name='dense2'))

    model.add(Dense(16, activation='relu', name='dense3'))

    model.add(Dense(2, activation='softmax', name='dense4'))
    #
    sgd = SGD(lr=1e-6, decay=4e-5, momentum=0.9, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_cnn_rev(model, input_img_name, info_layer_name, output_img):
    layers_output = dict()
    all_layers = list()

    print("Loading Image")
    img = image.load_img(input_img_name, target_size=(224,224))
    input_img_data = np.array([img_to_array(img)]).astype('float32')/255

    print("Obtaining all layers outputs")
    # obtain all inputs from all layers
    for i in model.layers:
        #if( i.__class__.__name__ == "Conv2D" or i.__class__.__name__ == "MaxPooling2D" ):
        func = K.function([model.input] , [model.get_layer(i.name).output])
        layers_output[i.name] = func([input_img_data])[0]

    # name of all layers
    for i in model.layers:
        all_layers.append(i.name)
        if(i.name == info_layer_name):
            break

    print("Copying conv layers filters, calculating the switches from previous maxpooling and creating deconv layers")
    # stores a list of all deconv layers stored each at one model and the maxpool correspondant
    models = list()
    model_functs = list()
    for i in range(len(all_layers)):
        # search for all conv layers
        if(all_layers[i].find("conv") != -1):
            
            # copy their weights and set to 1 all the weights
            wi = WeightCopy(model=model, layer=all_layers[i])
            bi = BiasCopy(output_size=model.get_layer(all_layers[i]).input_shape[3], layer=all_layers[i])
            
            # create a deconv layer with the same name of the conv layer
            model_deconv = [Sequential()]
            model_deconv[0].add(Conv2DTranspose(model.get_layer(all_layers[i]).input_shape[3], 
                        model.get_layer(all_layers[i]).get_config()["kernel_size"],
                        kernel_initializer=wi, bias_initializer=bi, activation="relu",
                        input_shape=model.get_layer(all_layers[i]).output_shape[1:], name="de"+all_layers[i]))
            model.get_layer(all_layers[i]).output_shape[1:]
            sgd = SGD(lr=1e-6, decay=4e-5, momentum=0.9, nesterov=False)
            model_deconv[0].compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
            
            # for this deconv layer find the correpondant maxpool layer
            k = i-1
            while ( k > 0 ):
                if(all_layers[k].find("pool") != -1):
                    # informations for stride and kernel size for unpooling
                    max_info = [model.get_layer(all_layers[k]).get_config()["pool_size"], model.get_layer(all_layers[k]).get_config()["pool_size"]]
                    # which unpooling layer to get switches
                    #idx_max = layers_seq.index(all_layers[k])
                    max_idxs = list()
                    
                    # perform the same maxpooling of the maxpool the will be inverted
                    # to find the switches, for this it needs the input of the maxpooling
                    for j in range(np.array(layers_output[all_layers[k-1]]).shape[3]):
                        output_interest = np.array(layers_output[all_layers[k-1]])[0,:,:,j]
                        # unpool all the filters of the previous layers
                        max_idx, _ = FindSwitches(ksize=model.get_layer(all_layers[i-1]).get_config()["pool_size"],
                                            stride=model.get_layer(all_layers[i-1]).get_config()["pool_size"],
                                            input_data=output_interest)
                        
                        #imsave("unpooling_{}_org.png".format(j),output_interest)
                        #imsave("unpooling_{}_unpooled.png".format(j),img_result)
                        #imsave("unpooling_{}.reconst.png".format(j),Reconstruct(max_idx, img_result))
                        
                        max_idxs.append(max_idx)
                    model_deconv.append(max_idxs)
                    break
                k -= 1
            models.append(model_deconv)
        if(all_layers[i] == info_layer_name):
            break
    
    '''
    models[
    
        [ sequential, [
        
                [filters, width, height]
        
        ]
    
    ]
    '''
    '''
    print("======================")
    for j in models:
        print(len(j))
        for i in j[0].layers:
            print("{} -> {} -> {}".format(i.name, i.__class__.__name__, i.output.shape))
            layers_seq.append(i.name)
            print()
            print(i.get_config())
            print(i.input_shape)
            print(i.output_shape)
            if(i.__class__.__name__ == "Conv2DTranspose"):
                print(i.get_weights()[1])
            print("--------------------")    
    '''
    

    print("Deconvolutions and Unpooling")    
    # the output of deconv is produced in inverse orther than the convolutions
    # the list of deconvs was constructed in the conv order, so it has to be
    # executed in reverse order
    i = len(models)-1
    while(i>=0):
        func = K.function([models[i][0].input] , [models[i][0].output])
        if(i == 0):
            # last layer of deconvolution uses does not use maxpooling on output
            if(i == len(models)-1):
                a = func([layers_output[models[i][0].layers[0].name.replace("deconv", "conv")]])[0]                
            else:
                a = func([a])[0]
        else:
            b = list()
            if(i == len(models)-1):
                # first layer of the deconvolution uses the last layer of the CNN as input
                a = func([layers_output[models[i][0].layers[0].name.replace("deconv", "conv")]])[0]
            else:
                # middle layers use the deconvolutions as input
                a = func([a])[0]                

            # middle and first deconvolution layers uses unpooling based on argmax of correpondant maxpooling
            for j in range(a.shape[3]):
                b.append([Unpool(models[i][1][j], a[0,:,:,j])])
            a = np.moveaxis(np.array(b), [0],[-1])
        i -= 1

    imsave(output_img, a[0])
        
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

    def __init__(self, output_size=None, layer=None):
        self.output_size = output_size
        self.layer = layer

    def __call__(self, shape, dtype=None):
        return np.ones(self.output_size)

    def get_config(self):
        return {
            'model': self.output_size,
            'layer': self.layer
        }


def set_callbacks(run_name):
    callbacks = list()
    checkpoint = ModelCheckpoint(filepath="models/unpooling",
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
    
    #callbacks.append(checkpoint)
    board = TensorBoard(log_dir='all_logs/cnn_fabio_{}__lr000001_lessdense_7x7filter_nesterov_decay00004_150epochs'.format(run_name), histogram_freq=0,
                            batch_size=32, write_graph=True, write_grads=False,
                            write_images=False, embeddings_freq=0,
                            embeddings_layer_names=None, embeddings_metadata=None)
    #callbacks.append(board)
    #
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1e-2, patience=5, min_lr=1e-9)
    callbacks.append(reduce_lr)
    #
    return callbacks

input_img_name = sys.argv[1]

try:
    model = load_model("models/unpooling")
except:
    print("Erro")
#except:
#    model = build_cnn(3)
#model = build_cnn(3)

'''
for i in model.layers:
    print("{} -> {} -> {}".format(i.name, i.__class__.__name__, i.output.shape))
    layers_seq.append(i.name)
    print()
    print(i.get_config())
    print(i.input_shape)
    print(i.output_shape)
    if(i.__class__.__name__ == "Conv2D"):
        print(i.get_weights()[1])
    print("--------------------")
    
print("----------------------------")    
'''

main_batch_size = 64
'''
train_imgs = LoadBreakhisList("folds_nonorm_nodataaug/dsfold1-100-train.txt")
val_imgs = LoadBreakhisList("folds_nonorm_nodataaug/dsfold1-100-validation.txt")
#
nr_batches_val = len(val_imgs)/main_batch_size
nr_batches = len(train_imgs)/main_batch_size


model.fit_generator(GeneratorImgs(train_imgs, batch_size=main_batch_size), \
        validation_steps=nr_batches_val, \
        validation_data=GeneratorImgs(val_imgs, batch_size=main_batch_size), \
        steps_per_epoch=nr_batches, epochs=30, verbose=False, max_queue_size=1, \
        workers=1, use_multiprocessing=False, \
        callbacks=set_callbacks("cnn_growing_{}".format(sys.argv[1])))
#
del train_imgs
del val_imgs
'''
#
'''
#
test_imgs = LoadBreakhisList("folds_nonorm_nodataaug/dsfold1-100-test.txt")  
#
scores = model.evaluate_generator(GeneratorImgs(test_imgs, batch_size=main_batch_size), 
        steps=len(test_imgs)/main_batch_size)
#
print('Test loss: {:.4f}'.format(scores[0]))
print('Test accuracy: {:.4f}'.format(scores[1]))
#
preds_proba = list()
preds = list()
labels = list()
#
for x, y, z in ReadImgs(test_imgs):
    predictions = model.predict(np.array([x])).squeeze()
    labels.append(y.argmax())
    preds.append(predictions.argmax())
    preds_proba.append(predictions[y.argmax()])
#
fpr, tpr, _ = roc_curve(labels, preds_proba, pos_label=0)
roc_auc = auc(fpr, tpr)
#
print("Test AUC 0: {:.4f}".format(roc_auc))
#
fpr, tpr, _ = roc_curve(labels, preds_proba, pos_label=1)
roc_auc = auc(fpr, tpr)
#
print("Test AUC 1: {:.4f}".format(roc_auc))
print("Confusion matrix:\n",confusion_matrix(labels, preds))
'''

build_cnn_rev(model, sys.argv[1], "conv2", "final_deconv.png")

exit(0)
