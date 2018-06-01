from __future__ import print_function
import keras
from keras import applications
from keras import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten
from scipy.misc import imsave
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import numpy as np
import time
#from keras.preprocessing.image import save_img
from keras import backend as K
import sys

# dimensions of the generated pictures for each filter.
img_width = 128
img_height = 128


if(len(sys.argv) != 2):
	exit(0)
	
input_img_name = sys.argv[1]

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
layer_name = 'res5c_branch2c'

# util function to convert a tensor into a valid image


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# build the VGG16 network with ImageNet weights
#model = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)
resnet = keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)
x = resnet.get_layer(name="avg_pool").output #output
x = Flatten()(x)
x = Dropout(0.2)(x)
#x = Dense(500, activation="relu")(x)
#x = Dropout(0.2)(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(50, activation="relu")(x)
x = Dense(2, activation='softmax', name="dense2")(x)
model = Model(inputs=resnet.inputs, outputs=x)
model.load_weights("lala.hdf5")
print('Model loaded.')

#for i in model.layers:
#	print('%s %s' % (i.name, i.__class__.__name__))

#model.summary()
#exit(0)

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


kept_filters = []

print(layer_dict[layer_name].output.shape)


nm_fil = 0
for filter_index in range(layer_dict[layer_name].output.shape[3]):
    if(nm_fil > 32 ):
        break
		
    nm_fil += 1
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    #if K.image_data_format() == 'channels_first':
    #    input_img_data = np.random.random((1, 3, img_width, img_height))
    #else:
    #    input_img_data = np.random.random((1, img_width, img_height, 3))
        
    img = image.load_img(input_img_name, target_size=(224,224))
    input_img_data = np.array([img_to_array(img)]).astype('float32')/255
    #input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        #print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    #if loss_value > 0:
    img = deprocess_image(input_img_data[0])
    kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

# we will stich the best 64 filters on a 8 x 8 grid.
n = 8

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)

predict = model.predict(input_img_data).squeeze().argmax()
print(predict)

t=0
img_dir = input_img_name[:-4]
for img,loss in kept_filters:
	imsave('%s/img_pred_%d_%d.png' % (img_dir,predict,t), img)
	t += 1

exit(0)

kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

# save the result to disk
#save_img('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
		
		
		
		
