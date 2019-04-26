from _warnings import filters

import keras
from keras import layers, optimizers
from keras import models
from keras.applications import ResNet50
from keras.initializers import RandomNormal, Constant
from keras.layers import Dense, Conv2D, Flatten, Masking, RepeatVector, MaxPooling2D, Activation, Dropout, \
    BatchNormalization, np
from keras_applications.inception_v3 import WEIGHTS_PATH_NO_TOP
import matplotlib.pyplot as plt
import PIL
from tensorflow.contrib import learn
import tensorflow as tf



from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def plt_modle(model_hist):
    acc = model_hist.history['acc']
    val_acc = model_hist.history['val_acc']
    loss = model_hist.history['loss']
    val_loss = model_hist.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 6));
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, color='#0984e3', marker='o', linestyle='none', label='Training Accuracy')
    plt.plot(epochs, val_acc, color='#0984e3', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, color='#eb4d4b', marker='o', linestyle='none', label='Training Loss')
    plt.plot(epochs, val_loss, color='#eb4d4b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()

# Split images into Training and Validation Sets (20%)

train = ImageDataGenerator(rescale=1./255,horizontal_flip=True, shear_range=0.2, zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)

img_size = 128
batch_size = 20
t_steps = 3462/batch_size
v_steps = 861/batch_size
classes = 5
flower_path = r'C:\Users\talmalu\Downloads\flowers\flowers'
train_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode='categorical', subset='training')
valid_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical', subset='validation')

# Model

model = models.Sequential()

# use model.add() to add any layers you like
# read Keras documentation to find which layers you can use:
#           https://keras.io/layers/core/
#           https://keras.io/layers/convolutional/
#           https://keras.io/layers/pooling/
#

model.add(layers.Conv2D(32,(3,3), input_shape=(128, 128, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
keras.layers.ActivityRegularization(l1=0.0, l2=0.0)
keras.layers.Masking(mask_value=0.0)
lr=np.array([1e-6,1e-4,1e-2])
keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)


model.add(layers.Conv2D(32,(3,3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32,(3,3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
keras.layers.Dropout(0, noise_shape=None, seed=None)

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.3))


def new_biases(length):
    return tf.Variable(tf.constant(0.2, shape=[length]))


def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


x = tf.placeholder(tf.float32, shape=[None, 128*128], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, 1])
y_true = tf.placeholder(tf.float32, shape=[None, 5], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=1, filter_size=4,
                                            num_filters=32, use_pooling=True)
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=32,
                                            filter_size=4,
                                            num_filters=64, use_pooling=True)
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2, num_input_channels=64,
                                            filter_size=4,
                                            num_filters=128, use_pooling=True)
layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3, num_input_channels=128,
                                            filter_size=4,
                                            num_filters=256, use_pooling=True)
layer_conv5, weights_conv5 = new_conv_layer(input=layer_conv4, num_input_channels=256,
                                            filter_size=4,
                                            num_filters=128, use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv5)
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=1024, use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=1024, num_outputs=5, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# last layer should be with softmax activation function - do not change!!!
model.add(layers.Dense(classes, activation='softmax'))

model.load_weights('C:\\Users\\talmalu\\PycharmProjects\\FlowerClassification\\flowers_model3.h5')

# fill optimizer argument using one of keras.optimizers.
# read Keras documentation : https://keras.io/models/model/
optimizer = 'Adam'

# fill loss argument using keras.losses.
# reads Keras documentation https://keras.io/losses/
loss = 'categorical_crossentropy'
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

#you can change number of epochs by changing the value of the 'epochs' paramter
model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=20, validation_data=valid_gen,
                                validation_steps=v_steps)
model.save('flowers_model4.h5')
plt_modle(model_hist)