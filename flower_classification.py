from _warnings import filters

import keras
from keras import layers, optimizers
from keras import models
from keras.applications import ResNet50
from keras.initializers import RandomNormal, Constant
from keras.layers import Dense, Conv2D, Flatten, Masking, RepeatVector, MaxPooling2D, Activation, Dropout, \
    BatchNormalization
from keras_applications.inception_v3 import WEIGHTS_PATH_NO_TOP
import matplotlib.pyplot as plt
import PIL

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
flower_path = r'C:\Users\USER\Desktop\flower_project\flowers'
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

model.add(layers.Conv2D(32,(3,3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32,(3,3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))





# last layer should be with softmax activation function - do not change!!!
model.add(layers.Dense(classes, activation='softmax'))


# fill optimizer argument using one of keras.optimizers.
# read Keras documentation : https://keras.io/models/model/
optimizer = 'Adam'

# fill loss argument using keras.losses.
# reads Keras documentation https://keras.io/losses/
loss = 'categorical_crossentropy'
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# you can change number of epochs by changing the value of the 'epochs' paramter
#model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=30, validation_data=valid_gen,
#                                 validation_steps=v_steps)
#model.save('flowers_model2.h5')
#plt_modle(model_hist)