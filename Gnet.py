from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.version import LooseVersion
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import os

if LooseVersion(keras.__version__) >= LooseVersion('2.0.0'):
    from keras.layers import Conv2D
else:
    from keras.layers import Convolution2D


def conv_2d(filters, kernel_shape, strides, padding):
    """
    Defines the right convolutional layer according to the
    version of Keras that is installed.
    :param filters: (required integer) the dimensionality of the output
                    space (i.e. the number output of filters in the
                    convolution)
    :param kernel_shape: (required tuple or list of 2 integers) specifies
                         the strides of the convolution along the width and
                         height.
    :param padding: (required string) can be either 'valid' (no padding around
                    input or feature map) or 'same' (pad to ensure that the
                    output feature map size is identical to the layer input)
    :return: the Keras layer
    """
    if LooseVersion(keras.__version__) >= LooseVersion('2.0.0'):
        return Conv2D(filters=filters, kernel_size=kernel_shape,
                      strides=strides, padding=padding)
    else:
        return Convolution2D(filters, kernel_shape[0], kernel_shape[1],
                             subsample=strides, border_mode=padding)


def G_10(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=64, nb_classes=10):
    """
    Defines a G10 model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """
    model = Sequential()

    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)

    model = Sequential()
    model.add(Conv2D(64, (3,3), padding = 'same',activation='relu',input_shape=input_shape))

    model.add(Conv2D(128, (3,3),padding = 'same',activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    if logits:
        logits_tensor = model(input_ph)

    if logits:
        return model, logits_tensor
    else:
        return model

def G_11(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=64, nb_classes=10):
    """
    Defines a G_11 model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """
    model = Sequential()

    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)

    model = Sequential()
    model.add(Conv2D(64, (3,3), padding = 'same',activation='relu',input_shape=input_shape))

    model.add(Conv2D(128, (3,3),padding = 'same',activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(128, (3,3),padding = 'same',activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    if logits:
        logits_tensor = model(input_ph)

    if logits:
        return model, logits_tensor
    else:
        return model

def G_12(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=64, nb_classes=10):
    """
    Defines a G_12 model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """
    model = Sequential()

    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)

    model = Sequential()
    model.add(Conv2D(64, (3,3), padding = 'same',activation='relu',input_shape=input_shape))

    model.add(Conv2D(128, (3,3),padding = 'same',activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(256, (3,3),padding = 'same',activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    if logits:
        logits_tensor = model(input_ph)

    if logits:
        return model, logits_tensor
    else:
        return model

def G_13(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=64, nb_classes=10):
    """
    Defines a G_13 model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """
    model = Sequential()

    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)

    model = Sequential()
    model.add(Conv2D(64, (3,3), padding = 'same',activation='relu',input_shape=input_shape))

    model.add(Conv2D(128, (3,3),padding = 'same',activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(256, (3,3),padding = 'same',activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(256, (3,3),padding = 'same',activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    if logits:
        logits_tensor = model(input_ph)

    if logits:
        return model, logits_tensor
    else:
        return model