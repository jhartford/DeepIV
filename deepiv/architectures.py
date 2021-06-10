from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Convolution2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm as maxnorm
#from tensorflow.keras.utils import np_utils
from tensorflow.python.keras import utils as np_utils
import numpy as np

def feed_forward_net(input, output, hidden_layers=[64, 64], activations='relu',
                     dropout_rate=0., l2=0., constrain_norm=False):
    '''
    Helper function for building a Keras feed forward network.

    input:  Keras Input object appropriate for the data. e.g. input=Input(shape=(20,))
    output: Function representing final layer for the network that maps from the last
            hidden layer to output.
            e.g. if output = Dense(10, activation='softmax') if we're doing 10 class
            classification or output = Dense(1, activation='linear') if we're doing
            regression.
    '''
    state = input
    if isinstance(activations, str):
        activations = [activations] * len(hidden_layers)
    
    for h, a in zip(hidden_layers, activations):
        if l2 > 0.:
            w_reg = tf.keras.regularizers.l2(l2)
        else:
            w_reg = None
        const = maxnorm(2) if constrain_norm else  None
        state = Dense(h, activation=a, kernel_regularizer=w_reg, kernel_constraint=const)(state)
        if dropout_rate > 0.:
            state = Dropout(dropout_rate)(state)
    return output(state)

def convnet(input, output, dropout_rate=0., input_shape=(1, 28, 28), batch_size=100,
            l2_rate=0.001, nb_epoch=12, img_rows=28, img_cols=28, nb_filters=64,
            pool_size=(2, 2), kernel_size=(3, 3), activations='relu', constrain_norm=False):
    '''
    Helper function for building a Keras convolutional network.

    input:  Keras Input object appropriate for the data. e.g. input=Input(shape=(20,))
    output: Function representing final layer for the network that maps from the last
            hidden layer to output.
            e.g. if output = Dense(10, activation='softmax') if we're doing 10 class
            classification or output = Dense(1, activation='linear') if we're doing
            regression.
    '''
    const = maxnorm(2) if constrain_norm else None

    state = Convolution2D(nb_filters, kernel_size, padding='valid',
                          input_shape=input_shape, activation=activations,
                          kernel_regularizer=l2(l2_rate), kernel_constraint=const)(input)

    state = Convolution2D(nb_filters, kernel_size,
                          activation=activations, kernel_regularizer=l2(l2_rate),
                          kernel_constraint=const)(state)

    state = MaxPooling2D(pool_size=pool_size)(state)

    state = Flatten()(state)

    if dropout_rate > 0.:
        state = Dropout(dropout_rate)(state)
    state = Dense(128, activation=activations, kernel_regularizer=l2(l2_rate), kernel_constraint=const)(state)

    if dropout_rate > 0.:
        state = Dropout(dropout_rate)(state)
    return output(state)

def feature_to_image(features, height=28, width=28, channels=1, backend=K):
    '''
    Reshape a flattened image to the input format for convolutions.

    Can be used either as a Keras operation using the default backend or
    with numpy by using the argument backend=np

    Conforms to the image data format setting defined in ~/.keras/keras.json
    '''
    if K.image_data_format() == "channels_first":
        return backend.reshape(features, (-1, channels, height, width))
    else:
        return backend.reshape(features, (-1, height, width, channels))

