from __future__ import absolute_import, division, print_function, unicode_literals

import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.utils import np_utils


def feed_forward_net(input, output, hidden_layers=[64, 64], activations='relu', dropout_rate=0., l2=0., constrain_norm=False):
    '''
    Helper function for building a Keras feed forward network.

    input: Keras Input object appropriate for the data. e.g. input=Input(shape=(20,))
    output: Function representing final layer for the network that maps from the last hidden layer to output.
    e.g. if output = Dense(10, activation='softmax') if we're doing 10 class classification
    or output = Dense(1, activation='linear') if we're doing regression.
    '''
    state = input
    for h in hidden_layers:
        if l2 > 0.:
            w_reg = keras.regularizers.l2(l2)
        else:
            w_reg = None
        const = maxnorm(2) if constrain_norm else  None
        state = Dense(h, activation=activations, W_regularizer=w_reg, W_constraint = const)(state)
        if dropout_rate > 0.:
            state = Dropout(dropout_rate)(state)
    return output(state)

def convnet(input, output, dropout_rate=0., input_shape=(1,28,28), batch_size=100, l2_rate=0.001, 
            nb_epoch = 12, img_rows=28, img_cols=28, nb_filters=64, 
            pool_size = (2,2), kernel_size = (3, 3), activations='relu', constrain_norm=False):
    const = maxnorm(2) if constrain_norm else  None
    
    state = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
              border_mode='valid',
              input_shape=input_shape, activation=activations, W_regularizer=l2(l2_rate),
              W_constraint=const)(input)
    
    state = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                          activation=activations, W_regularizer=l2(l2_rate), 
                          W_constraint=const)(state)
    
    state = MaxPooling2D(pool_size=pool_size)(state)
    
    state = Flatten()(state)
    
    if dropout_rate > 0.:
        state = Dropout(dropout_rate)(state)
    state = Dense(128, activation=activations, W_regularizer=l2(l2_rate), W_constraint=const)(state)

    if dropout_rate > 0.:
        state = Dropout(dropout_rate)(state)
    return output(state)
