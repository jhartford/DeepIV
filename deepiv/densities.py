from __future__ import absolute_import, division, print_function, unicode_literals

import numpy
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Reshape

def split(start, stop):
    return Lambda(lambda x: x[:, start:stop], output_shape=(None, stop-start))

def split_mixture_of_gaussians(x, n_components):
    pi = split(0, n_components)(x)
    mu = split(n_components, 2*n_components)(x)
    log_sig = split(2*n_components, 3*n_components)(x)
    return pi, mu, log_sig

def log_norm_pdf(x, mu, log_sig):
    z = (x - mu) / (K.exp(K.clip(log_sig, -40, 40))) #TODO: get rid of this clipping
    return -(0.5)*K.log(2*numpy.pi) - log_sig - 0.5*((z)**2)

def mix_gaussian_loss(x, mu, log_sig, w):
    '''
    Combine the mixture of gaussian distribution and the loss into a single function
    so that we can do the log sum exp trick for numerical stability...
    '''
    if K.backend() == "tensorflow":
        x.set_shape([None, 1])
    gauss = log_norm_pdf(K.repeat_elements(x=x, rep=mu.shape[1], axis=1), mu, log_sig)
    # TODO: get rid of clipping.
    gauss = K.clip(gauss, -40, 40)
    max_gauss = K.maximum((0.), K.max(gauss))
    # log sum exp trick...
    gauss = gauss - max_gauss
    out = K.sum(w * K.exp(gauss), axis=1)
    loss = K.mean(-K.log(out) + max_gauss)
    return loss

def mixture_of_gaussian_output(x, n_components):
    mu = keras.layers.Dense(n_components, activation='linear')(x)
    log_sig = keras.layers.Dense(n_components, activation='linear')(x)
    pi = keras.layers.Dense(n_components, activation='softmax')(x)
    return Concatenate(axis=1)([pi, mu, log_sig])

def mixture_of_gaussian_loss(y_true, y_pred, n_components):
    pi, mu, log_sig = split_mixture_of_gaussians(y_pred, n_components)
    return mix_gaussian_loss(y_true, mu, log_sig, pi)

def mixture_gaussian(n_components):
    '''
    Build a mixture of gaussian output and loss function that may be used in a keras graph.
    '''

    def output(x):
        return mixture_of_gaussian_output(x, n_components)

    def keras_loss(y, x):
        return mixture_of_gaussian_loss(y, x, n_components)
    return output, keras_loss
