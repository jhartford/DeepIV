from __future__ import absolute_import, division, print_function, unicode_literals

import numpy
import keras
from keras import backend as K

def mixture_gaussian(n_mixtures):
    '''
    Build a mixture of gaussian output and loss function that may be used in a keras graph.
    '''
    def log_norm_pdf(x, mu, log_sig):
        z = (x - mu) / (K.exp(K.clip(log_sig, -40, 40))) #TODO: get rid of this clipping
        return -(0.5)*K.log(2*numpy.pi) - log_sig - 0.5*((z)**2)
    
    def mix_gaussian_loss(x, mu, log_sig, w):
        '''
        Combine the mixture of gaussian distribution and the loss into a single function 
        so that we can do the log sum exp trick for numerical stability...
        '''
        gauss = log_norm_pdf(K.repeat_elements(x, mu.shape[1], axis=1), mu, log_sig)
        # TODO: get rid of clipping. 
        gauss = K.clip(gauss, -40, 40)
        m = K.maximum((0.), gauss.max())
        # log sum exp trick...
        gauss = gauss - m
        out = (w * K.exp(gauss)).sum(axis=1)
        p = -(K.log(out) + m).mean()
        return p
    
    def output(x):
        mu = keras.layers.Dense(n_mixtures, activation='linear')(x)
        log_sig = keras.layers.Dense(n_mixtures, activation='linear')(x)
        pi = keras.layers.Dense(n_mixtures, activation='softmax')(x)
        return keras.layers.merge([pi, mu, log_sig], mode='concat')
    
    def keras_loss(y, x):
        pi = x[:, 0:n_mixtures]
        mu = x[:, n_mixtures:2*n_mixtures]
        log_sig = x[:, 2*n_mixtures: 3*n_mixtures]
        return mix_gaussian_loss(y, mu, log_sig, pi)
    return output, keras_loss
