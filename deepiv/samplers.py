from __future__ import absolute_import, division, print_function, unicode_literals

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy
from keras import backend as K
from keras.engine.topology import InputLayer

_FLOATX = theano.config.floatX

def random_normal(shape, mean=0.0, std=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = numpy.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    return rng.normal(size=shape, avg=mean, std=std, dtype=dtype)


def random_multinomial(pvals, n=1, dtype=_FLOATX, seed=None):
    '''
    Theano function for sampling from a multinomal with probability given by `pvals`
    '''
    if seed is None:
        seed = numpy.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    return rng.multinomial(n=n, pvals=pvals, ndim=None, dtype=dtype)

def random_gmm(pi,mu,sig,n=None):
    '''
    Sample from a gaussian mixture model. Returns one sample for each row in
    the pi, mu and sig matrices... this is potentially wasteful (because you have to repeat
    the matrices n times if you want to get n samples), but makes it easy to implment 
    code where the parameters vary as they are conditioned on different datapoints.
    '''
    if n is not None:
        mu = tensor.repeat(mu, n, axis=0)
        sig = tensor.repeat(sig, n, axis=0)
        pi = tensor.repeat(pi, n, axis=0)
    normals = random_normal(mu.shape, mu, sig)
    k = random_multinomial(pi)
    return tensor.sum(normals * k, axis=1, keepdims=True, dtype=_FLOATX)

def gmm_sampler(model, n_mixtures):
    '''
    Given a Keras model that outputs mixture of gaussian parameters, this returns a function that
    draws samples from the resulting mixure model.
    '''
    x = model.layers[-1].output
    n = theano.tensor.iscalar()
    pi = x[:, 0:n_mixtures]
    mu = x[:, n_mixtures:2*n_mixtures]
    log_sig = x[:, 2*n_mixtures: 3*n_mixtures]
    inputs = [l.input for l in model.layers if type(l) == InputLayer]
    sampler = K.function(inputs + [n, K.learning_phase()], [random_gmm(pi, mu, K.exp(log_sig), n)])
    # helper function to make the Keras function nicer to work with...
    def sample(inputs,n=1, use_dropout=False):
        return sampler(inputs + [n, int(use_dropout)])[0]
    return sample

def multinomial_sampler(model, seed=None):
    '''
    Given a Keras model that outputs multinomial parameters, this returns a function that
    draws samples from the resulting distribution.
    '''
    pi = model.layers[-1].output
    n = theano.tensor.iscalar()
    inputs = [l.input for l in model.layers if type(l) == InputLayer] + [n, K.learning_phase()]
    sampler = K.function(inputs, [random_multinomial(pi.repeat(n, axis=0), seed=seed)])
    # helper function to make the Keras function nicer to work with...
    def sample_n(inputs, n=1, use_dropout=False):
        return sampler(inputs + [n, int(use_dropout)])[0]
    return sample_n


