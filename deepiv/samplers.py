from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy
from tensorflow.keras import backend as K
# from tensorflow.keras.layers import InputLayer

if K.backend() == "theano":
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    _FLOATX = theano.config.floatX
else:
    import tensorflow as tf


def random_laplace(shape, mu=0., b=1.):
    '''
    Draw random samples from a Laplace distriubtion.

    See: https://en.wikipedia.org/wiki/Laplace_distribution#Generating_random_variables_according_to_the_Laplace_distribution
    '''
    U = K.random_uniform(shape, -0.5, 0.5)  # alias to
    return mu - b * K.sign(U) * K.log(1 - 2 * K.abs(U))


def random_normal(shape, mean=0.0, std=1.0):
    # Returns: A tensor with normal distribution of values.
    return K.random_normal(shape, mean, std)


def random_multinomial(logits, seed=None):
    '''
    Theano function for sampling from a multinomal with probability given by `logits`
    '''
    if K.backend() == "theano":
        if seed is None:
            seed = numpy.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
        return rng.multinomial(n=1, pvals=logits, ndim=None, dtype=_FLOATX)
    elif K.backend() == "tensorflow":
        samples_multi = tf.random.categorical(logits=K.log(logits), num_samples=1)
        #samples_multi = tf.compat.v1.multinomial(logits=K.log(logits), num_samples=1)
        # smples_multi = tfp.distributions.Multinomial(total_count=1,logits=K.log(logits)).sample(1)
        sample_squeeze = tf.squeeze(samples_multi)
        return tf.one_hot(sample_squeeze, int(logits.shape[1]))


def random_gmm(pi, mu, sig):
    '''
    Sample from a gaussian mixture model. Returns one sample for each row in
    the pi, mu and sig matrices... this is potentially wasteful (because you have to repeat
    the matrices n times if you want to get n samples), but makes it easy to implment
    code where the parameters vary as they are conditioned on different datapoints.
    '''
    normals = random_normal(K.shape(mu), mu, sig)  # [None,n_components]
    k = random_multinomial(pi)  # shape None
    return K.sum(normals * k, axis=1, keepdims=True)
