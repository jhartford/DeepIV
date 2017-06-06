# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings

import deepiv.samplers as samplers
import deepiv.densities as densities

from keras.models import Model
from keras import backend as K
from keras.layers import Lambda, InputLayer

import numpy


class Treatment(Model):
    '''
    Adds sampling functionality to a Keras model and extends the losses to support
    mixture of gaussian losses.

    # Argument
    '''

    def _get_sampler_by_string(self, loss):
        output = self.outputs[0]
        inputs = self.inputs
        print("Sampler inputs:",inputs)

        if loss in ["MSE", "mse", "mean_squared_error"]:
            output += samplers.random_normal(K.shape(output), mean=0.0, std=1.0)
            draw_sample = K.function(inputs + [K.learning_phase()], [output])

            def sample_gaussian(inputs, use_dropout=False):
                '''
                Helper to draw samples from a gaussian distribution
                '''
                return draw_sample(inputs + [int(use_dropout)])[0]

            return sample_gaussian

        elif loss == "binary_crossentropy":
            output = K.random_binomial(K.shape(output), p=output)
            draw_sample = K.function(inputs + [K.learning_phase()], [output])

            def sample_binomial(inputs, use_dropout=False):
                '''
                Helper to draw samples from a binomial distribution
                '''
                return draw_sample(inputs + [int(use_dropout)])[0]

            return sample_binomial

        elif loss in ["mean_absolute_error", "mae", "MAE"]:
            output += samplers.random_laplace(K.shape(output), mu=0.0, b=1.0)
            draw_sample = K.function(inputs + [K.learning_phase()], [output])
            def sample_laplace(inputs, use_dropout=False):
                '''
                Helper to draw samples from a Laplacian distribution
                '''
                return draw_sample(inputs + [int(use_dropout)])[0]

            return sample_laplace

        elif loss == "mixture_of_gaussians":
            pi, mu, log_sig = densities.split_mixture_of_gaussians(output, self.n_components)
            samples = samplers.random_gmm(pi, mu, K.exp(log_sig))
            draw_sample = K.function(inputs + [K.learning_phase()], [samples])
            return lambda inputs, use_dropout: draw_sample(inputs + [int(use_dropout)])[0]

        else:
            raise NotImplementedError("Unrecognised loss: %s.\
                                       Cannot build a generic sampler" % loss)

    def _prepare_sampler(self, loss):
        '''
        Build sampler
        '''
        if isinstance(loss, str):
            self.sampler = self._get_sampler_by_string(loss)
        else:
            warnings.warn("You're using a custom loss function. Make sure you implement\
                           the model's sample() fuction yourself.")

    def compile(self, optimizer, loss, metrics=None, loss_weights=None,
                sample_weight_mode=None, n_components=None, **kwargs):
        '''
        Overrides the existing keras compile function to add a sampler building
        step to the model compilation phase. Once compiled, one can draw samples
        from the network using the sample() function and adds support for mixture
        of gaussian loss.

        '''
        if loss == "mixture_of_gaussians":
            if n_components is None:
                raise Exception("When using mixture of gaussian loss you must\
                                 supply n_components argument")
            self.n_components = n_components
            self._prepare_sampler(loss)
            loss = lambda y_true, y_pred: densities.mixture_of_gaussian_loss(y_true,
                                                                             y_pred,
                                                                             n_components)

            def predict_mean(x, batch_size=32, verbose=0):
                '''
                Helper to just predict the expected value of the mixture
                of gaussian rather than the parameters for the distribution.
                '''
                y_hat = super(Treatment, self).predict(x, batch_size, verbose)
                n_c = n_components
                return (y_hat[:, 0:n_c] * y_hat[:, n_c:2*n_c]).sum(axis=1, keepdims=True)

            self.predict_mean = predict_mean
        else:
            self._prepare_sampler(loss)

        super(Treatment, self).compile(optimizer, loss, metrics=metrics, loss_weights=loss_weights,
                                       sample_weight_mode=sample_weight_mode, **kwargs)

    def sample(self, inputs, n_samples=1, use_dropout=False):
        '''
        Draw samples from the keras model.
        '''
        if hasattr(self, "sampler"):
            if not isinstance(inputs, list):
                inputs = [inputs]
            inp = [i.repeat(n_samples, axis=0) for i in inputs]
            return self.sampler(inp, use_dropout)
        else:
            raise Exception("Compile model with loss before sampling")

class Response(Model):
    '''
    Extends the Keras Model class to support sampling from the Treatment
    model during training.

    Overwrites the existing fit_generator function.

    # Arguments
    In addition to the standard model arguments, a Response object takes
    a Treatment object as input so that it can sample from the fitted treatment
    distriubtion during training.
    '''
    def __init__(self, treatment, **kwargs):
        if isinstance(treatment, Treatment):
            self.treatment = treatment
        else:
            raise TypeError("Expected a treatment model of type Treatment. \
                             Got a model of type %s. Remember to train your\
                             treatment model first." % type(treatment))
        super(Response, self).__init__(**kwargs)

    def _prepare_generator(self, n_samples=2, seed=123):
        def data_generator(inputs, outputs, batch_size):
            '''
            Data generator that samples from the treatment network during training
            '''
            n_train = outputs.shape[0]
            rng = numpy.random.RandomState(seed)
            batch_size = min(batch_size, n_train)
            idx = numpy.arange(n_train)
            while 1:
                idx = rng.permutation(idx)
                inputs = [inp[idx, :] for inp in inputs] #shuffle examples
                outputs = outputs[idx, :]
                n_batches = n_train//batch_size
                for i in range(n_batches):
                    instruments = [inputs[0][i*batch_size:(i+1)*batch_size, :]]
                    features = [inp[i*batch_size:(i+1)*batch_size, :] for inp in inputs[1:]]
                    sampler_input = instruments + features
                    sampled_t = self.treatment.sample(sampler_input, n_samples)
                    response_inp = [inp.repeat(n_samples, axis=0) for inp in features] + [sampled_t]
                    y_train = outputs[i*batch_size:(i+1)*batch_size, :].repeat(n_samples, axis=0)
                    yield response_inp, y_train
                
        return data_generator

    def fit(self, x=None, y=None, batch_size=512, epochs=1, verbose=1, callbacks=None,
            validation_data=None, class_weight=None, initial_epoch=0, samples_per_batch=2,
            seed=None):
        '''
        Trains the model by sampling from the fitted treament distribution.

        # Arguments
            x: list of numpy arrays. The first element should *always* be the instrument variables.
            y: (numpy array). Target response variables.
            The remainder of the arguments correspond to the Keras definitions.
        '''
        if seed is None:
            seed = numpy.random.randint(0, 1e6)
        generator = self._prepare_generator(samples_per_batch, seed)
        steps_per_epoch = y.shape[0]  // batch_size
        super(Response, self).fit_generator(generator=generator(x, y, batch_size), steps_per_epoch=steps_per_epoch,
                                            epochs=epochs, verbose=verbose,
                                            callbacks=callbacks, validation_data=validation_data,
                                            class_weight=class_weight, initial_epoch=initial_epoch)

    def fit_generator(self, **kwargs):
        '''
        We use override fit_generator to support sampling from the treatment model during training.

        If you need this functionality, you'll need to build a generator that samples from the
        treatment and performs whatever transformations you're performing. Please submit a pull
        request if you implement this.
        '''
        raise NotImplementedError("We use override fit_generator to support sampling from the\
                                   treatment model during training.")
        