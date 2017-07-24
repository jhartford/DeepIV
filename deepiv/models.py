# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings

import deepiv.samplers as samplers
import deepiv.densities as densities

from deepiv.custom_gradients import replace_gradients_mse

from keras.models import Model
from keras import backend as K
from keras.layers import Lambda, InputLayer

import keras.utils

import numpy
from sklearn import linear_model
from sklearn.decomposition import PCA
from scipy.stats import norm

class Treatment(Model):
    '''
    Adds sampling functionality to a Keras model and extends the losses to support
    mixture of gaussian losses.

    # Argument
    '''

    def _get_sampler_by_string(self, loss):
        output = self.outputs[0]
        inputs = self.inputs

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
            self._sampler = self._get_sampler_by_string(loss)
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
        if hasattr(self, "_sampler"):
            if not isinstance(inputs, list):
                inputs = [inputs]
            inputs = [i.repeat(n_samples, axis=0) for i in inputs]
            return self._sampler(inputs, use_dropout)
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

    def compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None,
                unbiased_gradient=False,n_samples=1, batch_size=None):
        super(Response, self).compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights,
                                      sample_weight_mode=sample_weight_mode)
        self.unbiased_gradient = unbiased_gradient
        if unbiased_gradient:
            if loss in ["MSE", "mse", "mean_squared_error"]:
                if batch_size is None:
                    raise ValueError("Must supply a batch_size argument if using unbiased gradients. Currently batch_size is None.")
                replace_gradients_mse(self, optimizer, batch_size=batch_size, n_samples=n_samples)
            else:
                warnings.warn("Unbiased gradient only implemented for mean square error loss. It is unnecessary for\
                              logistic losses and currently not implemented for absolute error losses.")
            

    def fit(self, x=None, y=None, batch_size=512, epochs=1, verbose=1, callbacks=None,
            validation_data=None, class_weight=None, initial_epoch=0, samples_per_batch=None,
            seed=None, observed_treatments=None):
        '''
        Trains the model by sampling from the fitted treament distribution.

        # Arguments
            x: list of numpy arrays. The first element should *always* be the instrument variables.
            y: (numpy array). Target response variables.
            The remainder of the arguments correspond to the Keras definitions.
        '''
        batch_size = numpy.minimum(y.shape[0], batch_size)
        if seed is None:
            seed = numpy.random.randint(0, 1e6)
        if samples_per_batch is None:
            if self.unbiased_gradient:
                samples_per_batch = 2
            else:
                samples_per_batch = 1

        if observed_treatments is None:
            generator = SampledSequence(x[1:], x[0], y, batch_size, self.treatment.sample, samples_per_batch)
        else:
            generator = OnesidedUnbaised(x[1:], x[0], y, observed_treatments, batch_size,
                                         self.treatment.sample, samples_per_batch)
        
        steps_per_epoch = y.shape[0]  // batch_size
        super(Response, self).fit_generator(generator=generator,
                                            steps_per_epoch=steps_per_epoch,
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

    def expected_representation(self, x, z, n_samples=100, batch_size=None, seed=None):
        inputs = [z, x]
        if not hasattr(self, "_E_representation"):
            if batch_size is None:
                batch_size = inputs[0].shape[0]
                steps = 1
            else:
                steps = inputs[0].shape[0] // batch_size

            intermediate_layer_model = Model(inputs=self.inputs,
                                             outputs=self.layers[-2].output)
            
            def pred(inputs, n_samples=100, seed=None):
                features = inputs[1]

                samples = self.treatment.sample(inputs, n_samples)
                batch_features = [features.repeat(n_samples, axis=0)] + [samples]
                representation = intermediate_layer_model.predict(batch_features)
                return representation.reshape((inputs[0].shape[0], n_samples, -1)).mean(axis=1)
            self._E_representation = pred
            return self._E_representation(inputs, n_samples, seed)
        else:
            return self._E_representation(inputs, n_samples, seed)

    def conditional_representation(self, x, p):
        inputs = [x, p]
        if not hasattr(self, "_c_representation"):          
            intermediate_layer_model = Model(inputs=self.inputs,
                                             outputs=self.layers[-2].output)

            self._c_representation = intermediate_layer_model.predict
            return self._c_representation(inputs)
        else:
            return self._c_representation(inputs)

    def dropout_predict(self, x, z, n_samples=100):
        if isinstance(x, list):
            inputs = [z] + x
        else:
            inputs = [z, x]
        if not hasattr(self, "_dropout_predict"):
            
            predict_with_dropout = K.function(self.inputs + [K.learning_phase()],
                                              [self.layers[-1].output])

            def pred(inputs, n_samples = 100):
                # draw samples from the treatment network with dropout turned on
                samples = self.treatment.sample(inputs, n_samples, use_dropout=True)
                # prepare inputs for the response network
                rep_inputs = [i.repeat(n_samples, axis=0) for i in inputs[1:]] + [samples]
                # return outputs from the response network with dropout turned on (learning_phase=0)
                return predict_with_dropout(rep_inputs + [1])[0]
            self._dropout_predict = pred
            return self._dropout_predict(inputs, n_samples)
        else:
            return self._dropout_predict(inputs, n_samples)

    def credible_interval(self, x, z, n_samples=100, p=0.95):
        '''
        Return a credible interval of size p using dropout variational inference.
        '''
        if isinstance(x, list):
            n = x[0].shape[0]
        else:
            n = x.shape[0]
        alpha = (1-p) / 2.
        samples = self.dropout_predict(x, z, n_samples).reshape((n, n_samples, -1))
        upper = numpy.percentile(samples.copy(), 100*(p+alpha), axis=1)
        lower = numpy.percentile(samples.copy(), 100*(alpha), axis=1)
        return lower, upper

    def _add_constant(self, X):
        return numpy.concatenate((numpy.ones((X.shape[0], 1)), X), axis=1)
    
    def predict_confidence(self, x, p):
        if hasattr(self, "_predict_confidence"):
            return self._predict_confidence(x, p)
        else:
            raise Exception("Call fit_confidence_interval before running predict_confidence")

    
    def fit_confidence_interval(self, x_lo, z_lo, p_lo, y_lo, n_samples=100, alpha=0.):
        eta_bar = self.expected_representation(x=x_lo, z=z_lo, n_samples=n_samples)
        pca = PCA(1-1e-16, svd_solver="full", whiten=True)
        pca.fit(eta_bar)

        eta_bar = pca.transform(eta_bar)
        eta_lo_prime = pca.transform(self.conditional_representation(x_lo, p_lo))
        eta_lo = self._add_constant(eta_lo_prime)

        ols1 = linear_model.Ridge(alpha=alpha, fit_intercept=True)
        ols1.fit(eta_bar, eta_lo_prime)
        hhat = ols1.predict(eta_bar)
        ols2 = linear_model.Ridge(alpha=alpha, fit_intercept=False)
        ols2.fit(self._add_constant(hhat), y_lo)

        yhat = ols2.predict(eta_lo)
        hhi = numpy.linalg.inv(numpy.dot(eta_lo.T, eta_lo))

        heh = numpy.dot(eta_lo.T, numpy.square(y_lo - yhat) * eta_lo)
        V = numpy.dot(numpy.dot(hhi, heh), hhi)

        def pred(xx, pp):
            H = self._add_constant(pca.transform(self.conditional_representation(xx,pp)))
            sdhb = numpy.sqrt(numpy.diag(numpy.dot(numpy.dot(H, V), H.T)))
            hb = ols2.predict(H).flatten()
            return hb, sdhb
        
        self._predict_confidence = pred




class SampledSequence(keras.utils.Sequence):
    def __init__(self, features, instruments, outputs, batch_size, sampler, n_samples=1, seed=None):
        self.rng = numpy.random.RandomState(seed)
        if not isinstance(features, list):
            features = [features.copy()]
        else:
            features = [f.copy() for f in features]
        self.features = features
        self.instruments = instruments.copy()
        self.outputs = outputs.copy()
        if batch_size < self.instruments.shape[0]:
            self.batch_size = batch_size
        else:
            self.batch_size = self.instruments.shape[0]
        self.sampler = sampler
        self.n_samples = n_samples
        self.current_index = 0
        self.shuffle()

    def __len__(self):
        if isinstance(self.outputs, list):
            return self.outputs[0].shape[0] // self.batch_size
        else:
            return self.outputs.shape[0] // self.batch_size

    def shuffle(self):
        idx = self.rng.permutation(numpy.arange(self.instruments.shape[0]))
        self.instruments = self.instruments[idx,:]
        self.outputs = self.outputs[idx,:]
        self.features = [f[idx,:] for f in self.features]
    
    def __getitem__(self,idx):
        instruments = [self.instruments[idx*self.batch_size:(idx+1)*self.batch_size, :]]
        features = [inp[idx*self.batch_size:(idx+1)*self.batch_size, :] for inp in self.features]
        sampler_input = instruments + features
        samples = self.sampler(sampler_input, self.n_samples)
        batch_features = [f[idx*self.batch_size:(idx+1)*self.batch_size].repeat(self.n_samples, axis=0) for f in self.features] + [samples]
        batch_y = self.outputs[idx*self.batch_size:(idx+1)*self.batch_size].repeat(self.n_samples, axis=0)
        if idx == (len(self) - 1):
            self.shuffle()
        return batch_features, batch_y

class OnesidedUnbaised(SampledSequence):
    def __init__(self, features, instruments, outputs, treatments, batch_size, sampler, n_samples=1, seed=None):
        self.rng = numpy.random.RandomState(seed)
        if not isinstance(features, list):
            features = [features.copy()]
        else:
            features = [f.copy() for f in features]
        self.features = features
        self.instruments = instruments.copy()
        self.outputs = outputs.copy()
        self.treatments = treatments.copy()
        self.batch_size = batch_size
        self.sampler = sampler
        self.n_samples = n_samples
        self.current_index = 0
        self.shuffle()

    def shuffle(self):
        idx = self.rng.permutation(numpy.arange(self.instruments.shape[0]))
        self.instruments = self.instruments[idx,:]
        self.outputs = self.outputs[idx,:]
        self.features = [f[idx,:] for f in self.features]
        self.treatments = self.treatments[idx,:]

    def __getitem__(self, idx):
        instruments = [self.instruments[idx*self.batch_size:(idx+1)*self.batch_size, :]]
        features = [inp[idx*self.batch_size:(idx+1)*self.batch_size, :] for inp in self.features]
        observed_treatments = self.treatments[idx*self.batch_size:(idx+1)*self.batch_size, :]
        sampler_input = instruments + features
        samples = self.sampler(sampler_input, self.n_samples // 2)
        samples = numpy.concatenate([observed_treatments, samples], axis=0)
        batch_features = [f[idx*self.batch_size:(idx+1)*self.batch_size].repeat(self.n_samples, axis=0) for f in self.features] + [samples]
        batch_y = self.outputs[idx*self.batch_size:(idx+1)*self.batch_size].repeat(self.n_samples, axis=0)
        if idx == (len(self) - 1):
            self.shuffle()
        return batch_features, batch_y
