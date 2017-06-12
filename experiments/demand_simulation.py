from __future__ import print_function

import warnings

from deepiv.models import Treatment, Response
import deepiv.architectures as architectures
import deepiv.densities as densities

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.merge import Concatenate

import numpy

import data_generator

n = 5000
dropout_rate = min(1000./(1000. + n), 0.5)
epochs = int(1500000./float(n)) # heuristic to select number of epochs
batch_size = 100

x, z, t, y, g_true = data_generator.demand(n=n, seed=1, ypcor=0.5, use_images=False, test=False)

print("Data shapes:\n\
Features:{x},\n\
Instruments:{z},\n\
Treament:{t},\n\
Response:{y}".format(**{'x':x.shape, 'z':z.shape,
                        't':t.shape, 'y':y.shape}))

# Build and fit treatment model
instruments = Input(shape=(z.shape[1],), name="instruments")
features = Input(shape=(x.shape[1],), name="features")
treatment_input = Concatenate(axis=1)([instruments, features])

hidden = [128,64,32]

est_treat = architectures.feed_forward_net(treatment_input, lambda x: densities.mixture_of_gaussian_output(x, 10),
                                           hidden_layers=hidden,
                                           dropout_rate=dropout_rate, l2=0.0001,
                                           activations='tanh')

treatment_model = Treatment(inputs=[instruments, features], outputs=est_treat)
treatment_model.compile('adam',
                        loss="mixture_of_gaussians",
                        n_components=10)

treatment_model.fit([z, x], t, epochs=epochs, batch_size=batch_size)

# Build and fit response model

treatment = Input(shape=(t.shape[1],), name="treatment")
response_input = Concatenate(axis=1)([features, treatment])

est_response = architectures.feed_forward_net(response_input, Dense(1),
                                              activations='relu',
                                              hidden_layers=hidden,
                                              l2=0.001,
                                              dropout_rate=dropout_rate)

response_model = Response(treatment=treatment_model,
                          inputs=[features, treatment],
                          outputs=est_response)
response_model.compile('adam', loss='mse')
response_model.fit([z, x], y, epochs=epochs, verbose=1,
                   batch_size=batch_size, samples_per_batch=2)

def datafunction(n, s, images=False, test=False):
    return data_generator.demand(n=n, seed=s, ypcor=0.5, use_images=images, test=test)

oos_perf = data_generator.monte_carlo_error(lambda x,z,t: response_model.predict([x,t]), datafunction, has_latent=False, debug=False)
print("Out of sample performance evaluated against the true function: %f" % oos_perf)
