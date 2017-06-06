from __future__ import print_function

import warnings

from deepiv.models import Treatment, Response
import deepiv.architectures as architectures
import deepiv.densities as densities

from keras.layers import Input, Dense, Reshape
from keras.models import Model
from keras.layers.merge import Concatenate
import keras.backend as K

import numpy

import data_generator

def conv_embedding(images, output, other_features = [], dropout_rate=0.1,
                   embedding_dropout=0.1, embedding_l2=0.05, constrain_norm=True):
    print("Building conv net")
    x_embedding = architectures.convnet(images, Dense(64, activation='linear'),
                        dropout_rate=embedding_dropout,
                        activations='relu',
                        l2_rate=embedding_l2, constrain_norm=constrain_norm)

    if len(other_features) > 0:
        embedd = Concatenate(axis=1)([x_embedding] + other_features)
    else:
        embedd = x_embedding
    out = architectures.feed_forward_net(embedd, output,
                        hidden_layers=[32],
                        dropout_rate=dropout_rate,
                        activations='relu', constrain_norm=constrain_norm)
    return out

n = 5000
dropout_rate = min(1000./(1000. + n), 0.5)
epochs = 100
embedding_dropout = 0.1
embedding_l2 = 0.1
epochs = int(1500000./float(n))
batch_size = 100

x, z, t, y, g_true = data_generator.demand(n=n, seed=1, ypcor=0.5, use_images=True, test=False)

print("Data shapes:\n\
Features:{x},\n\
Instruments:{z},\n\
Treament:{t},\n\
Response:{y}".format(**{'x':x.shape, 'z':z.shape,
                        't':t.shape, 'y':y.shape}))

# Build and fit treatment model
if K.image_data_format() == "channels_first":
    image_shape = (1, 28, 28)
else:
    image_shape = (28, 28, 1)

images = Input(shape=(28 * 28,), name='treat_images')
image_reshaped = Reshape(image_shape)(images) # reshape
time = Input(shape=(1,), name='treat_time')
instruments = Input(shape=(z.shape[1],), name='treat_instruments')

mix_gaussian_output = lambda x: densities.mixture_of_gaussian_output(x, 10)

treatment_output = conv_embedding(image_reshaped, mix_gaussian_output,
                                  [time, instruments], 
                                  dropout_rate=dropout_rate,
                                  embedding_dropout=embedding_dropout,
                                  embedding_l2=embedding_l2)


treatment_model = Treatment(inputs=[instruments, time, images], outputs=treatment_output)
treatment_model.compile('adam',
                        loss="mixture_of_gaussians",
                        n_components=10)

treatment_model.fit([z, x[:,0:1], x[:,1:]], t, epochs=epochs, batch_size=batch_size)

# Build and fit response model

treatment = Input(shape=(t.shape[1],), name="treatment")

out_res = conv_embedding(image_reshaped, Dense(1, activation='linear'), [time, treatment],
                dropout_rate=dropout_rate, embedding_dropout=embedding_dropout, embedding_l2=embedding_l2)

response_model = Response(treatment=treatment_model,
                          inputs=[time, images, treatment],
                          outputs=out_res)
response_model.compile('adam', loss='mse')
response_model.fit([z, x[:,0:1], x[:,1:]], y, epochs=epochs, verbose=1,
                   batch_size=batch_size, samples_per_batch=2)

def datafunction(n, s, images=True, test=False):
    return data_generator.demand(n=n, seed=s, ypcor=0.5, use_images=images, test=test)

oos_perf = data_generator.monte_carlo_error(lambda x,z,t: response_model.predict([x[:,0:1], x[:,1:],t]), datafunction, has_latent=True, debug=False)
print("Out of sample performance evaluated against the true function: %f" % oos_perf)
