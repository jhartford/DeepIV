from __future__ import print_function
import data_generator
import numpy

import warnings

from deepiv.models import Treatment, Response
import deepiv.architectures as architectures
import deepiv.densities as densities

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
# tf.executing_eagerly()


n = 1000
dropout_rate = min(1000./(1000. + n), 0.5)
epochs = int(1500000./float(n))  # heuristic to select number of epochs
epochs = 100  # 300
batch_size = 100
images = False


def datafunction(n, s, images=images, test=False):
    return data_generator.demand(n=n, seed=s, ypcor=0.5, use_images=images, test=test)


x, z, t, y, g_true = datafunction(n, 1)  # g_true is the ture function, t is the treatment (price in the paper)

print("Data shapes:\n\
Features:{x},\n\
Instruments:{z},\n\
Treament:{t},\n\
Response:{y}".format(**{'x': x.shape, 'z': z.shape,
                        't': t.shape, 'y': y.shape}))

# Build and fit treatment model
instruments = Input(shape=(z.shape[1],), name="instruments")
features = Input(shape=(x.shape[1],), name="features")
treatment_input = Concatenate(axis=1)([instruments, features])

hidden = [128, 64, 32]

act = "relu"

n_components = 10
# first step


def treatment_output(x):
    return densities.mixture_of_gaussian_output(x, n_components)  # Concatenate(axis=1)([pi, mu, log_sig])


print(treatment_input)
# est_treat is  Concatenate(axis=1)([pi, mu, log_sig])
est_treat = architectures.feed_forward_net(treatment_input, treatment_output,
                                           hidden_layers=hidden,
                                           dropout_rate=dropout_rate, l2=0.0001,
                                           activations=act)

print("Input", instruments.shape)
print("est_treat", est_treat.shape)
treatment_model = Treatment(inputs=[instruments, features], outputs=est_treat)
treatment_model.compile('adam',
                        loss="mixture_of_gaussians",
                        n_components=n_components)

treatment_model.fit([z, x], t, epochs=epochs, batch_size=batch_size)

# Build and fit response model, t is the treatment

treatment = Input(shape=(t.shape[1],), name="treatment")  # placeholder for treatment from treatment model
response_input = Concatenate(axis=1)([features, treatment])
print("response input shape:", response_input.shape)
est_response = architectures.feed_forward_net(response_input, Dense(1),
                                              activations=act,
                                              hidden_layers=hidden,
                                              l2=0.001,
                                              dropout_rate=dropout_rate)

response_model = Response(treatment=treatment_model,
                          inputs=[features, treatment],
                          outputs=est_response)
response_model.compile('adam', loss='mse')  # unbiased_gradient=True, batch_size=batch_size)
response_model.fit([z, x], y, epochs=epochs, verbose=1,
                   batch_size=batch_size, samples_per_batch=2)

# monte_carlo_error(g_hat, data_fn, ntest=5000, has_latent=False, debug=False):
oos_perf = data_generator.monte_carlo_error(lambda x, z, t: response_model.predict(
    [x, t]), datafunction, has_latent=images, debug=False)
print("Out of sample performance evaluated against the true function: %f" % oos_perf)


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-n', '--n_samples', help='Number of training samples', default=1000, type=int)
parser.add_argument('-s', '--seed', help='Random seed', default=1, type=int)
parser.add_argument('--endo', help='Endogeneity', default=0.5, type=float)
parser.add_argument('--heartbeat', help='Use philly heartbeat', action='store_true')
parser.add_argument('--results', help='Results file', default='twosls.csv')
args = parser.parse_args()

# prepare_file("./results/DeepIV_results.csv")
# with open("DeepIV_results.csv", 'a') as f:
#     f.write('%d,%d,%f,%f\n' % (args.n_samples, args.seed, args.endo, oos_perf))
