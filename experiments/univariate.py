from __future__ import print_function
import data_generator
import numpy
import argparse
import warnings
import time
import os
import pickle

from deepiv.models import Treatment, Response
import deepiv.architectures as architectures
import deepiv.densities as densities

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
# tf.executing_eagerly()

parser = argparse.ArgumentParser(description='deman simulation')
parser.add_argument('--n',  help='Number of training samples', default=5000, type=int)
parser.add_argument('--n_test',  help='Number of test samples', default=5000, type=int)
parser.add_argument('--ypcor',  help='correlation between p and e', default=0.5, type=float)
parser.add_argument('--seed', help='Random seed', default=1, type=int)
parser.add_argument('--unbiased', default=False, action="store_true")
parser.add_argument('--samples_per_batch',   default=2, type=int)

parser.add_argument('--results_fn', help='Results file', default='', type=str)
parser.add_argument('--test_fn',  default='', type=str)
args = parser.parse_args()


n = args.n
dropout_rate = min(1000./(1000. + n), 0.5)
epochs = int(1500000./float(n))  # heuristic to select number of epochs
epochs = 30  # 300
batch_size = 100
images = False


def datafunction(n, s, images=images,  test=False, ypcor=0.5, ynoise=1.):
    return data_generator.demand_univariate(n=n, seed=s, ypcor=ypcor, ynoise=ynoise, use_images=images, test=test)


# g_true is the ture function, t is the treatment (price in the paper)
x, z, t, y, g_true = datafunction(n, 1, ypcor=args.ypcor)

x_test, z_test, t_test, y_test = data_generator.load_test_data(
    test_fn=args.test_fn, data_fn=datafunction, ntest=args.n_test, ypcor=args.ypcor)  # to keep consistent, using same seed as 1234


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
start = time.time()
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
# response_model.compile('adam', loss='mse')  # unbiased_gradient=True, batch_size=batch_size)
response_model.compile('adam', loss='mse', unbiased_gradient=True, batch_size=batch_size)
response_history = response_model.fit([z, x], y, epochs=epochs, verbose=1, batch_size=batch_size,
                                      samples_per_batch=2, validation_data=([x_test, t_test], y_test))

if not response_history:
    response_history = response_model.history.history
else:
    response_history = response_history.history

response_fn = "./experiments/results/response_history_univariate_N{}_cor{}Dict".format(args.n, args.ypcor)

response_fn = response_fn+"_unbiased" if args.unbiased else response_fn
response_fn = response_fn+"_S{}".format(args.samples_per_batch) if args.samples_per_batch > 2 else response_fn


print("response fn: {}".format(response_fn))
with open(response_fn, "wb") as file_pi:
    pickle.dump(response_history, file_pi)


end = time.time()
print("total training time of sample size: {} is {}".format(args.n, end-start))

results_fn = args.results_fn
if args.results_fn:
    results_fn += "_univariate"
    results_fn = args.results_fn + "_N{}_P{}".format(args.n, args.ypcor)
    results_fn = results_fn+"_unbiased" if args.unbiased else results_fn
    results_fn = results_fn+"_S{}".format(args.samples_per_batch) if args.samples_per_batch > 2 else results_fn
    print("results_fn : {}".format(results_fn))


# monte_carlo_error(g_hat, data_fn, ntest=5000, has_latent=False, debug=False):
oos_perf = data_generator.monte_carlo_error(lambda x, z, t: response_model.predict(
    [x, t]), datafunction, ntest=args.n_test, has_latent=images, debug=False, results_fn=results_fn)
print("Out of sample performance evaluated against the true function: %f" % oos_perf)


# prepare_file("./results/DeepIV_results.csv")
# with open("DeepIV_results.csv", 'a') as f:
#     f.write('%d,%d,%f,%f\n' % (args.n_samples, args.seed, args.endo, oos_perf))
#
