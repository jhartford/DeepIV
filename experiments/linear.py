from __future__ import print_function

from deepiv.models import Treatment, Response
import deepiv.densities as densities

from keras.layers import Input, Dense
from keras.models import Model

import data_generator

epochs = 100
# Very simple linear data where two stage least squares is optimal
x, z, t, y, g_true = data_generator.linear_data(n=1000, seed=1)

print("Starting experiment with linear data\n" + "-"*50 +
"\nData shapes:\n\
Features:{x},\n\
Instruments:{z},\n\
Treament:{t},\n\
Response:{y}".format(**{'x':x.shape, 'z':z.shape,
                        't':t.shape, 'y':y.shape}))

# Build and fit treatment model
instruments = Input(shape=(z.shape[1],))
x = Dense(64, activation='relu')(instruments)
x = Dense(64, activation='relu')(x)
#est_treat = Dense(1)(x)
est_treat = densities.mixture_of_gaussian_output(x, 10)

treatment_model = Treatment(inputs=[instruments], outputs=est_treat)
treatment_model.compile('adam',
                        loss="mixture_of_gaussians",
                        n_components=10)
treatment_model.fit([z],t, epochs=epochs)

# Build and fit response model
x = Dense(64, activation='relu')(instruments)
x = Dense(64, activation='relu')(x)
est_resp = Dense(1)(x)
response_model = Response(treatment=treatment_model,
                          inputs=[instruments],
                          outputs=est_resp)
response_model.compile('adam', loss='mse')
response_model.fit([z], y, epochs=epochs, verbose=1,
                    batch_size=100, samples_per_batch=2)

def datafunction(n, s, images=False, test=False):
    return data_generator.linear_data(n=n, seed=s)

oos_perf = data_generator.monte_carlo_error(lambda x,z,t: response_model.predict([t]), datafunction, has_latent=False, debug=False)
print("Out of sample performance evaluated against the true function: %f" % oos_perf)
