from __future__ import print_function

from deepiv.models import Treatment, Response

from keras.layers import Input, Dense
from keras.models import Model

import data_generator

x, z, t, y, g_true = data_generator.linear_data(n=1000, seed=1)

print("Data shapes:\n\
Features:{x},\n\
Instruments:{z},\n\
Treament:{t},\n\
Response:{y}".format(**{'x':x.shape, 'z':z.shape,
                        't':t.shape, 'y':y.shape}))

# Build and fit treatment model
instruments = Input(shape=(z.shape[1],))
x = Dense(64, activation='relu')(instruments)
x = Dense(64, activation='relu')(x)
est_treat = Dense(1)(x)

treatment_model = Treatment(inputs=[instruments], outputs=est_treat)
treatment_model.compile('adam',
                        loss="mixture_of_gaussians",
                        n_components=10)
treatment_model.fit([z],t, epochs=20)

# Build and fit response model
x = Dense(64, activation='relu')(instruments)
x = Dense(64, activation='relu')(x)
est_resp = Dense(1)(x)
response_model = Response(treatment=treatment_model,
                          inputs=[instruments],
                          outputs=est_resp)
response_model.compile('adam', loss='mse')
response_model.fit([z], y, epochs=20, verbose=1,
                    batch_size=50, samples_per_batch=200)
