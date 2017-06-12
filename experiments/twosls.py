import argparse
import os

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.metrics.pairwise import polynomial_kernel
import numpy as np

import data_generator

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-n','--n_samples', help='Number of training samples', default=1000, type=int)
parser.add_argument('-s','--seed', help='Random seed', default=1, type=int)
parser.add_argument('--endo', help='Endogeneity', default=0.5, type=float)
parser.add_argument('--heartbeat', help='Use philly heartbeat', action='store_true')
parser.add_argument('--results', help='Results file', default='twosls.csv')
args = parser.parse_args()

def fit_twosls(x, z, t, y):
    '''
    Two stage least squares with polynomial basis function.
    '''
    params = dict(poly__degree=range(1,4),
                  ridge__alpha=np.logspace(-5, 5, 11))
    pipe = Pipeline([('poly', PolynomialFeatures()),
                        ('ridge', Ridge())])
    stage_1 = GridSearchCV(pipe, param_grid=params, cv=5)
    if z.shape[1] > 0:
        X = np.concatenate([x,z], axis=1)
    else:
        X = z
    stage_1.fit(X,t)
    t_hat = stage_1.predict(X)
    print("First stage paramers: " + str(stage_1.best_params_ ))

    pipe2 = Pipeline([('poly', PolynomialFeatures()),
                        ('ridge', Ridge())])
    stage_2 = GridSearchCV(pipe2, param_grid=params, cv=5)
    X2 = np.concatenate([x,t_hat], axis=1)
    stage_2.fit(X2, y)
    print("Best in sample score: %f" % stage_2.score(X2, y))
    print("Second stage paramers: " + str(stage_2.best_params_  ))

    def g_hat(x,z,t):
        X_new = np.concatenate([x, t], axis=1)
        return stage_2.predict(X_new)
    return g_hat

def prepare_file(filename):
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('n,seed,endo,mse\n')

df = lambda n, s, test: data_generator.demand(n, s, ypcor=args.endo, test=test)
x,z,t,y,g = df(args.n_samples, args.seed, False)
g_hat = fit_twosls(x,z,t,y)

oos_perf = data_generator.monte_carlo_error(g_hat, df, has_latent=False, debug=False)
print("Out of sample performance evaluated against the true function: %f" % oos_perf)

prepare_file(args.results)
with open(args.results, 'a') as f:
    f.write('%d,%d,%f,%f\n' % (args.n_samples, args.seed,args.endo, oos_perf))

