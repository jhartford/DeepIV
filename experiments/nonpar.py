import argparse
import os
import time

import data_generator

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from threading import Thread
R = robjects.r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import numpy as np

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-n','--n_samples', help='Number of training samples', default=1000, type=int)
parser.add_argument('-s','--seed', help='Random seed', default=1, type=int)
parser.add_argument('--endo', help='Endogeneity', default=0.5, type=float)
parser.add_argument('--heartbeat', help='Use philly heartbeat', action='store_true')
parser.add_argument('--results', help='Results file', default='nonpar_iv.csv')
args = parser.parse_args()

DONE = False # global variable for the heartbeat

def dummy_heartbeat(freq = 20, timeout=100):
    '''
    Heartbeat function that outputs a dummy progress bar.

    Only necessary for our cluster.
    '''
    i = 0
    global DONE
    while not DONE:
        print("PROGRESS: %1.2f%%" % (100*float(i) / timeout))
        for _ in xrange(freq):
            # check if we're done every second
            if DONE:
                break
            time.sleep(1)
        i += 1

if args.heartbeat:
    t = Thread(target=lambda:dummy_heartbeat(60, 10)) 
    t.start()


def test_points(data_fn, ntest=5000, has_latent=False, debug=False):
    '''
    Generate and return test set points with their true values.
    '''
    seed = np.random.randint(1e9)
    try:
        # test = True ensures we draw test set images
        x, z, t, y, g_true = data_fn(ntest, seed, test=True)
    except ValueError:
        warnings.warn("Too few images, reducing test set size")
        ntest = int(ntest * 0.7)
        # test = True ensures we draw test set images
        x, z, t, y, g_true = data_fn(ntest, seed, test=True)

    ## re-draw to get new independent treatment and implied response
    t = np.linspace(np.percentile(t, 2.5), np.percentile(t, 97.5), ntest).reshape(-1, 1)
    ## we need to make sure z _never_ does anything in these g functions (fitted and true)
    ## above is necesary so that reduced form doesn't win
    if has_latent:
        x_latent, _, _, _, _ = data_fn(ntest, seed, images=False)
        y = g_true(x_latent, z, t)
    else:
        y = g_true(x, z, t)
    y_true = y.flatten()
    return (x,t), y_true

def to_array(x):
    '''
    Convert r vector to numpy array
    '''
    return np.array(list(x))

def fit_and_evaluate(x,z,t,y,df):
    '''
    Fit and evaluate non-parametric regression using  Darolles, Fan, Florens and Renault (2011)

    Implemented in the `np` package in R.

    See [the np package documation](https://cran.r-project.org/web/packages/np/np.pdf) for details.
    '''
    npr=importr('np')
    y_R = robjects.FloatVector(list(y.flatten()))
    (x_eval, t_eval), y_true = test_points(df, 10000)
    mod = npr.npregiv(y_R, t, z, x=x, zeval=t_eval, xeval=x_eval,
                    method="Tikhonov", p=0, optim_method ="BFGS")
    return ((y_true - to_array(mod.rx2('phi.eval')))**2).mean()

def prepare_file(filename):
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('n,seed,endo,mse\n')


df = lambda n, s, test: data_generator.demand(n, s, ypcor=args.endo, test=test)
x,z,t,y,g = df(args.n_samples, args.seed, False)
mse = fit_and_evaluate(x,z,t,y,df)
DONE = True # turn off the heartbeat

prepare_file(args.results)
with open(args.results, 'a') as f:
    f.write('%d,%d,%f,%f\n' % (args.n_samples, args.seed, args.endo, mse))

