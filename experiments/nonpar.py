import data_generator
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
R = robjects.r

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import numpy as np

def test_points(data_fn, ntest=5000, has_latent=False, debug=False):
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
    t = np.linspace(np.percentile(t, 2.5),np.percentile(t, 97.5),ntest).reshape(-1, 1)
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
    return np.array(list(x))

def fit_and_evaluate(x,z,t,y,df):
    npr=importr('np')
    y_R = robjects.FloatVector(list(y.flatten()))
    (x_eval, t_eval), y_true = test_points(df, 10000)
    mod = npr.npregiv(y_R, t, z, x=x, zeval=t_eval, xeval=x_eval,
                    method="Tikhonov", p=0, optim_method ="BFGS")
    return ((y_true - to_array(mod.rx2('phi.eval')))**2).mean()


df = lambda n, s, test: data_generator.demand(n, s, test=test)
x,z,t,y,g = df(1000, 1, False)
print(fit_and_evaluate(x,z,t,y,df))


