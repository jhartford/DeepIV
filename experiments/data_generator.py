from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 
#from causenet.datastream import DataStream, prepare_datastream
from sklearn.preprocessing import OneHotEncoder

X_mnist = None
y_mnist = None

def monte_carlo_error(g_hat, data_fn, ntest=5000, has_latent=False, debug=False):
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
    y_hat = g_hat(x, z, t).flatten()
    return ((y_hat - y_true)**2).mean()


def loadmnist():
    '''
    Load the mnist data once into global variables X_mnist and y_mnist.
    '''
    from keras.datasets import mnist
    global X_mnist
    global y_mnist
    train, test = mnist.load_data()
    X_mnist = []
    y_mnist = []
    for d in [train, test]:
        X, y = d
        X = X.astype('float32')
        X /= 255.
        idx = np.argsort(y)
        X_mnist.append(X[idx, :, :])
        y_mnist.append(y[idx])

def get_images(digit, n, seed=None, testset=False):
    if X_mnist is None:
        loadmnist()
    is_test = int(testset)
    rng = np.random.RandomState(seed)
    X_i = X_mnist[is_test][y_mnist[is_test] == digit, :, :]
    n_i, i, j = X_i.shape
    perm = rng.permutation(np.arange(n_i))
    if n > n_i:
        raise ValueError('You requested %d images of digit %d when there are \
						  only %d unique images in the %s set.' % (n, digit, n_i, 'test' if testset else 'training'))
    return X_i[perm[0:n], :, :].reshape((n,i*j))

def one_hot(col, **kwargs):
    z = col.reshape(-1,1)
    enc = OneHotEncoder(sparse=False, **kwargs)
    return enc.fit_transform(z)

def get_test_valid_train(generator, n, batch_size=128, seed=123, **kwargs):
    x, z, t, y, g = generator(n=int(n*0.6), seed=seed, **kwargs)
    train = prepare_datastream(x, z, t, y, True, batch_size, **kwargs)
    x, z, t, y, g = generator(n=int(n*0.2), seed=seed+1, **kwargs)
    valid = prepare_datastream(x, z, t, y, False, batch_size, **kwargs)
    x, z, t, y, g = generator(n=int(n*0.2), seed=seed+2, **kwargs)
    test = prepare_datastream(x, z, t, y, False, batch_size, **kwargs)
    return train, valid, test, g

def sensf(x):
    return 2.0*((x - 5)**4 / 600 + np.exp(-((x - 5)/0.5)**2) + x/10. - 2)

def emocoef(emo):
    emoc = (emo * np.array([1., 2., 3., 4., 5., 6., 7.])[None, :]).sum(axis=1)
    return emoc

psd = 3.7
pmu = 17.779
ysd = 158.#292.
ymu = -292.1

def storeg(x, price):
    emoc = emocoef(x[:, 1:])
    time = x[:, 0]
    g = sensf(time)*emoc*10. + (emoc*sensf(time)-2.0)*(psd*price.flatten() + pmu)
    y = (g - ymu)/ysd
    return y.reshape(-1, 1)

def demand(n, seed=1, ynoise=1., pnoise=1., ypcor=0.8, use_images=False, test=False):
    rng = np.random.RandomState(seed)

    # covariates: time and emotion
    time = rng.rand(n) * 10
    emotion_id = rng.randint(0, 7, size=n)
    emotion = one_hot(emotion_id)
    if use_images:
        idx = np.argsort(emotion_id)
        emotion_feature = np.zeros((0, 28*28))
        for i in range(7):
            img = get_images(i, np.sum(emotion_id == i), seed, test)
            emotion_feature = np.vstack([emotion_feature, img])
        reorder = np.argsort(idx)
        emotion_feature = emotion_feature[reorder, :]
    else:
        emotion_feature = emotion

    # random instrument
    z = rng.randn(n)

    # z -> price
    v = rng.randn(n)*pnoise
    price = sensf(time)*(z + 3)  + 25.
    price = price + v
    price = (price - pmu)/psd

    # true observable demand function
    x = np.concatenate([time.reshape((-1, 1)), emotion_feature], axis=1)
    x_latent = np.concatenate([time.reshape((-1, 1)), emotion], axis=1)
    g = lambda x, z, p: storeg(x, p) # doesn't use z

    # errors 
    e = (ypcor*ynoise/pnoise)*v + rng.randn(n)*ynoise*np.sqrt(1-ypcor**2)
    e = e.reshape(-1, 1)
    
    # response
    y = g(x_latent, None, price) + e

    return (x,
            z.reshape((-1, 1)),
            price.reshape((-1, 1)),
            y.reshape((-1, 1)),
            g)


def linear_data(n, seed=None, sig_d=0.5, sig_y=2, sig_t=1.5,
				alpha=4, noiseless_t=False, **kwargs):
    rng = np.random.RandomState(seed)
    nox = lambda z, d: z + 2*d
    house_price = lambda alpha, d, nox_val: alpha + 4*d + 2*nox_val

    d = rng.randn(n) * sig_d
    law = rng.randint(0, 2, n)

    if noiseless_t:
        t = nox(law, d.mean()) + sig_t*rng.randn(n)
    else:
        t = (nox(law, d) + sig_t*rng.randn(n) - 0.5) / 1.8
    z = law.reshape((-1, 1))
    x = np.zeros((n, 0))
    y = (house_price(alpha, d, t) + sig_y*rng.randn(n) - 5.)/5.
    g_true = lambda x, z, t: house_price(alpha, 0, t)
    return x, z, t.reshape((-1, 1)), y.reshape((-1, 1)), g_true


def main():
    pass

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
