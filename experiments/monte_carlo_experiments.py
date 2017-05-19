from __future__ import absolute_import, division, print_function, unicode_literals

import data_generator as dg
from causenet.utils import floatx, unzip
import numpy as np
import keras
from causenet_keras import architectures, densities, samplers, custom_gradients
from keras.layers import Dense, merge
import alternative_dgps

def reshape_inp(data):
    '''
    Helper function to reshape data into the expected shape for the neural net
    '''
    new_data = []
    for d in data:
        if len(d.shape) == 1:
            d = d.reshape(-1,1)
        new_data.append(d)
    return new_data


def cause_net_keras_simple(x,z,t,y, epocs=1000,dropout_rate=0.5, l2=0.001, resp_hidden=[50],
                    treatment_hidden=[50], n_mixtures=10, name='', opt='adam', dg=None, args=None,
                    objective='mse', n_samples = 1, grad_debug=None):
    from causenet_keras import architectures, densities, samplers, custom_gradients
    import keras
    from keras import backend as K
    def dataGenerator(X,Z,Y,sampler, batch_size=200, n_samples = 2, seed=123):
        n = X.shape[0]
        rng = np.random.RandomState(seed)
        batch_size = min(batch_size, n)
        idx = np.arange(n)
        while 1:
            idx = rng.permutation(idx)
            X = X[idx,:]
            Z = Z[idx,:]
            Y = Y[idx,:]
            for i in range(n/batch_size):
                x_train = X[i*batch_size:(i+1)*batch_size, :]
                z_train = Z[i*batch_size:(i+1)*batch_size, :]
                s = sampler([x_train,z_train], n_samples)
                k_input = [x_train.repeat(n_samples, axis=0), s]
                y_train = Y[i*batch_size:(i+1)*batch_size, :].repeat(n_samples, axis=0)
                yield k_input, y_train
    
    def build_ghat(model):
        def g_hat(x,z,t):
            x,z,t = reshape_inp([x,z,t])
            y = model.predict([x,t])
            return y
        return g_hat

    mix_gaussian_output, mix_gaussian_loss = densities.mixture_gaussian(n_mixtures)

    print("Using naive MLP architecture")
    inp_x = keras.layers.Input(shape=(x.shape[1],))
    inp_z = keras.layers.Input(shape=(z.shape[1],))
    inp = [inp_x, inp_z]
    inp_treat = merge(inp, mode='concat')
    out = architectures.feed_forward_net(inp_treat, mix_gaussian_output,
                        hidden_layers=treatment_hidden,
                        dropout_rate=dropout_rate,
                        activations='tanh')
    model_treatment = keras.models.Model(input=inp, output=out)
    data_input = [x,z]

    model_treatment.compile(optimizer=opt, loss=mix_gaussian_loss)
    model_treatment.fit(data_input, t,batch_size=100, nb_epoch=epocs)

    if name != '':
        name += '_'
    
    model_treatment.save_weights('weights/'+name+'treatment_model_weights.h5')

    inp_x = keras.layers.Input(shape=(x.shape[1],))
    inp_t = keras.layers.Input(shape=(t.shape[1],))
    inp = [inp_x,inp_t]
    inp_res = merge(inp, mode='concat')
    out_res = architectures.feed_forward_net(inp_res, keras.layers.Dense(1, activation='linear'), 
                                            activations='relu',
                                            hidden_layers=resp_hidden, l2=l2, dropout_rate=dropout_rate
                                            )
    model_res = keras.models.Model(input=inp, output=out_res)

    mc_mse = custom_gradients.build_mc_mse_loss(n_samples)

    model_res.compile(optimizer=opt, loss=mc_mse if objective == 'mc_mse' else 'mse')
    
    #TODO: FIX!
    if args is not None:
        if args.indep:
            print("Using %d independent samples" % n_samples)
            model_res = custom_gradients.replace_gradients_mse(model_res, opt, n_samples=n_samples)
        else:
            print("Using %d dependent samples" % n_samples)
    
    sampler = samplers.gmm_sampler(model_treatment, n_mixtures)
    batch_size = 100
    print(x.shape, z.shape, y.shape)
    model_res.fit_generator(dataGenerator(x,z,y,sampler, batch_size=batch_size, n_samples=n_samples * 2), 2*n_samples*x.shape[0], epocs)

    model_res.save_weights('weights/'+name+'response_weights.h5')
    ghat = build_ghat(model_res)
    return ghat, (model_treatment, model_res)


def mlp_embedding(images, output, other_features = [], dropout_rate=0.1,
                  embedding_dropout=0.1, embedding_l2=0.1):
    x_embedding = architectures.feed_forward_net(images, Dense(64, activation='linear'),
                        hidden_layers=[256,128,128],
                        dropout_rate=embedding_dropout,
                        activations='relu',
                        l2=embedding_l2)

    if len(other_features) > 0:
        embedd = merge([x_embedding] + other_features, mode='concat')
    else:
        embedd = x_embedding
    out = architectures.feed_forward_net(embedd, output,
                        hidden_layers=[32],
                        dropout_rate=dropout_rate,
                        activations='relu')
    return out

def conv_embedding(images, output, other_features = [], dropout_rate=0.1,
                   embedding_dropout=0.1, embedding_l2=0.05, constrain_norm=True):
    print("Building conv net")
    x_embedding = architectures.convnet(images, Dense(64, activation='linear'),
                        dropout_rate=embedding_dropout,
                        activations='relu',
                        l2_rate=embedding_l2, constrain_norm=constrain_norm)

    if len(other_features) > 0:
        embedd = merge([x_embedding] + other_features, mode='concat')
    else:
        embedd = x_embedding
    out = architectures.feed_forward_net(embedd, output,
                        hidden_layers=[32],
                        dropout_rate=dropout_rate,
                        activations='relu', constrain_norm=constrain_norm)
    return out

def cause_net_keras_embedding(x,z,t,y, epocs=1000,dropout_rate=0.5, l2=0.001, resp_hidden=[50],
                    treatment_hidden=[50], n_mixtures=10, name='', opt='adam', embedding='mlp', args=None,
                    valid=None, objective='mc_mse',n_samples = 1):
    import keras
    from keras.layers import Dense, merge
    def dataGenerator(X,Z,Y,sampler, batch_size=200, n_samples = 2, convolution=False, return_y=True, seed=None):
        n = X.shape[0]
        idx = np.arange(n)
        batch_size = min(batch_size, n)
        rng = np.random.RandomState(seed)
        while 1:
            idx = rng.permutation(idx)
            X = X[idx,:]
            Z = Z[idx,:]
            Y = Y[idx,:]
            for i in range(n/batch_size):
                x_train = X[i*batch_size:(i+1)*batch_size, :]
                z_train = Z[i*batch_size:(i+1)*batch_size, :]
                time = x_train[:,0].reshape(-1,1)
                image = x_train[:,1:]
                if convolution:
                    image = image.reshape((-1,1,28,28))
                samples = sampler([image, z_train, time], n_samples)
               # samples = sampler([image, z_train, time], 1)
               # samples = samples.repeat(n_samples, axis=0)
                k_input = [time.repeat(n_samples, axis=0), image.repeat(n_samples, axis=0), samples]
                y_train = Y[i*batch_size:(i+1)*batch_size, :].repeat(n_samples, axis=0)
                if return_y:
                    yield k_input, y_train
                else:
                    yield k_input
    
    def build_ghat(model, embedding='mlp'):
        def g_hat(x,z,t):
            x,z,t = reshape_inp([x,z,t])
            time = x[:,0]
            image = x[:,1:]
            if embedding == 'conv':
                image = image.reshape((-1,1,28,28))
            return model.predict([time,image,t])
        return g_hat

    mix_gaussian_output, mix_gaussian_loss = densities.mixture_gaussian(n_mixtures)

    if args is not None:
        embedding_dropout = args.e_dpo
        embedding_l2 = args.e_l2
    else:
        embedding_dropout = 0.1
        embedding_l2 = 0.05
    print("Treatment using MLP embedding architecture")
    shape = (x.shape[1] - 1,) if embedding=='mlp' else (1,28,28)
    images = keras.layers.Input(shape=shape, name='treat_images')
    time = keras.layers.Input(shape=(1,), name='treat_time')
    instruments = keras.layers.Input(shape=(z.shape[1],), name='treat_instruments')
    if embedding == 'mlp':
        treatment_output = mlp_embedding(images, mix_gaussian_output, [time, instruments], dropout_rate=dropout_rate)
        treatment_input_data = [x[:,0], x[:,1:],z]
    elif embedding == 'conv':
        treatment_output = conv_embedding(images, mix_gaussian_output, [time, instruments], 
                                          dropout_rate=dropout_rate, embedding_dropout=embedding_dropout,
                                          embedding_l2=embedding_l2)
        treatment_input_data = [x[:,0], x[:,1:].reshape((-1,1,28,28)),z]
    model_treatment = keras.models.Model(input=[time, images, instruments], output=treatment_output)
    model_treatment.compile(optimizer=opt, loss=mix_gaussian_loss)
    model_treatment.fit(treatment_input_data, t,batch_size=100, nb_epoch=epocs, verbose=2)

    if name != '':
        name += '_'
    model_treatment.save_weights('weights/'+name+'treatment_model_weights.h5')
    #model_treatment.load_weights('weights/'+name+'treatment_model_weights.h5')

    print("Response using MLP embedding architecture")
    res_images = keras.layers.Input(shape=shape)
    res_inp_time = keras.layers.Input(shape=(1,))
    res_inp_t = keras.layers.Input(shape=(t.shape[1],))
    if embedding == 'mlp':
        out_res = mlp_embedding(res_images, keras.layers.Dense(1, activation='linear'), [res_inp_time, res_inp_t], dropout_rate = dropout_rate)
    elif embedding == 'conv':
        out_res = conv_embedding(res_images, keras.layers.Dense(1, activation='linear'), [res_inp_time, res_inp_t],
                dropout_rate=dropout_rate, embedding_dropout=embedding_dropout, embedding_l2=embedding_l2)
    
    model_res = keras.models.Model(input=[res_inp_time, res_images, res_inp_t], output=out_res)
    mc_mse = custom_gradients.build_mc_mse_loss(n_samples)
    model_res.compile(optimizer=opt, loss=mc_mse if objective == 'mc_mse' else 'mse')

    if args is not None:
        if args.indep:
            print("Using %d independent samples" % n_samples)
            model_res = custom_gradients.replace_gradients_mse(model_res, opt, n_samples=n_samples)
        else:
            print("Using dependent samples")
    sampler = samplers.gmm_sampler(model_treatment, n_mixtures)

    batch_size = 100
    model_res.fit_generator(dataGenerator(x,z,y,sampler, batch_size=batch_size, convolution=embedding=='conv', n_samples=n_samples * 2),
                            2 * n_samples*x.shape[0], epocs)


    model_res.save_weights('weights/'+name+'response_weights.h5')
    n_samples = 1000
    if valid is not None:
        print("Evaluating model")
        x_valid,z_valid,t_valid,y_valid, _ = valid
        y_hat = np.zeros(y_valid.shape[0])
        bs = 50
        for i in xrange(y_hat.shape[0] / bs):
            time = x_valid[i*bs:(i+1)*bs,0:1]
            image = x_valid[i*bs:(i+1)*bs,1:]
            if embedding == 'conv':
                image = image.reshape((-1,1,28,28))

            samples = sampler([image, z_valid[i*bs:(i+1)*bs,:], time], n_samples)
            f_input = [time.repeat(n_samples, axis=0), image.repeat(n_samples, axis=0), samples]
            y_hat[i*(bs):(i+1)*(bs)] = model_res.predict(f_input).reshape((bs,n_samples)).mean(axis=1)
        print('Done with y_hat')
        with open('validation_performance.csv', 'a') as f:
            perf = ((y_valid - y_hat)**2).mean()
            callstring = '-'.join(sys.argv[1:])
            f.write(name+','+callstring+','+str(perf)+'\n')
    ghat = build_ghat(model_res, embedding=embedding)
    return ghat, (model_treatment, model_res)

def cause_net(x,z,t,y,verbose=0, epocs=1000, 
            resp_hidden=[128,64,32], treatment_hidden=[128,64,32],
            dropout_rate=0.5, l2 = 0.1, 
            name='', error_true=None, save_freq=100,
            response_loss='mc_sqr_error', t_out='mix_gaussian',
            exact_integral=False, dropout_treatment=False, data_fn=None, args=None):
    import causenet.causal_net_model as causal_net_model
    def build_ghat(c):
        def g_hat(x,z,t):
            x,z,t = reshape_inp([x,z,t])
            y = c.predict(floatx(x), floatx(t))
            return y
        return g_hat
    CausalNetModel = causal_net_model.CausalNetModel

    c = CausalNetModel(feature_dim=x.shape[1], instrument_dim=z.shape[1],
                       mixture_components=10, treatment_hidden=treatment_hidden,
                       response_hidden=resp_hidden,independent_sampling=args.indep,
                       opt='adam', learning_rate=1.,n_samples=1,dropout=True,
                       dropout_treatment=dropout_treatment, exact_integral=exact_integral,
                       dropout_rate=dropout_rate, l2 = l2, t_out=t_out,
                       treatment_dim=t.shape[1], response_loss=response_loss,
                       response_dim=y.shape[1])
    if verbose==0:
        print_freq = epocs
    elif verbose==1:
        print_freq = epocs / 20
    else:
        print_freq = 1000
    n = x.shape[0]
    idx = np.random.permutation(n)
    n = int(n * 0.8)
    bs = 100
    train = dg.prepare_datastream(x[idx[0:n],:],z[idx[0:n],:],t[idx[0:n],:],y[idx[0:n],:], shuffle=True, batch_size=bs)
    validation = dg.prepare_datastream(x[idx[n:],:],z[idx[n:],:],t[idx[n:],:],y[idx[n:],:], shuffle=False, batch_size=bs)
    weight_name = 'monte_weights' + (('_' + name) if name !='' else '')
    c.fit_treatment(train=train, validation=validation, max_epocs=epocs, threshold=-np.inf,
                    print_freq=print_freq, save_weight=weight_name)

    if error_true is None:
        c.fit_response(train=train, validation=validation, max_epocs=epocs,
                       print_freq=print_freq, save_weight=weight_name, 
                       save_freq=save_freq,eval_samples=1, continuous_save=1000,
                       update_samples=5)
        g_hat = build_ghat(c)
    else:
        epoc_per_check = 100
        nchecks = epocs // epoc_per_check 
        int_file = 'results/intermediate_results/'+name+'.csv'
        with open(int_file, 'w') as f:
            f.write('epoc,error,train,valid\n')
        best = None
        best_score = None
        for i in xrange(nchecks):
            score = c.fit_response(train=train, validation=validation, max_epocs=epoc_per_check,
                        print_freq=print_freq, save_weight=weight_name, save_freq=save_freq,
                        epoc_offset=i*epoc_per_check)
            best = unzip(c.tparams)
            g_hat = build_ghat(c)
            error = error_true(g_hat)
            with open(int_file, 'a') as f:
                f.write('%d,%f,%f,%f\n' % (i*epoc_per_check, error, score[0], score[1]))
    
    return g_hat, c

def nnet(x,z,t,y,hiddens=[50], images=False, activation="relu", dropout_rate=0., epocs=100, verbose=2, error_true=None):
    '''
    Simple feedforward neural network implementation for comparison purposes.
    
    Takes features, instruments and the treatment and fits a neural network on the
    response directly using all three as input.
    '''
    # keras imports
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    # prepare data
    x,z,t,y = reshape_inp([x,z,t,y])
    if images:
        images = keras.layers.Input(shape=(1,28,28))
        inp_time = keras.layers.Input(shape=(1,))
        inp_t = keras.layers.Input(shape=(1,))
        inp_z = keras.layers.Input(shape=(1,))
        out = conv_embedding(images, keras.layers.Dense(1, activation='linear'), [inp_time, inp_t, inp_z],
                            dropout_rate=dropout_rate, embedding_dropout=0.4, embedding_l2=0.1)
        model = keras.models.Model(input=[inp_time, images, inp_t, inp_z], output=out)
        model_input = [x[:,0].reshape(-1,1), x[:,1:].reshape((-1,1,28,28)), t, z]
    else:
        X = np.concatenate([x,z,t], axis=1)
        inp = keras.layers.Input(shape=(X.shape[1],))
        out = architectures.feed_forward_net(inp, Dense(1),
                        hidden_layers=hiddens,
                        dropout_rate=dropout_rate,
                        activations=activation)
        model = keras.models.Model(input=inp, output=out)
        model_input = [X]

    model.compile(loss='mse', optimizer='adam')
    # train network
    model.fit(model_input, y, nb_epoch=epocs, batch_size=np.minimum(x.shape[0], 100), verbose=verbose)
    def g_hat(x,z,t):
        x,z,t = reshape_inp([x,z,t])
        if images:
            inpt = [x[:,0].reshape(-1,1), x[:,1:].reshape((-1,1,28,28)), t, z]
        else:
            inpt = np.concatenate([x,z,t], axis=1)
        return model.predict(inpt)
    return g_hat, model

def twosls(x, z, t, y, feat_in_first_stage=False, error_true=None, alpha=0.000001):
    '''
    Simple two-stage least squares implementation (doesn't do correct inference, etc.).
    Just fits a linear regression for predicting the treatment in stage 1 and a linear regression for 
    predicting the response in stage 2.
    
    feat_in_first_stage determines whether to use the exogenous features, x, in stage 1.
    '''
    from sklearn.linear_model import Ridge
    stage_1 = Ridge(alpha=alpha)
    #x = x[:,0]
    #x = x.reshape(-1,1)
    if feat_in_first_stage:
        stage_1.fit(np.concatenate([x,z], axis=1),t)
        t_hat = stage_1.predict(np.concatenate([x,z], axis=1))
    else:
        stage_1.fit(z,t)
        t_hat = stage_1.predict(z)
    stage_2 = Ridge(alpha=alpha)
    stage_2.fit(np.concatenate([x,t_hat], axis=1), y)
    y_hat = stage_2.predict(np.concatenate([x,t_hat], axis=1))
    def g_hat(x,z,t):
        #x = x[:,0]
        #x = x.reshape(-1,1)
        X = np.concatenate([np.ones((x.shape[0], 1)), x, t], axis=1)
        W = np.array([stage_2.intercept_]+list(stage_2.coef_.flatten())).reshape((-1,1))
        return np.dot(X,W)
    return g_hat, stage_2

def get_uncertainty(inp, model, l2, p, l, T=100):
    N = inp[0].shape[0]
    preds = []
    for _ in xrange(T):
        preds += [model(inp).flatten()]
    predictive_mean = np.mean(preds, axis=0)
    predictive_variance = np.var(preds, axis=0)
    tau = l**2 * (p) / (2 * N * l2)
    predictive_variance += 0*tau**-1
    return predictive_mean, predictive_variance

def monte_carlo_error(g_hat, data_fn, ntest=5000, has_latent=False, debug=False):
    seed = np.random.randint(ntest)
    try:
        x, z, t, y, g_true = data_fn(ntest, seed, test=True) # test = True ensures we draw test set images
    except ValueError:
        ntest = int(ntest * 0.7)
        x, z, t, y, g_true = data_fn(ntest, seed, test=True) # test = True ensures we draw test set images


    ## re-draw to get new independent treatment and implied response
    t = np.linspace(np.percentile(t,2.5),np.percentile(t,97.5),ntest).reshape(-1,1)
    ## we need to make sure z _never_ does anything in these g functions (fitted and true)
    ## above is necesary so that reduced form doesn't win
    if has_latent:
        x_latent, _, _, _, _ = data_fn(ntest, seed, images=False)
        y = g_true(x_latent,z,t)
    else:
        y = g_true(x,z,t)
    [x,z,t,y] = reshape_inp([x,z,t,y])
    x, z, t = [floatx(i) for i in [x, z, t]]
    y_true = y.flatten()# g_true(x,z,t).flatten()
    y_hat = g_hat(x,z,t).flatten()
    return ((y_hat - y_true)**2).mean()

def run_experiment(N, seed, data_fn, models=['cause_net', 'nnet', 'twosls'], 
                   max_epocs=1000, dropout_rate=0.5, l2=0.01, 
                   data_name='', intermediate_results=False, args=None, images=False, validation=True):
    x, z, t, y, g_true = data_fn(N, seed)
    data = [floatx(i) for i in [x,z,t,y]]
    data = reshape_inp(data)
    mod = []
    g_fn = []
    hidden = [128,64,32]
    if intermediate_results:
        error_true = lambda g: monte_carlo_error(g, data_fn)
    else:
        error_true = None
    if 'cause_net' in models:
        print("Running causenet")
        g_1, mod_1 = cause_net(*data, epocs=max_epocs,
                               dropout_rate=dropout_rate, l2 = l2, name='%s_%d'%(data_name, seed), save_freq=max_epocs,
                               error_true=error_true, exact_integral=args.exact, dropout_treatment=args.drop_treat, data_fn=data_fn,
                               args=args)
        g_fn.append(('cause_net',g_1))
    if 'nnet' in models:
        print("Running nerual net")
        g_2, mod_2 = nnet(x, z, t, y, hiddens=hidden, dropout_rate=dropout_rate, verbose=1, epocs=max_epocs, error_true=error_true, images=images)
        g_fn.append(('nnet',g_2))
    if 'twosls' in models:
        print("Running linear")
        g_3, mod_3 = twosls(x, z, t, y, error_true=error_true)
        g_fn.append(('2SLS',g_3))
    if 'cause_net_embedding' in models:
        print("Running causenet keras")
        if validation:
            valid = data_fn(5000, seed+1234)
        else:
            valid = None
            
        g_4, mod_4 = cause_net_keras_embedding(*data, epocs=max_epocs, 
                               dropout_rate=dropout_rate, l2 = l2,
                               name='%s_%d'%(data_name, seed),
                               treatment_hidden=hidden,
                               resp_hidden=hidden, args=args, valid=valid)
        g_fn.append(('cause_net_embedding',g_4))

    if 'cause_net_mlp' in models:
        print("Running causenet keras")
        g_4, mod_4 = cause_net_keras_simple(*data, epocs=max_epocs, 
                               dropout_rate=dropout_rate, l2 = l2,
                               name='%s_%d'%(data_name, seed),
                               treatment_hidden=hidden,
                               resp_hidden=hidden, n_samples=args.n_samples, 
                               args=args)
        g_fn.append(('cause_net_mlp',g_4))
    
    losses = []

    for m, g in g_fn:
        losses.append((m,monte_carlo_error(g, data_fn, has_latent=images, debug=args.debug)))
    return losses

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='store', help='data to test')
    parser.add_argument("--name", default='exp', help='experiment name')
    parser.add_argument("-n", required=True, type=int, help='sample size')
    parser.add_argument("--n_samples", default=1, type=int, help='Number of independent samples size')
    parser.add_argument("--epocs", default=-1, type=int, help='Number of epocs samples size')
    parser.add_argument("--seed", required=True, type=int, help='random seed')
    parser.add_argument("--endo", default=0.5, type=float, help='error correlation')
    parser.add_argument("--e_l2", default=0.05, type=float, help='embedding l2')
    parser.add_argument("--e_dpo", default=0.1, type=float, help='embedding dropout')
    parser.add_argument("--exact", action='store_true', help='Exact integration - only works if using discrete treatment')
    parser.add_argument("--models", default='', help='model to test', nargs='+')
    parser.add_argument("--debug", action='store_true', help='Debug mode')
    parser.add_argument("--indep", action='store_true', help='Independent gradients')
    parser.add_argument("--images", action='store_true', help='Use image features')
    parser.add_argument("--drop_treat", action='store_true', help='Use dropout on the treatment network')
    args = parser.parse_args()
    exp = args.data
    if exp == 'store':
        def datafunction(n, s, images=args.images, test=False):
            return dg.demand(n, seed=s, ypcor=args.endo, use_images=images, test=test)
        df = datafunction
    elif exp == 'olddemand':
        def datafunction(n, s, images=args.images, test=False):
            return alternative_dgps.old_demand(n, seed=s)
        df = datafunction
    elif exp == 'linear':
        def datafunction(n, s, images=args.images, test=False):
            return dg.linear_data(n, seed=s)
        df = datafunction
    
    if args.models == '':
        models = ['cause_net_mlp', 'nnet', 'twosls']
        models = ['nnet', 'twosls']
        models = ['cause_net_embedding']
    else:
        models = args.models
    if args.epocs == -1:
        max_epocs =  int(1500000./float(args.n))
    else:
        max_epocs = args.epocs

    dropout_rate = 1000./(1000. + args.n)
    if dropout_rate > 0.5:
        dropout_rate = 0.5
    losses = run_experiment(args.n,args.seed, df, models=models,  
        max_epocs=max_epocs, dropout_rate=dropout_rate,
        data_name='%s_%d'%(exp, args.n), l2=0.0001, 
        args=args, images=args.images, validation=False)

    for m, l in losses:
        exp_name = exp
        if args.images:
            exp_name += '_images'
        with open('results/%s_%s.txt' % (args.name, exp_name), 'a') as f:
            f.write(','.join([
                str(args.n), str(args.seed),str(args.images), str(args.endo), str(dropout_rate), str(max_epocs), 
                str(args.n_samples)] + [m] + [str(float(l))]) + '\n')

if __name__ == '__main__':
    sys.exit(int(main() or 0))
