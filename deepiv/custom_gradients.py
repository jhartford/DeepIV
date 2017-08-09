from __future__ import absolute_import, division, print_function, unicode_literals

import keras
from keras import backend as K
if K.backend() == "theano":
    import theano.tensor as tensor
    Lop = tensor.Lop
elif K.backend() == "tensorflow":
    import tensorflow as tf
    def Lop(output, wrt, eval_points):
        grads = tf.gradients(output, wrt, grad_ys=eval_points)
        return grads
import types

# Used to modify the default keras Optimizer object to allow
# for custom gradient computation.

def get_gradients(self, loss, params):
    '''
    Replacement for the default keras get_gradients() function.
    Modification: checks if the object has the attribute grads and 
    returns that rather than calculating the gradients using automatic
    differentiation. 
    '''
    if hasattr(self, 'grads'):
        grads = self.grads
    else:
        grads = K.gradients(loss, params)
    if hasattr(self, 'clipnorm') and self.clipnorm > 0:
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
    if hasattr(self, 'clipvalue') and self.clipvalue > 0:
        grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
    return grads

def replace_gradients_mse(model, opt, batch_size, n_samples = 1):
    '''
    Replace the gradients of a Keras model with mean square error loss.
    '''
    # targets has been repeated twice so the below creates two identical columns
    # of the target values - we'll only use the first column.
    targets = K.reshape(model.targets[0], (batch_size, n_samples * 2))
    output =  K.mean(K.reshape(model.outputs[0], (batch_size, n_samples, 2)), axis=1)
    # compute d Loss / d output
    dL_dOutput = (output[:,0] - targets[:,0]) * (2.) / batch_size
    # compute (d Loss / d output) (d output / d theta) for each theta
    trainable_weights = model.trainable_weights
    grads = Lop(output[:,1], wrt=trainable_weights, eval_points=dL_dOutput) 
    # compute regularizer gradients

    # add loss with respect to regularizers
    reg_loss = model.total_loss * 0.
    for r in model.losses:
         reg_loss += r
    reg_grads = K.gradients(reg_loss, trainable_weights)
    grads = [g+r for g,r in zip(grads, reg_grads)]
    
    opt = keras.optimizers.get(opt)
    # Patch keras gradient calculation to allow for user defined gradients
    opt.get_gradients = types.MethodType( get_gradients, opt )
    opt.grads = grads
    model.optimizer = opt
    return model

def build_mc_mse_loss(n_samples):
    def mc_mse(y_true, y_predicted):
        n_examples = y_true.shape[0] /  n_samples / 2
        targets = y_true.reshape((n_examples , n_samples * 2))
        output = y_predicted.reshape((n_examples, n_samples * 2)).mean(axis=1)
        return K.mean(K.square(targets[:,0] - output))
    return mc_mse

