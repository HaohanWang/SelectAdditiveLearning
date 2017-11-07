from __future__ import division

__author__ = 'Haohan Wang'

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import cPickle
from collections import OrderedDict

from utility.optimizers_RLEM import Optimizer

epsilon = 1e-7
seed = 0


def tanh(x):
    return T.tanh(x)


def relu(x):
    return T.maximum(epsilon, x)


def linear(x):
    return x


def sigmoid(x):
    return T.nnet.sigmoid(x)


activation = tanh
activation_cnn = tanh
activation_output = T.nnet.softmax

optimize = Optimizer()
opt = optimize.adagrad


class SALModel:
    def __init__(self, continuous, n_class,
                 filterShapeText, poolSizeText, imageShapeText, dropOutSizeText,
                 x_train_fix, y_train, groupId,  x_valid_fix, y_valid, groupId_valid, x_test_fix, y_test,
                 b1=5, b2=1, batch_size=100, learning_rate=0.001, lam=0, dropOutRate1=0.5, dropOutRate2=0.5,
                 sigmaBias=0.01, yLam=0.1):
        self.continuous = continuous
        # self.hu_encoder = hu_encoder
        # self.variance_encoder = variance_encoder
        # self.n_latent = n_latent
        [self.N, self.features] = x_train_fix.shape
        self.groupIDsize = groupId.shape[1]
        self.validSize = y_train.shape[0]
        self.testSize = x_test_fix.shape[0]

        self.prng = np.random.RandomState(seed)

        self.b1 = np.float32(b1)
        self.b2 = np.float32(b2)
        self.learning_rate = np.float32(learning_rate)
        self.lam = np.float32(lam)
        self.dropOutRate1 = np.float32(dropOutRate1)
        self.dropOutRate2 = np.float32(dropOutRate2)
        self.sigmaBias = np.float32(sigmaBias)
        self.ylam = np.float32(yLam)

        self.batch_size = batch_size

        self.filterShapeText = filterShapeText
        self.poolSizeText = poolSizeText
        self.dropOutSizeText = dropOutSizeText
        self.imageShapeText = imageShapeText

        sigma_init = 1

        create_weight = lambda dim_input, dim_output: self.prng.normal(0, sigma_init, (dim_input, dim_output)).astype(
            theano.config.floatX)
        create_bias = lambda dim_output: np.zeros(dim_output).astype(theano.config.floatX)
        create_weight_zeros = lambda dim_input, dim_output: np.zeros([dim_input, dim_output])

        # Fix Effect Convolution Layer 1
        fan_in = np.prod(filterShapeText[0][1:])
        fan_out = (filterShapeText[0][0] * np.prod(filterShapeText[0][2:]) //
                   np.prod(poolSizeText[0]))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W_fh0 = theano.shared(
            np.asarray(
                self.prng.uniform(low=-W_bound, high=W_bound, size=filterShapeText[0]),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        b_fh0 = theano.shared(create_bias((filterShapeText[0][0],)), name='b_th0')

        # Fix Effect Convolution Layer 2
        fan_in = np.prod(filterShapeText[1][1:])
        fan_out = (filterShapeText[1][0] * np.prod(filterShapeText[1][2:]) //
                   np.prod(poolSizeText[1]))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W_fh1 = theano.shared(
            np.asarray(
                self.prng.uniform(low=-W_bound, high=W_bound, size=filterShapeText[1]),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        b_fh1 = theano.shared(create_bias((filterShapeText[1][0],)), name='b_th1')

        W_rh = theano.shared(create_weight(self.groupIDsize, dropOutSizeText[0][0]), name='W_hsigma')
        b_rh = theano.shared(create_bias(dropOutSizeText[0][0]), name='b_hsigma')

        # Combines Features Dropout Layer
        W_ch = theano.shared(create_weight(dropOutSizeText[0][0], dropOutSizeText[0][1]), name='W_th2')
        b_ch = theano.shared(create_bias(dropOutSizeText[0][1]), name='b_th2')

        # Final Logistic Regression Part
        W_yh = theano.shared(create_weight(dropOutSizeText[0][1], n_class), name='W_yh')
        b_yh = theano.shared(create_bias(n_class), name='b_yh')

        self.params = OrderedDict([("W_fh0", W_fh0), ("b_fh0", b_fh0), ("W_fh1", W_fh1), ("b_fh1", b_fh1),
                                   ("W_rh", W_rh), ("b_rh", b_rh),
                                   ("W_ch", W_ch), ("b_ch", b_ch),
                                   ("W_yh", W_yh), ("b_yh", b_yh)])

        self.params_fixed = OrderedDict([
                                   #("W_fh0", W_fh0), ("b_fh0", b_fh0), ("W_fh1", W_fh1), ("b_fh1", b_fh1),
                                   ("W_ch", W_ch), ("b_ch", b_ch),
                                   ("W_yh", W_yh), ("b_yh", b_yh)])

        self.params_random = OrderedDict([("W_rh", W_rh), ("b_rh", b_rh),
                                          #("W_ch", W_ch), ("b_ch", b_ch),
                                          #("W_yh", W_yh), ("b_yh", b_yh)
                                          ])

        # # Adam parameters
        # self.m = OrderedDict()
        # self.v = OrderedDict()
        #
        # for key, value in self.params.items():
        #     self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key)
        #     self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key)

        x_train_fix = theano.shared(x_train_fix.astype(theano.config.floatX), name="x_train_fix")
        y_train = T.cast(y_train, 'int32')
        groupId = theano.shared(groupId.astype(theano.config.floatX), name="groupId")
        x_valid_fix = theano.shared(x_valid_fix.astype(theano.config.floatX), name="x_valid_fix")
        y_valid = T.cast(y_valid, 'int32')
        groupId_valid = theano.shared(groupId_valid.astype(theano.config.floatX), name="groupId_valid")
        x_test_fix = theano.shared(x_test_fix.astype(theano.config.floatX), name="x_test_fix")
        y_test = T.cast(y_test, 'int32')

        self.update, self.test, self.predict, self.update_fixed, self.update_random, self.valid_random, self.update_random_params, self.update_match \
            = self.create_gradientfunctions(x_train_fix, y_train, groupId,x_valid_fix, y_valid, groupId_valid,
                                                                             x_test_fix, y_test)

    def encoder(self, x, b):
        # h_encoder = activation(T.dot(x, self.params['W_xh']) + self.params['b_xh'].dimshuffle('x', 0))
        # # b_encoder = activation(T.dot(b, self.params['W_bh']) + self.params['b_bh'].dimshuffle('x', 0))
        #
        # mu = T.dot(h_encoder, self.params['W_hmu']) + self.params['b_hmu'].dimshuffle('x', 0)
        log_sigma = T.dot(b, self.params['W_rh']) + self.params['b_rh'].dimshuffle('x', 0)

        return x, log_sigma

    def encoder_rand(self, x, b):
        x, log_sigma = self.encoder(x, b)
        if self.dropOutRate2 > 0:
            srng = theano.tensor.shared_randomstreams.RandomStreams(
                self.prng.randint(999999))
            mask = srng.binomial(n=1, p=1 - self.dropOutRate2, size=log_sigma.shape)
            log_sigma = log_sigma * T.cast(mask, theano.config.floatX)
        return x, log_sigma


    def textCNN(self, t):
        layer0_input = t.reshape(self.imageShapeText[0])
        conv_out0 = conv.conv2d(
            input=layer0_input,
            filters=self.params['W_fh0'],
            image_shape=self.imageShapeText[0],
            filter_shape=self.filterShapeText[0]
        )
        pooled_out0 = downsample.max_pool_2d(
            input=conv_out0,
            ds=self.poolSizeText[0],
            ignore_border=True
        )
        output0 = activation_cnn(pooled_out0 + self.params['b_fh0'].dimshuffle('x', 0, 'x', 'x'))
        conv_out1 = conv.conv2d(
            input=output0,
            filters=self.params['W_fh1'],
            image_shape=self.imageShapeText[1],
            filter_shape=self.filterShapeText[1]
        )
        pooled_out1 = downsample.max_pool_2d(
            input=conv_out1,
            ds=self.poolSizeText[1],
            ignore_border=True
        )
        output1 = activation_cnn(pooled_out1 + self.params['b_fh1'].dimshuffle('x', 0, 'x', 'x'))
        return output1.flatten(2)


    def sampler(self, mu, log_sigma):
        if "gpu" in theano.config.device:
            from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams
            srng = CURAND_RandomStreams(seed=seed)
            # srng = T.shared_randomstreams.RandomStreams(seed=seed)
        else:
            srng = T.shared_randomstreams.RandomStreams(seed=seed)

        eps = srng.normal(mu.shape)

        # Reparametrize
        z = mu +  (T.exp(0.5 * log_sigma) - 1) * eps * 5e-1

        return z

    def sampler_rand(self, mu, log_sigma):
        return mu +  (T.exp(0.5 * log_sigma) - 1)

    def dropOutOutput(self, n):
        output2 = activation(T.dot(n, self.params['W_ch']) + self.params['b_ch'])
        if self.dropOutRate1 > 0:
            srng = theano.tensor.shared_randomstreams.RandomStreams(
                self.prng.randint(999999))
            mask = srng.binomial(n=1, p=1 - self.dropOutRate1, size=output2.shape)
            output2 = output2 * T.cast(mask, theano.config.floatX)
        return output2

    def dropOutOutput_rand(self, n):
        return activation(T.dot(n, self.params['W_ch']) + self.params['b_ch'])

    def logisticOutput(self, o, y):
        p_y_given_x = activation_output(T.dot(o, self.params['W_yh']) + self.params['b_yh'].dimshuffle('x', 0))
        y_pred = T.argmax(p_y_given_x, axis=1)
        nll = T.cast(-T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y]), T.config.floatX)
        error = self.errors(y_pred, y)
        return y_pred, nll, error


    def decoder(self, x, z):
        h_decoder = relu(T.dot(z, self.params['W_zh']) + self.params['b_zh'].dimshuffle('x', 0))

        if self.continuous:
            reconstructed_x = T.dot(h_decoder, self.params['W_hxmu']) + self.params['b_hxmu'].dimshuffle('x', 0)
            log_sigma_decoder = T.dot(h_decoder, self.params['W_hxsigma']) + self.params['b_hxsigma']

            logpxz = (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma_decoder) -
                      0.5 * ((x - reconstructed_x) ** 2 / T.exp(log_sigma_decoder))).sum(axis=1, keepdims=True)
        else:
            reconstructed_x = T.nnet.sigmoid(
                T.dot(h_decoder, self.params['W_hx']) + self.params['b_hx'].dimshuffle('x', 0))
            logpxz = - T.nnet.binary_crossentropy(reconstructed_x, x).sum(axis=1, keepdims=True)

        return reconstructed_x, logpxz

    def errors(self, y_pred, y):
        if y.ndim != y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.cast(T.mean(T.neq(y_pred, y)), T.config.floatX)
        else:
            raise NotImplementedError()

    def create_gradientfunctions(self, x_train_fix, y_train, groupId,  x_valid_fix, y_valid, groupId_valid,
                                 x_test_fix, y_test):
        """This function takes as input the whole dataset and creates the entire model"""
        t = T.matrix("t")
        y = T.ivector('y')
        b = T.matrix('b')
        wr = T.matrix('wr')
        br = T.vector('br')
        batch = T.iscalar('batch')

        epoch = T.iscalar("epoch")

        m = self.textCNN(t)

        mu, log_sigma = self.encoder(m, b)

        n = self.sampler(m, log_sigma)

        o = self.dropOutOutput(n)

        # as a classifier
        y_pred, nll, error = self.logisticOutput(o, y)

        # for random part
        mu_rand, log_sigma_rand = self.encoder_rand(m, b)
        n_rand = self.sampler_rand(m, log_sigma_rand)
        o_rand = self.dropOutOutput(n_rand)

        y_pred_rand, nll_rand, error_rand = self.logisticOutput(o_rand, y)

        L1_lower = (T.sum(np.abs(self.params['W_fh0'])) + T.sum(np.abs(self.params['b_fh0'])) + T.mean(
            np.abs(self.params['W_fh1'])) + T.mean(np.abs(self.params['b_fh1']))) * self.ylam

        L1_middle = (T.sum(np.abs(self.params['W_ch'])) + T.sum(np.abs(self.params['b_ch']))) * self.ylam

        L1_rand = (T.sum(np.abs(self.params['W_rh'])) + T.sum(np.abs(self.params['b_rh']))) * 1e-3

        L1_rand_output =  (T.sum(np.abs(n_rand))) * 0 #1e3

        L1_upper = (T.sum(np.abs(self.params['W_yh'])) + T.sum(np.abs(self.params['b_yh']))) * self.ylam

        # cost_random = T.cast(-T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), np.abs(1-y)]), T.config.floatX)
        cost_random = nll_rand + L1_rand + L1_rand_output

        cost = nll + L1_middle + L1_upper

        error_match = T.cast(T.mean(np.abs(m - n_rand)), T.config.floatX)

        cost_match = error_match + L1_rand + L1_rand_output

        # Expectation of (logpz - logqz_x) over logqz_x is equal to KLD (see appendix B):
        # KLD = 0.5 * T.sum(1 + log_sigma - mu**2 - T.exp(log_sigma), axis=1, keepdims=True)

        # Average over batch dimension
        # logpx = T.mean(logpxz + KLD)*0.0001

        updates = opt(cost, self.params, self.learning_rate)

        updates_fixed = opt(cost, self.params_fixed, self.learning_rate)
        updates_random = opt(cost_random, self.params_random, 1)
        updates_match = opt(cost_match, self.params_random, 5e0)

        givens = {
            t: x_train_fix[batch * self.batch_size:(batch + 1) * self.batch_size, :],
            y: y_train[batch * self.batch_size:(batch + 1) * self.batch_size],
            b: groupId[batch * self.batch_size:(batch + 1) * self.batch_size, :],
        }

        groupFakeFixed = theano.shared(np.zeros([self.N, self.groupIDsize]).astype(theano.config.floatX),
                                  name="groupId2")

        givens_fixed = {
            t: x_train_fix[batch * self.batch_size:(batch + 1) * self.batch_size, :],
            y: y_train[batch * self.batch_size:(batch + 1) * self.batch_size],
            b: groupFakeFixed[batch * self.batch_size:(batch + 1) * self.batch_size, :],
        }

        x_trainFakeRandom = theano.shared(np.zeros([self.N, self.features]).astype(theano.config.floatX),
                                  name="xFake")

        givens_random = {
            t: x_trainFakeRandom[batch * self.batch_size:(batch + 1) * self.batch_size, :],
            y: y_train[batch * self.batch_size:(batch + 1) * self.batch_size],
            b: groupId[batch * self.batch_size:(batch + 1) * self.batch_size, :],
        }

        givens_match = {
            t: x_train_fix[batch * self.batch_size:(batch + 1) * self.batch_size, :],
            b: groupId[batch * self.batch_size:(batch + 1) * self.batch_size, :],
        }

        groupFake = theano.shared(np.zeros([self.validSize, self.groupIDsize]).astype(theano.config.floatX),
                                  name="groupId2")

        givens2 = {
            t: x_valid_fix[batch * self.batch_size:(batch + 1) * self.batch_size, :],
            y: y_valid[batch * self.batch_size:(batch + 1) * self.batch_size],
            b: groupFake[batch * self.batch_size:(batch + 1) * self.batch_size, :]
        }

        groupFake2 = theano.shared(np.zeros([self.testSize, self.groupIDsize]).astype(theano.config.floatX),
                                   name="groupId2")

        givens3 = {
            t: x_test_fix[batch * self.batch_size:(batch + 1) * self.batch_size, :],
            y: y_test[batch * self.batch_size:(batch + 1) * self.batch_size],
            b: groupFake2[batch * self.batch_size:(batch + 1) * self.batch_size, :]
        }

        x_validFakeRandom = theano.shared(np.zeros([self.validSize, self.features]).astype(theano.config.floatX),
                                  name="xFake")

        givens_random2 = {
            t: x_validFakeRandom[batch * self.batch_size:(batch + 1) * self.batch_size, :],
            y: y_valid[batch * self.batch_size:(batch + 1) * self.batch_size],
            b: groupId_valid[batch * self.batch_size:(batch + 1) * self.batch_size, :],
        }

        # Define a bunch of functions for convenience
        update = theano.function([batch], cost, updates=updates, givens=givens)
        update_fixed = theano.function([batch], cost, updates=updates_fixed, givens=givens)
        update_random = theano.function([batch], error_rand, updates=updates_random, givens=givens_random)
        update_match = theano.function([batch], cost_match, updates=updates_match, givens=givens_match)
        # likelihood = theano.function([x], logpx)
        # encode = theano.function([x], z)
        # decode = theano.function([z], reconstructed_x)
        test = theano.function([batch], error, givens=givens2)

        predict = theano.function([batch], error, givens=givens3)

        valid_random = theano.function([batch], error_match, givens=givens_random2)

        update_random_params = theano.function(inputs=[wr, br], updates=[(self.params['W_rh'], wr), (self.params['b_rh'], br)])

        return update, test, predict, update_fixed, update_random, valid_random, update_random_params, update_match

    def load_parameters(self, path):
        """Load the variables in a shared variable safe way"""

        params = np.load(path)
        assert len(params) == 8
        self.params['W_fh0'].set_value(params[6].astype(theano.config.floatX))
        self.params['b_fh0'].set_value(params[7].astype(theano.config.floatX))
        self.params['W_fh1'].set_value(params[4].astype(theano.config.floatX))
        self.params['b_fh1'].set_value(params[5].astype(theano.config.floatX))
        self.params['W_ch'].set_value(params[2].astype(theano.config.floatX))
        self.params['b_ch'].set_value(params[3].astype(theano.config.floatX))
        self.params['W_yh'].set_value(params[0].astype(theano.config.floatX))
        self.params['b_yh'].set_value(params[1].astype(theano.config.floatX))

    def get_parameters(self):
        l = []
        l.append(self.params['W_fh0'].get_value(True))
        l.append(self.params['b_fh0'].get_value(True))
        l.append(self.params['W_fh1'].get_value(True))
        l.append(self.params['b_fh1'].get_value(True))
        l.append(self.params['W_ch'].get_value(True))
        l.append(self.params['b_ch'].get_value(True))
        l.append(self.params['W_yh'].get_value(True))
        l.append(self.params['b_yh'].get_value(True))
        return l