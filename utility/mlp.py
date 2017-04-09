"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None,
                 activation=None):
        self.activation = activation
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if self.activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            W = theano.shared(W)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            b = theano.shared(b)

        self.W = W
        self.b = b
        self.rng = rng

        self.params = [self.W, self.b]

    def output(self, input):
        lin_output = T.dot(input, self.W) + self.b
        return (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, n_in, n_out,
                 activation, dropout_rate, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)
        self.dropout_rate = dropout_rate
        # if dropout_rate > 0:
        #     self.output = self.dropout_from_layer(rng, self.output, p=dropout_rate)

    def dropout_from_layer(self, layer):
        """p is the probablity of dropping a unit
        """
        srng = theano.tensor.shared_randomstreams.RandomStreams(
                self.rng.randint(999999))
        # p=1-p because 1's indicate keep and p is prob of dropping
        mask = srng.binomial(n=1, p=1-self.dropout_rate, size=layer.shape)
        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        output = layer * T.cast(mask, theano.config.floatX)
        return output

    def output(self, input):
        lin_output = T.dot(input, self.W) + self.b

        r = self.activation(lin_output)

        if self.dropout_rate>0:
            return self.dropout_from_layer(r)
        else:
            return r

