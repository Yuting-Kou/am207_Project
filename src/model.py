from abc import ABC
from autograd import numpy as np


class Model(ABC):
    modeltype = {}

    @classmethod
    def register_submodel(cls, submodel):
        def decorator(subclass):
            cls.modeltype[submodel] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, submodel_type, **kwargs):
        if submodel_type not in cls.modeltype:
            raise ValueError('Not implemented subspaces type: {}'.format(submodel_type))
        return cls.modeltype[submodel_type](**kwargs)

    def __init__(self):
        """construct neural network"""
        self.weights = None
        self.D = None
        pass

    def get_likelihood(self, X, z, P, w_hat):
        """
        reconstruct original weights W = w_hat + P@z
        return likelihood function feed forward.
        :param X: input data X
        :param z: subspace weights : (n_subspace,1)
        :param P: projection matrix: (n_parameter, n_subspace)
        :param w_hat: shift vector: (n_parameter,1)
        :return: likelihood of nn.forward(X, W)
        """
        pass

    def turn_weights(self):
        """
        get weights vector from pytorch neural network
        :return: a mutable vector of weights
        """
        pass

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights):
        """
        update weights of current neural networks (put it into dictionary).
        :param new_weights: a mutable vector
        """
        pass

    def get_D(self):
        """return # of dimension of original weight space. = n_parameters"""
        return self.D


@Model.register_submodel('Feedforward')
class Feedforward(Model):
    def __init__(self, architecture, random=None, weights=None):
        self.params = {'H': architecture['width'],
                       'L': architecture['hidden_layers'],
                       'D_in': architecture['input_dim'],
                       'D_out': architecture['output_dim'],
                       'activation_type': architecture['activation_fn_type'],
                       'activation_params': architecture['activation_fn_params']}

        self.D = ((architecture['input_dim'] * architecture['width'] + architecture['width'])
                  + (architecture['output_dim'] * architecture['width'] + architecture['output_dim'])
                  + (architecture['hidden_layers'] - 1) * (architecture['width'] ** 2 + architecture['width']))

        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.h = architecture['activation_fn']

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            self.weights = weights

    def set_weights(self, new_weights):
        assert new_weights.shape == self.weights.shape
        self.weights = new_weights

    def forward(self, weights, x):
        ''' Forward pass given weights and input '''
        H = self.params['H']
        D_in = self.params['D_in']
        D_out = self.params['D_out']

        assert weights.shape[1] == self.D

        if len(x.shape) == 2:
            assert x.shape[0] == D_in
            x = x.reshape((1, D_in, -1))
        else:
            assert x.shape[1] == D_in

        weights = weights.T

        # input to first hidden layer
        W = weights[:H * D_in].T.reshape((-1, H, D_in))
        b = weights[H * D_in:H * D_in + H].T.reshape((-1, H, 1))
        input = self.h(np.matmul(W, x) + b)
        index = H * D_in + H

        assert input.shape[1] == H

        # additional hidden layers
        for _ in range(self.params['L'] - 1):
            before = index
            W = weights[index:index + H * H].T.reshape((-1, H, H))
            index += H * H
            b = weights[index:index + H].T.reshape((-1, H, 1))
            index += H
            output = np.matmul(W, input) + b
            input = self.h(output)

            assert input.shape[1] == H

        # output layer
        W = weights[index:index + H * D_out].T.reshape((-1, D_out, H))
        b = weights[index + H * D_out:].T.reshape((-1, D_out, 1))
        output = np.matmul(W, input) + b
        assert output.shape[1] == self.params['D_out']

        return output

    def get_likelihood(self, X, z, P, w_hat):
        """
        W = w_hat + P@z
        :param X: input data
        :param z: subspace weights : (n_subspace,1)
        :param P: projection matrix: (n_parameter, n_subspace)
        :param w_hat: shift vector: (n_parameter,1)
        :return:
        """
        # W = w_hat + P@z
        assert z.shape[0] == P.shape[-1]
        assert (P.shape[0], z.shape[1]) == w_hat.shape
        weights = w_hat + P@z
        print(X.shape)
        return self.forward(weights=weights.reshape(-1, P.shape[0]), x=X)
