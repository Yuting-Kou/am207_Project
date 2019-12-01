from abc import ABC

from autograd import grad
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

    def forward(self, X, use_subweights=True, z=None, P=None, w_hat=None, weights=None):
        """
        return predict Y, use either original weights W or subweights z.
        :param X: input data X
        :param use_subweights: default is True, which means use subspace weights.
        :param z: subspace weights : (n_subspace,1)
        :param P: projection matrix: (n_parameter, n_subspace)
        :param w_hat: shift vector: (n_parameter,1)
        :param weights: original weights
        :return: nn.forward(X, W)
        """
        pass

    def get_likelihood(self, X, y, use_subweights=True, z=None, P=None, w_hat=None, weights=None):
        """
        return likelihood of data y in function feed forward.
        :param X: input data X
        :param y: input predictor y
        :param use_subweights: default is True, which means use subspace weights.
        :param z: subspace weights : (n_subspace,1)
        :param P: projection matrix: (n_parameter, n_subspace)
        :param w_hat: shift vector: (n_parameter,1)
        :param weights: original weights
        :return: likelihood of nn.forward(X, W)
        """
        pass

    def get_W_from_z(self, z, P, w_hat):
        """ return original weights from subspace weights """
        assert z is not None
        assert P is not None
        assert w_hat is not None

        # W = w_hat + P@z
        assert z.shape[0] == P.shape[-1]
        assert (P.shape[0], z.shape[1]) == w_hat.shape
        return w_hat + P @ z

    def vectorize_weights(self):
        """
        get weights vector from pytorch neural network
        :return: a mutable vector of weights
        """
        pass

    def get_weights(self):
        """return original weights W"""
        return self.weights

    def set_weights(self, new_weights):
        """
        update weights of current neural networks (put it into dictionary).
        :param new_weights: a mutable vector. Original weights W
        """
        pass

    def get_D(self):
        """return # of dimension of original weight space. = n_parameters"""
        return self.D

    def make_objective(self, x_train, y_train, weights, return_grad=True, reg_params=None):
        """
        return Mean Square Error of loss function and its gradient.
        :param x_train: input data X
        :param y_train: input predictor Y
        :param weights: network parameters
        :param return_grad: whether or not to return gradiant
        :param reg_params: whether to add regularization terms.
        :return: MSE loss of data, if return_grad=True then also return gradient.
        """
        pass


@Model.register_submodel('Feedforward')
class Feedforward(Model):
    def __init__(self, architecture, random=None, weights=None, y_var=None):
        self.params = {'H': architecture['width'],
                       'L': architecture['hidden_layers'],
                       'D_in': architecture['input_dim'],
                       'D_out': architecture['output_dim'],
                       'activation_type': architecture['activation_fn_type'],
                       'activation_params': architecture['activation_fn_params']}
        # D is total dimension of original weights
        self.D = ((architecture['input_dim'] * architecture['width'] + architecture['width'])
                  + (architecture['output_dim'] * architecture['width'] + architecture['output_dim'])
                  + (architecture['hidden_layers'] - 1) * (architecture['width'] ** 2 + architecture['width']))

        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.h = architecture['activation_fn']

        # weights is original weights W
        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            self.weights = weights

        # y variance is a matrix
        if y_var is None:
            self.y_var = np.eye(self.params['D_out'])
        else:
            if len(y_var.shape) > 1:
                assert y_var.shape[0] == y_var.shape[1] == self.params['D_out']
            else:
                y_var = np.copy(y_var).reshape(1, 1)
            self.y_var = y_var
        self.Sigma_Y_inv = np.linalg.inv(self.y_var)
        self.Sigma_Y_det = np.linalg.det(self.y_var)

    def set_weights(self, new_weights):
        """ change original weights W"""
        assert new_weights.shape == self.weights.shape
        self.weights = new_weights

    def forward(self, X, use_subweights=True, z=None, P=None, w_hat=None, weights=None):
        '''
        Forward pass given weights and input
        :param X: input data X.shape is (D_in, -1) or (1, D_in, -1)
        :param weights: weights.shape is (-1, D)
        :return predicted Y
        '''
        if use_subweights:
            weights = self.get_W_from_z(z=z, P=P, w_hat=w_hat)
            weights = weights.reshape(-1, self.D)

        H = self.params['H']
        D_in = self.params['D_in']
        D_out = self.params['D_out']

        assert weights.shape[1] == self.D

        if len(X.shape) == 2:
            assert X.shape[0] == D_in
            X = X.reshape((1, D_in, -1))
        else:
            assert X.shape[1] == D_in

        weights = weights.T

        # input to first hidden layer
        W = weights[:H * D_in].T.reshape((-1, H, D_in))
        b = weights[H * D_in:H * D_in + H].T.reshape((-1, H, 1))
        input = self.h(np.matmul(W, X) + b)
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

    def get_likelihood(self, X, y, use_subweights=True, z=None, P=None, w_hat=None, weights=None):
        """
        return likelihood function N(y; mean=likelihood of nn.forward(X, W), var=y_var)
        :param X: input data X
        :param y: input data y
        :param use_subweights: default is True, which means use subspace weights.
        :param z: subspace weights : (n_subspace,1)
        :param P: projection matrix: (n_parameter, n_subspace)
        :param w_hat: shift vector: (n_parameter,1)
        :param weights: original weights
        :return: N(y; mean=likelihood of nn.forward(X, W), var=y_var)
        """
        # W = w_hat + P@z
        if use_subweights:
            weights = self.get_W_from_z(z=z, P=P, w_hat=w_hat)
        y = y.reshape(-1, self.params['D_out'])
        y_pred = self.forward(weights=weights.reshape(-1, self.D), x=X)
        # y_pred shape -1, D_out
        constant = -0.5 * (self.params['D_out'] * np.log(2 * np.pi)) + np.log(self.Sigma_Y_det)
        exponential_Y = -0.5 * np.diag(np.dot(np.dot((y - y_pred, self.Sigma_Y_inv), (y - y_pred).T)))
        ##### what is the mean? y_pred is mean.
        return constant + exponential_Y

    def make_objective(self, x_train, y_train, weights, return_grad=True, reg_param=None):
        """
        return Mean Square Error of loss function and its gradient. Over original parameter space.
        :param x_train: input data X
        :param y_train: input predictor Y
        :param weights: network parameters
        :param return_grad: whether or not to return gradiant
        :param reg_param: regularization coefficients
        :return: MSE loss of data, if return_grad=True then also return gradient.
        """

        def objective(W, t):
            squared_error = np.linalg.norm(y_train - self.forward(X=x_train, use_subweights=True, weights=W),
                                           axis=1) ** 2
            if reg_param is None:
                sum_error = np.sum(squared_error)
                return sum_error
            else:
                mean_error = np.mean(squared_error) + reg_param * np.linalg.norm(W)
                return mean_error

        if return_grad:
            return objective, grad(objective)
        else:
            return objective
