from abc import ABC

from autograd import grad
from autograd import numpy as np
from autograd.misc.optimizers import adam


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

    def fit(self, x_train, y_train, reg_param=None, params=None):
        """
        train MSE model
        :param params= {'step_size' = 0.01,
                        'max_iteration' = 5000,
                        'check_point' = 100,
                        'optimizer' = 'adam'
                        'random_restarts' = 5
                        }
        """
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
        """
        return original weights from subspace weights
        Output shape: w_hat.T: (-1,D) in order to fit forward need.
        """
        assert z is not None
        assert P is not None
        assert w_hat is not None

        # W = w_hat + P@z
        z = z.reshape(P.shape[-1], -1)
        assert P.shape[0] == w_hat.shape[0]
        return (w_hat + P @ z).T

    def get_z_from_W(self, weights, P, w_hat):
        """
        return z: W=w_hat+P@z
        P.T@(W-w_hat) = P.T@P@z
        """
        pp = P.T @ P
        return np.linalg.pinv(pp) @ P.T @ (weights.reshape(-1, 1) - w_hat.reshape(-1, 1))

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

    def make_objective(self, x_train, y_train, return_grad=True, reg_params=None):
        """
        return Mean Square Error of loss function and its gradient.
        :param x_train: input data X
        :param y_train: input predictor Y
        :param return_grad: whether or not to return gradiant
        :param reg_params: whether to add regularization terms.
        :return: MSE loss of data, if return_grad=True then also return gradient.
        """
        pass

    def update_Sigma_Y(self, Sigma_Y):
        """
        update Sigma_Y: Y~N(nn, Sigma_Y)
        """
        pass


@Model.register_submodel('Feedforward')
class Feedforward(Model):
    def __init__(self, architecture, random=None, weights=None, Sigma_Y=None):
        """
        input data X.shape is (D_in, -1) or (1, D_in, -1)
        :param weights: original shape in (1, D)
        """
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
        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))

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
        self.update_Sigma_Y(Sigma_Y=Sigma_Y)

    def update_Sigma_Y(self, Sigma_Y):
        """update Sigma Y"""
        if Sigma_Y is None:
            Sigma_Y = np.eye(self.params['D_out'])
        else:
            if isinstance(Sigma_Y, int) or isinstance(Sigma_Y, float):
                # if sigma_Z is a number, turn it into (1,1)
                Sigma_Y = np.eye((self.params['D_out'])) * Sigma_Y
            else:
                assert Sigma_Y.shape[0] == Sigma_Y.shape[1] == self.params['D_out']
        self.Sigma_Y_inv = np.linalg.inv(Sigma_Y)
        self.Sigma_Y_det = np.linalg.det(Sigma_Y)

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
        return log likelihood function N(y; mean=likelihood of nn.forward(X, W), var=Sigma_Y)
        :param X: input data X
        :param y: input data y
        :param use_subweights: default is True, which means use subspace weights.
        :param z: subspace weights : (n_subspace,1)
        :param P: projection matrix: (n_parameter, n_subspace)
        :param w_hat: shift vector: (n_parameter,1)
        :param weights: original weights
        :return: N(y; mean=likelihood of nn.forward(X, W), var=Sigma_Y) shape: (nsamples,)
        """
        # W = w_hat + P@z
        if use_subweights:
            y_pred = self.forward(z=z, P=P, w_hat=w_hat, X=X)
        else:
            y_pred = self.forward(use_subweights=False, weights=weights, X=X)  # S, d-out, n_train

        # print('----model.get_likelihood()')
        # print('y_pred.shape',y_pred.shape)
        # print('y_shape',y.shape)
        # print('Xshape',X.shape)
        # print('-----')

        constant = -0.5 * self.params['D_out'] * np.log(2 * np.pi) - np.log(self.Sigma_Y_det)

        # y_pred = y_pred.reshape(-1, self.params['D_out'], y.shape[-1])  # S, d-out, n_train
        if self.params['D_out'] > 1:
            exp_part = np.einsum('abc,acd->abd', np.einsum('efg,fh->egh', y - y_pred, self.Sigma_Y_inv),
                                 y - y_pred)  # (y-y_pred).T@Sigma_Y_inv@(y-y_pred)
            # according to weiwei's code, likelihood is a value. sum of all the results.
            exponential_Y = -0.5 * np.diagonal(exp_part, axis1=-1, axis2=-2).sum(axis=1)
        else:
            exponential_Y = -0.5 * self.Sigma_Y_inv[0, 0] * np.sum((y - y_pred) ** 2, axis=2).ravel()
        assert exponential_Y.shape == (y_pred.shape[0],) # S
        return constant + exponential_Y

    def make_objective(self, x_train, y_train, return_grad=True, reg_param=None):
        """
        return Mean Square Error of loss function and its gradient. Over original parameter space.
        :param x_train: input data X
        :param y_train: input predictor Y
        :param return_grad: whether or not to return gradiant
        :param reg_param: regularization coefficients
        :return: MSE loss of data, if return_grad=True then also return gradient.
        """

        def objective(W, t):
            squared_error = np.linalg.norm(y_train - self.forward(X=x_train, use_subweights=False, weights=W),
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

    def fit(self, x_train, y_train, reg_param=None, params=None):
        """
        train MSE model.
        :param x_train:
        :param y_train:
        :param reg_param:
        :param params:
        :return:
        """

        assert x_train.shape[0] == self.params['D_in']
        assert y_train.shape[0] == self.params['D_out']

        ### make objective function for training
        self.objective, self.gradient = self.make_objective(x_train=x_train, y_train=y_train, reg_param=reg_param)

        ### set up optimization
        step_size = 0.01
        max_iteration = 5000
        check_point = 100
        weights_init = self.weights.reshape((1, -1))
        optimizer = 'adam'
        random_restarts = 5

        if 'step_size' in params.keys():
            step_size = params['step_size']
        if 'max_iteration' in params.keys():
            max_iteration = params['max_iteration']
        if 'check_point' in params.keys():
            self.check_point = params['check_point']
        if 'init' in params.keys():
            weights_init = params['init']
        if 'optimizer' in params.keys():
            optimizer = params['optimizer']
        if 'random_restarts' in params.keys():
            random_restarts = params['random_restarts']
        if 'check_point' in params.keys():
            check_point = params['check_point']

        def call_back(weights, iteration, g):
            ''' Actions per optimization step '''
            objective = self.objective(weights, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.weight_trace = np.vstack((self.weight_trace, weights))
            if check_point != 0 and iteration % check_point == 0:
                print("Iteration {} lower bound {}; gradient mag: {}".format(iteration, objective, np.linalg.norm(
                    self.gradient(weights, iteration))))

        ### train with random restarts
        optimal_obj = 1e16

        for i in range(random_restarts):
            if optimizer == 'adam':
                adam(self.gradient, weights_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
            local_opt = np.min(self.objective_trace[-100:])
            if local_opt < optimal_obj:
                opt_index = np.argmin(self.objective_trace[-100:])
                self.weights = self.weight_trace[-100:][opt_index].reshape((1, -1))
            weights_init = self.random.normal(0, 1, size=(1, self.D))

        self.objective_trace = self.objective_trace[1:]
        self.weight_trace = self.weight_trace[1:]
