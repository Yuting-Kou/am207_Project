"""
Implement SWA model and return A and W_Swa
"""
from model import Model
from loss_function import make_objective
from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import sgd


class SWA:
    def __init__(self, model:Model, X, y, method, lr, T, c, max_rank):
        """
        Implement Algo 2 of stochastic weight averaging (SWA)
        :type model: Model
        :method: "T": every T step to record (default)
        :param lr: learning rate of SGD
        :param T: # of SGD steps
        :param c: moment update frequency 
        :param max_rank: # of maximum columns in deviation matrix (M in paper's algorithm2)
        """
        self.model = model
        self.A = np.ones(self.model.get_D()).reshape(1, -1) # deviation matrix
        self.w0 = self.model.weights 
        self.w_swa = self.w0
        self.lr = lr
        self.T = T
        self.c = c
        self.max_rank = max_rank
        self.method = method
        _, self.gradient = make_objective(self.model, X.reshape(1,-1), y)
        self.train_SGD()

        
    def train_SGD(self):
        """
        train model using SGD, and record SGD deviation matrix, get final w_swa.
        """
        if self.method == "T":
            ### this function is to update A and w_swa. get SGD updates.
            self.w = self.w0
            for i in range(self.T):
                self.w = sgd(self.gradient, self.w0, step_size=self.lr, num_iters=1, callback=None)
                if i % self.c == 0:
                    n = i/self.c
                    self.w_swa = (n * self.w_swa + self.w) / (n + 1)
                    self.a = self.w_swa - self.w
                    if self.A.shape[0] + 1 > self.max_rank:
                        self.A = self.A[1:, :]
                    self.A = np.concatenate([self.A, self.a], axis = 0)
            
        else:
            raise NotImplementedError("method {} not implemented".format(method))

    def get_A(self):
        return self.A

    def get_w_swa(self):
        return self.w_swa.T # return size as (n_params, 1)

if __name__ == '__main__':
    # test
    print("All possible models:{}".format(Model.modeltype))
    data = np.load(r'..\example\data.npy')
    # data = np.load('data.npy')
    x, y = data[:, 0], data[:, 1]

    alpha = 1
    c = 0
    h = lambda x: np.exp(-alpha * (x - c) ** 2)

    ###neural network model design choices
    width = 5
    hidden_layers = 1
    input_dim = 1
    output_dim = 1

    architecture = {'width': width,
                    'hidden_layers': hidden_layers,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation_fn_type': 'rbf',
                    'activation_fn_params': 'c=0, alpha=1',
                    'activation_fn': h}

    my_nn = Model.create(submodel_type="Feedforward", architecture=architecture)
    swa = SWA(my_nn, x, y)
    print(swa.get_A().shape)
    print(swa.get_w_swa().shape)