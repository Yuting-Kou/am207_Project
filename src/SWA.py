"""
Implement SWA model and return A and W_Swa
"""
from autograd import numpy as np
from autograd.misc.optimizers import sgd

from model import Model


class SWA:
    def __init__(self, model: Model, X, y, method, lr, T, c, max_rank):
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
        self.A = np.ones(self.model.get_D()).reshape(1, -1)  # deviation matrix
        self.w0 = self.model.weights
        self.w_swa = self.w0
        self.lr = lr
        self.T = T
        self.c = c
        self.max_rank = max_rank
        self.method = method
        _, self.gradient = self.model.make_objective(x_train=X.reshape(1, -1), y_train=y, return_grad=True)
        self.train_SGD()

    def train_SGD(self):
        """
        train model using SGD, and record SGD deviation matrix, get final w_swa.
        """
        if self.method == "T":
            ### this function is to update A and w_swa. get SGD updates.
            self.w = self.w0
            for i in range(self.T):
                self.w = sgd(self.gradient, self.w, step_size=self.lr, num_iters=1, callback=None)
                if i % self.c == 0:
                    n = i / self.c
                    self.w_swa = (n * self.w_swa + self.w) / (n + 1)
                    self.a = self.w_swa - self.w
                    if self.A.shape[0] + 1 > self.max_rank:
                        self.A = self.A[1:, :]
                    self.A = np.concatenate([self.A, self.a], axis=0)

        else:
            raise NotImplementedError("method {} not implemented".format(method))

    def get_A(self):
        return self.A

    def get_w_swa(self):
        return self.w_swa.T  # return size as (n_params, 1)



