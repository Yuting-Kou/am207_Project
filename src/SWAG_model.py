"""
Implement SWA model and return A and W_Swa
"""
from .model import Model


class SWAG:
    def __init__(self, model):
        """
        Implement Algo 2 of stochastic weight averaging (SWA)
        :type model: Model
        """
        self.model = model
        self.A = None # deviation matrix
        self.w_swa = None # final result of w_swa
        pass

    def train_SGD(self, X,y, method="T"):
        """
        train model using SGD, and record SGD deviation matrix, get final w_swa.
        method: "T": every T step to record (default)
        """
        if method == "T":
            ### this function is to update A and w_swa. get SGD updates.
            self.A = None
            self.w_swa = None
        else:
            raise NotImplementedError("method {} not implemented".format(method))

    def get_A(self):
        return self.A

    def get_w_swa(self):
        return self.w_swa