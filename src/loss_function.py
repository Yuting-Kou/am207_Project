from model import Model
from autograd import grad
from autograd import numpy as np
def make_objective(model:Model, x_train, y_train):
    def objective(W, t):
        squared_error = np.linalg.norm(y_train - model.forward(W, x_train), axis=1)**2
        sum_error = np.sum(squared_error)
        return sum_error
    return objective, grad(objective)