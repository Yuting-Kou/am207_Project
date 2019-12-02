from abc import ABC

from autograd import numpy as np
from sklearn.utils.extmath import randomized_svd

from SWA import SWA
from model import Model


class Subspace(ABC):
    subclasses = {}

    @classmethod
    def register_subclass(cls, subspace_type):
        def decorator(subclass):
            cls.subclasses[subspace_type] = subclass
            return subclass

        return decorator

    def __init__(self):
        pass

    @classmethod
    def create(cls, subspace_type, **kwargs):
        if subspace_type not in cls.subclasses:
            raise ValueError('Not implemented subspaces type: {}'.format(subspace_type))
        return cls.subclasses[subspace_type](**kwargs)

    def collect_vector(self, X, y):
        """Each method to construct their own vector based on data(X,y)"""
        pass

    def get_space(self):
        """return transformed matrix P and shift vector w_hat
            - P matrix: shape: (n_large, n_small)
            - shift vector: (n_large, 1)
            - # W = w_hat + P@z
        """
        pass


@Subspace.register_subclass('random')
class RandomSpace(Subspace):
    def __init__(self, model:Model, n_subspace=20):
        """
        Initialize random subspace method
        :param n_subspace: # of dimension of low_dim representation (small)
        :type model: Model
        :param model: model
        """
        self.n_parameters = model.get_D()
        self.subspace = np.random.randn(self.n_parameters, n_subspace)
        self.model = model
        self.w_hat = np.zeros((self.n_parameters, 1))
        
    def collect_vector(self, X, y, method="T", lr = 0.02, T = 2000, c = 10, max_rank = 20):
        """set shift vector as SWA results
        params info see SWA.py"""
        self.myswag = SWA(self.model, X, y, method, lr, T, c, max_rank)
        self.w_hat = self.myswag.get_w_swa()

    def get_space(self):
        """
        :return:
            subspace: projection matrix with shape [p_small, p_large]
            w_hat: shift vectors = w_{SWA}
        """
        return self.subspace, self.w_hat


@Subspace.register_subclass('pca')
class PCASpace(Subspace):
    def __init__(self, model:Model, n_subspace=20):
        """
        Initialize random subspace method
        :param model: # of dimension of original weight space
        :type model: Model
        :param n_subspace: # of dimension of low_dim representation (small)
        """
        super(PCASpace, self).__init__()

        self.n_parameters = model.get_D()
        self.n_subspace = n_subspace
        self.model = model
        self.w_hat = np.zeros((self.n_parameters, 1))


    def collect_vector(self, X, y, method="T", lr = 0.02, T = 2000, c = 10, max_rank = 20):
        """set shift vector as SWA results
        params info see SWA.py"""
        self.myswag = SWA(self.model, X, y, method, lr, T, c, max_rank)
        self.w_hat = self.myswag.get_w_swa()
 

    def get_space(self):
        self.A = self.myswag.get_A()
        _, s, Vt = randomized_svd(self.A, n_components=self.n_subspace)
        self.subspace = (np.diag(s)@Vt).T

        return self.subspace, self.w_hat


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
    my_subspace = Subspace.create(subspace_type="pca", model=my_nn, n_subspace=2)

    my_subspace.collect_vector(X=x, y=y)
    P, w = my_subspace.get_space()
    print(P.shape, w.shape)


    z = np.random.randn(2,1)
    print(my_nn.forward(X=x.reshape(1,-1), z=z, P=P, w_hat=w).shape)


