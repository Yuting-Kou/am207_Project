from abc import ABC

from autograd import numpy as np
from sklearn.utils.extmath import randomized_svd

from SWA import SWA


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
    def __init__(self, model, n_subspace=20):
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
        
    def collect_vector(self, X, y, method="T", lr = 0.02, T = 2000, c = 10, max_rank = 10):
        """set shift vector as SWA results
        params info see SWA.py"""
        self.myswag = SWA(self.model, X, y, method, lr, T, c, max_rank)
        self.w_hat = self.myswag.get_w_swa()

    def get_space(self):
        """
        :return:
            subspace: projection matrix with shape [p_large, p_small]
            w_hat: shift vectors = w_{SWA}
        """
        return self.subspace, self.w_hat


@Subspace.register_subclass('pca')
class PCASpace(Subspace):
    def __init__(self, model, n_subspace=20):
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


    def collect_vector(self, X, y, method="T", lr = 0.02, T = 2000, c = 10, max_rank = 10):
        """set shift vector as SWA results
        params info see SWA.py"""
        self.myswag = SWA(self.model, X, y, method, lr, T, c, max_rank)
        self.w_hat = self.myswag.get_w_swa()
 

    def get_space(self):
        self.A = self.myswag.get_A()
        _, s, Vt = randomized_svd(self.A, n_components=self.n_subspace)
        self.subspace = (np.diag(s)@Vt).T

        return self.subspace, self.w_hat

