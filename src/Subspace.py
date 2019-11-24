from abc import ABC

from autograd import numpy as np
# from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

from .SWAG_model import SWAG
from .model import Model


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
        self.n_parameters = len(model.get_weights())
        self.subspace = np.random.randn(self.n_parameters, n_subspace)
        self.model = model
        self.w_hat = np.zeros((self.n_parameters, 1))

    def collect_vector(self, X, y, method="T"):
        """set shift vector as SWA results"""
        myswag = SWAG(model=self.model)
        myswag.train_SGD(X=X, y=y, method=method)
        self.w_hat = myswag.get_w_swa()

    def get_space(self):
        """
        :return:
            subspace: projection matrix with shape [p_small, p_large]
            w_hat: shift vectors = w_{SWA}
        """
        return self.subspace, self.w_hat


@Subspace.register_subclass('pca')
class PCASpace(Subspace):
    def __init__(self, model, PCA_rank=20, max_rank=20):
        """
        Initialize random subspace method
        :param model: # of dimension of original weight space
        :type model: Model
        :param PCA_rank # of dimension of low_dim representation (K in paper's algorithm2)
        :param max_rank # of maximum columns in deviation matrix (M in paper's algorithm2)
        """
        super(PCASpace, self).__init__()

        self.n_parameters = len(model.get_weights())
        self.PCA_rank = PCA_rank
        self.max_rank = max_rank

        self.rank = 1
        # final shape should be (max_rank, n_parameters)
        self.deviation_matrix = np.ones(self.n_parameters).reshape(1, -1)

    def collect_vector(self, vector):
        if self.rank + 1 > self.max_rank:
            self.deviation_matrix = self.deviation_matrix[1:, :]  # keep the last (max_rank - 1) deviations
        self.deviation_matrix = np.concatenate([self.deviation_matrix, vector.reshape(1, -1)], axis=0)
        self.rank = min(self.rank + 1, self.max_rank)  # update the matrix rank

    def get_space(self):
        deviation_matrix = self.deviation_matrix.copy()

        # deviation_matrix /= (max(1, self.rank)-1)**0.5
        # pca_rank = max(1, min(self.PCA_rank, self.rank))
        # pca_decomp = TruncatedSVD(n_components=self.PCA_rank)
        # pca_decomp.fit(deviation_matrix) # PCA based on randomized SVD

        _, s, Vt = randomized_svd(deviation_matrix, n_components=self.PCA_rank)

        return s[:, None] * Vt
