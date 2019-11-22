from abc import ABC
from autograd import numpy as np
# from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import Curve_subspace


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

    def collect_vector(self, vector):
        pass

    def get_space(self):
        pass


@Subspace.register_subclass('random')
class RandomSpace(Subspace):
    def __init__(self, n_parameters, n_subspace=20):
        """
        Initialize random subspace method
        :param n_parameters: # of dimension of original weight space
        :param n_subspace: # of dimension of low_dim representation
        """
        self.n_parameters = n_parameters
        self.subspace = np.random.randn(n_subspace, n_parameters)

    def collect_vector(self, vector):
        pass

    def get_space(self):
        return self.subspace

@Subspace.register_subclass('pca')
class PCASpace(Subspace):
    def __init__(self, n_parameters, PCA_rank=20, max_rank = 20):
        """
        Initialize random subspace method
        :param n_parameters: # of dimension of original weight space
        :param PCA_rank # of dimension of low_dim representation (K in paper's algorithm2)
        :param max_rank # of maximum columns in deviation matrix (M in paper's algorithm2)
        """
        super(PCASpace, self).__init__()

        self.n_parameters = n_parameters
        self.PCA_rank = PCA_rank
        self.max_rank = max_rank

        self.rank = 1
        self.deviation_matrix = np.ones(self.n_parameters).reshape(1,-1) # final shape should be (max_rank, n_parameters)
    
    def collect_vector(self, vector):
        if self.rank + 1 > self.max_rank: 
            self.deviation_matrix = self.deviation_matrix[1:, :] # keep the last (max_rank - 1) deviations
        self.deviation_matrix = np.concatenate([self.deviation_matrix, vector.reshape(1,-1)], axis = 0)
        self.rank = min(self.rank + 1, self.max_rank) # update the matrix rank

    def get_space(self):
        deviation_matrix = self.deviation_matrix.copy()

        # deviation_matrix /= (max(1, self.rank)-1)**0.5 
        # pca_rank = max(1, min(self.PCA_rank, self.rank))
        # pca_decomp = TruncatedSVD(n_components=self.PCA_rank)
        # pca_decomp.fit(deviation_matrix) # PCA based on randomized SVD
        
        _, s, Vt = randomized_svd(deviation_matrix, n_components = self.PCA_rank)

        return s[:, None]*Vt
