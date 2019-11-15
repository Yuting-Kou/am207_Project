from abc import ABC
from autograd import numpy as np


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
        :param n_parameters: # of dimension of low_dim representation
        :param n_subspace: original weight space
        """
        self.n_parameters = n_parameters
        self.subspace = np.random.randn(n_subspace, n_parameters)

    def get_space(self):
        return self.subspace
