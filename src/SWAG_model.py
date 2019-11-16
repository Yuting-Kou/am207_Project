"""
All three subspaces involves with SWAG method. Then we can construct a SWAG model to call subspace methods.
"""
from .Subspace import Subspace
from .Inference import Inference


class SWAG:
    def __init__(self, model, subspace_type, subspace_kwargs=None, *args, **kwargs):
        """

        :param model: Bayesian Neural Network to be inferenced
        :param subspace_type: 3 subspaces {'random', 'PCA', 'curve'}
        :param subspace_kwargs: key words for subspaces
        """
        self.model = model
        self.num_parameter = self.model.count_params()  # keras model
        subspace_kwargs = dict() if subspace_kwargs is None else subspace_kwargs
        self.subspace = Subspace.create(subspace_type, n_parameters=self.num_parameter, **subspace_kwargs)
