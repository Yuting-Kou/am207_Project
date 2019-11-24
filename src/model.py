from abc import ABC


class Model(ABC):
    modeltype = {}

    @classmethod
    def register_submodel(cls, submodel):
        def decorator(subclass):
            cls.modeltype[submodel] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, submodel_type, **kwargs):
        if submodel_type not in cls.modeltype:
            raise ValueError('Not implemented subspaces type: {}'.format(submodel_type))
        return cls.modeltype[submodel_type](**kwargs)

    def __init__(self):
        """construct neural network"""
        pass

    def get_likelihood(self, X, z, P, w_hat):
        """
        reconstruct original weights from subspace weight z based on P and shift vector w_hat
        return likelihood function feed forward.
        :param X: input data X
        :param z: subspace weights: z = P@W+ w_hat
        :param P: projection matrix
        :param w_hat: shift vectors
        :return: likelihood of nn.forward(X, W)
        """
        pass

    def get_weights(self):
        """
        get weights vector from pytorch neural network
        :return: a mutable vector of weights
        """
    def set_weights(self, new_weights):
        """
        update weights of current neural networks (put it into dictionary).
        :param new_weights: a mutable vector
        """
        pass
