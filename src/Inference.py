from autograd import numpy as np
from autograd import grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam, sgd
from abc import ABC

class Inference(ABC):
    """ An inference instance which has inference method and get_posterior methods."""
    def __init__(self,params):
        self.params = params

    def get_posterior(self):
        pass

class HMC(Inference):
    """Implement HMC using hw7 solutions"""
    def __init__(self, potential_energy, kinetic_energy, kinetic_energy_distribution, random=None, diagnostic_mode=False):
        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)
        self.D = np.max(kinetic_energy_distribution(1).shape)
        self.potential_energy = potential_energy
        self.kinetic_energy = kinetic_energy
        self.total_energy = lambda position, momentum: potential_energy(position) + kinetic_energy(momentum)

        self.sample_momentum = lambda n: kinetic_energy_distribution(n).reshape((1, self.D))
        self.grad_potential_energy = grad(potential_energy)

        self.params = {'step_size': 0.1,
                       'leapfrog_steps': 10,
                       'total_samples': 1000,
                       'burn_in': 0.1,
                       'thinning_factor': 1,
                       'diagnostic_mode': diagnostic_mode}

        self.accepts = 0.
        self.iterations = 0.
        self.trace = np.empty((1, self.D))
        self.potential_energy_trace = np.empty((1,))

        assert self.sample_momentum(1).shape == (1, self.D)
        assert isinstance(self.potential_energy(self.sample_momentum(1)), float)
        assert isinstance(self.kinetic_energy(self.sample_momentum(1)), float)
        assert isinstance(self.total_energy(self.sample_momentum(1), self.sample_momentum(1)), float)
        assert self.grad_potential_energy(self.sample_momentum(1)).shape == (1, self.D)