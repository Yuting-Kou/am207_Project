from autograd import numpy as np
from autograd import grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
from abc import ABC


class Inference(ABC):
    """ An inference instance which has inference method and get_posterior methods."""

    def __init__(self, D, Sigma_W):
        """
        - D: dimension of weights
        - Sigma_W: prior variance of weights
        """
        self.D = D
        self.Sigma_W = Sigma_W
        self.Sigma_W_inv = np.linalg.inv(self.Sigma_W)
        self.Sigma_W_det = np.linalg.det(self.Sigma_W)

    def get_posterior(self, n_samples):
        pass


def log_prior(W, Sigma_W_det, Sigma_W_inv, D):
    """
    define log prior of W. Assume W ~ N(0, Sigma_W)
    ===
    W: weights is in [-1, D] shapes
    Sigma_W_det: determinant of Sigma_W
    Sigma_W_inv: inverse matrix of Sigma_W
    ===
    """
    assert len(W.shape) == 2 and W.shape[1] == D
    S = len(W)
    constant_W = -0.5 * (D * np.log(2 * np.pi) + np.log(Sigma_W_det))
    exponential_W = -0.5 * np.diag(np.dot(np.dot(W, Sigma_W_inv), W.T))
    assert exponential_W.shape == (S,)
    log_p_W = constant_W + exponential_W
    return log_p_W


class BBB(Inference):
    """
    Implement BBVI using https://github.com/HIPS/autograd/blob/master/examples/black_box_svi.py

    === Usage ===

    # instantiate a BBB sampler
    BBB_sampler = BBB(log_lklhd, D, Sigma_W, random=np.random.RandomState(0))

    #sample from the bayesian neural network posterior
    BBB_sampler.variational_inference(S=None, max_iteration=None, step_size=None,
                              verbose=None, init_mean=None, init_log_std=None)
    === ===

    """

    def __init__(self, log_lklhd, D, Sigma_W, random=None, tune_params=None):
        """
        log_lklhd: is log likelihood function of target distribution
        D: dimension of weights
        Sigma_W: prior variance of weights

        (default) tuning_params: a disk of following info:
            {'step_size': 0.1,
            'S':2000,
            'max_iteration':20000,
           'verbose': True,
           'init_mean': None,
           'init_log_std': None}
        """
        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.D = D
        self.Sigma_W = Sigma_W
        self.Sigma_W_inv = np.linalg.inv(self.Sigma_W)
        self.Sigma_W_det = np.linalg.det(self.Sigma_W)
        self.log_lklhd = log_lklhd
        self.log_prior = lambda W: log_prior(W, Sigma_W_det=self.Sigma_W_det, Sigma_W_inv=self.Sigma_W_inv, D=self.D)
        self.log_density = lambda W, t: log_lklhd(W) + self.log_prior(W)

        if tune_params is None:
            self.tune_params = {'step_size': 0.1,
                                'S': 2000,
                                'max_iteration': 20000,
                                'verbose': True,
                                'init_mean': None,
                                'init_log_std': None}
        else:
            self.tune_params = tune_params

    def variational_inference(self, S=None, max_iteration=None, step_size=None,
                              verbose=None, init_mean=None, init_log_std=None):
        '''implements wrapper for variational inference via bbb for bayesian regression'''

        # Set model parameters
        if step_size is not None:
            self.tune_params['step_size'] = step_size
        if S is not None:
            self.tune_params['S'] = S
        if max_iteration is not None:
            self.tune_params['max_iteration'] = max_iteration
        if verbose is not None:
            self.tune_params['verbose'] = verbose
        if init_mean is not None:
            self.tune_params['init_mean'] = init_mean
        if init_log_std is not None:
            self.tune_params['init_log_std'] = init_log_std


        # build variational objective.
        objective, gradient, unpack_params = self.black_box_variational_inference()

        def callback(params, t, g):
            if self.tune_params['verbose']:
                if t % 100 == 0:
                    print("Iteration {} lower bound {}; gradient mag: {}".format(t, -objective(params, t),
                                                                                 np.linalg.norm(gradient(params, t))))

        # initialize variational parameters
        if self.tune_params['init_mean'] is None:
            init_mean = np.ones(self.D)
        if self.tune_params['init_log_std'] is None:
            init_log_std = -100 * np.ones(self.D)
        init_var_params = np.concatenate([init_mean, init_log_std])

        # perform gradient descent using adam (a type of gradient-based optimizer)
        self.variational_params = adam(gradient, init_var_params, step_size=self.tune_params['step_size'],
                                       num_iters=self.tune_params['max_iteration'], callback=callback)

    def black_box_variational_inference(self):
        """Implements http://arxiv.org/abs/1401.0118, and uses the
        local reparameterization trick from http://arxiv.org/abs/1506.02557"""

        def unpack_params(params):
            # Variational dist is a diagonal Gaussian.
            mean, log_std = params[:self.D], params[self.D:]
            return mean, log_std

        def gaussian_entropy(log_std):
            return 0.5 * self.D * (1.0 + np.log(2 * np.pi)) + np.sum(log_std)

        rs = npr.RandomState(0)

        def variational_objective(params, t):
            """Provides a stochastic estimate of the variational lower bound."""
            mean, log_std = unpack_params(params)
            samples = rs.randn(self.tune_params['S'], self.D) * np.exp(log_std) + mean
            lower_bound = gaussian_entropy(log_std) + np.mean(self.log_density(samples, t))
            return -lower_bound

        gradient = grad(variational_objective)

        return variational_objective, gradient, unpack_params

    def get_posterior(self, n_samples):
        var_means = self.variational_params[:self.D]
        var_variance = np.diag(np.exp(self.variational_params[self.D:]) ** 2)
        posterior_samples = np.random.multivariate_normal(var_means, var_variance, size=n_samples)
        return posterior_samples


class HMC(Inference):
    """
    Implement HMC using hw7 solutions.

    === Usage ===

    # instantiate an HMC sampler
    HMC_sampler = HMC(log_lklhd, D, Sigma_W, random=np.random.RandomState(0))

    #sample from the bayesian neural network posterior
    HMC_sampler.sample(position_init=position_init,
                       step_size=step_size,
                       leapfrog_steps=leapfrog_steps,
                       total_samples=total_samples,
                       burn_in=burn_in,
                       thinning_factor=thinning_factor)
    === ===
    """

    def __init__(self, log_lklhd, D, Sigma_W, random=None, diagnostic_mode=False, tune_params=None):
        """
        (default) potential_energy = -(log_lklhd + log_prior)
        (default) kinetic_energy = lambda W: np.sum(W**2) / 2.0
        (default) kinetic_energy_distribution = lambda D: random.normal(0, 1, size=D)

        params is a dictionary includes at least following information:
            - D: dimension of weights
            - Sigma_W: prior variance of weights

        (default) tuning_params: a disk of following info:
            {'step_size': 0.1,
           'leapfrog_steps': 10,
           'total_samples': 1000,
           'burn_in': 0.1,
           'thinning_factor': 1,
           'diagnostic_mode': diagnostic_mode}
        """
        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.D = D
        self.Sigma_W = Sigma_W
        self.Sigma_W_inv = np.linalg.inv(self.Sigma_W)
        self.Sigma_W_det = np.linalg.det(self.Sigma_W)
        self.log_prior = lambda W: log_prior(W, Sigma_W_det=self.Sigma_W_det, Sigma_W_inv=self.Sigma_W_inv, D=self.D)

        self.potential_energy = lambda W: -1 * (log_lklhd(W) + self.log_prior(W))[0]
        self.kinetic_energy = lambda W: np.sum(W ** 2) / 2.0
        self.total_energy = lambda position, momentum: self.potential_energy(position) + self.kinetic_energy(momentum)

        self.sample_momentum = lambda n: random.normal(0, 1, size=self.D).reshape((1, self.D))
        self.grad_potential_energy = grad(self.potential_energy)

        if tune_params is None:
            self.tune_params = {'step_size': 0.1,
                                'leapfrog_steps': 10,
                                'total_samples': 1000,
                                'burn_in': 0.1,
                                'thinning_factor': 1,
                                'diagnostic_mode': diagnostic_mode}
        else:
            self.tune_params = tune_params

        self.accepts = 0.
        self.iterations = 0.
        self.trace = np.empty((1, self.D))
        self.potential_energy_trace = np.empty((1,))

        assert self.sample_momentum(1).shape == (1, self.D)
        assert isinstance(self.potential_energy(self.sample_momentum(1)), float)
        assert isinstance(self.kinetic_energy(self.sample_momentum(1)), float)
        assert isinstance(self.total_energy(self.sample_momentum(1), self.sample_momentum(1)), float)
        assert self.grad_potential_energy(self.sample_momentum(1)).shape == (1, self.D)

    def leap_frog(self, position_init, momentum_init):
        # initialize position
        position = position_init

        # half step update of momentum
        momentum = momentum_init - self.tune_params['step_size'] * self.grad_potential_energy(position_init) / 2

        # full leap frog steps
        for _ in range(self.tune_params['leapfrog_steps'] - 1):
            position += self.tune_params['step_size'] * momentum
            momentum -= self.tune_params['step_size'] * self.grad_potential_energy(position)
            assert not np.any(np.isnan(position))
            assert not np.any(np.isnan(momentum))

        # full step update of position
        position_proposal = position  # + self.tune_params['step_size'] * momentum
        # half step update of momentum
        momentum_proposal = momentum - self.tune_params['step_size'] * self.grad_potential_energy(position) / 2

        return position_proposal, momentum_proposal

    def hmc(self, position_current, momentum_current):
        # Refresh momentum
        momentum_current = self.sample_momentum(1)

        # Simulate Hamiltonian dynamics using Leap Frog
        position_proposal, momentum_proposal = self.leap_frog(position_current, momentum_current)

        # compute total energy in current position and proposal position
        current_total_energy = self.total_energy(position_current, momentum_current)
        proposal_total_energy = self.total_energy(position_proposal, momentum_proposal)

        # Output for diganostic mode
        if self.tune_params['diagnostic_mode']:
            print('potential energy change:',
                  self.potential_energy(position_current),
                  self.potential_energy(position_proposal))
            print('kinetic energy change:',
                  self.kinetic_energy(momentum_current),
                  self.kinetic_energy(momentum_proposal))
            print('total enregy change:',
                  current_total_energy,
                  proposal_total_energy)
            print('\n\n')

        # Metropolis Hastings Step
        # compute accept probability
        accept_prob = np.min([1, np.exp(current_total_energy - proposal_total_energy)])
        # accept proposal with accept probability
        if self.random.rand() < accept_prob:
            self.accepts += 1.
            position_current = np.copy(position_proposal)
            momentum_current = momentum_proposal

        return position_current, momentum_current

    def tuning(self, burn_in_period, position_init, momentum_init):
        # Determine check point
        if self.tune_params['diagnostic_mode']:
            check_point = 10
        else:
            check_point = 100

        # Initialize position and momentum
        position_current = position_init
        momentum_current = momentum_init

        # Tune step size param during burn-in period
        for i in range(burn_in_period):
            # Checks accept rate at check point iterations and adjusts step size
            if i % check_point == 0 and i > 0:
                accept_rate = self.accepts / i
                print('HMC {}: accept rate of {} with step size {}'.format(i, accept_rate * 100.,
                                                                           self.tune_params['step_size']))

                if accept_rate < 0.5:
                    self.tune_params['step_size'] *= 0.95
                if accept_rate > 0.8:
                    self.tune_params['step_size'] *= 1.05

            # perform one HMC step
            position_current, momentum_current = self.hmc(position_current, momentum_current)

        # Reset number of accepts
        self.accepts = 0

        return position_current, momentum_current

    def run_hmc(self, check_point, position_init, momentum_init):
        # Initialize position and momentum
        position_current = position_init
        momentum_current = momentum_init

        # Perform multiple HMC steps
        for i in range(self.tune_params['total_samples']):
            self.iterations += 1
            # output accept rate at check point iterations
            if i % check_point == 0 and i > 0:
                accept_rate = self.accepts * 100. / i
                print('HMC {}: accept rate of {}'.format(i, accept_rate))

            position_current, momentum_current = self.hmc(position_current, momentum_current)

            # add sample to trace
            if i % self.tune_params['thinning_factor'] == 0:
                self.trace = np.vstack((self.trace, position_current))
                self.potential_energy_trace = np.vstack((self.potential_energy_trace,
                                                         self.potential_energy(position_current)))

        self.trace = self.trace[1:]

    def sample(self, position_init=None, step_size=None, leapfrog_steps=None,
               total_samples=None, burn_in=None, thinning_factor=None, check_point=200,
               alpha=None, diagnostic_mode=None):

        # Sample random initial momentum
        momentum_init = self.sample_momentum(1)

        # Set model parameters
        if position_init is None:
            position_init = self.random.normal(0, 1, size=momentum_init.shape)
        else:
            assert position_init.shape == (1, self.D)
        if step_size is not None:
            self.tune_params['step_size'] = step_size
        if leapfrog_steps is not None:
            self.tune_params['leapfrog_steps'] = leapfrog_steps
        if total_samples is not None:
            self.tune_params['total_samples'] = total_samples
        if burn_in is not None:
            self.tune_params['burn_in'] = burn_in
        if thinning_factor is not None:
            self.tune_params['thinning_factor'] = thinning_factor
        if diagnostic_mode is not None:
            self.tune_params['diagnostic_mode'] = diagnostic_mode

        # Tune parameters during burn-in period
        burn_in_period = int(self.tune_params['burn_in'] * self.tune_params['total_samples'])
        position_current, momentum_current = self.tuning(burn_in_period, position_init, momentum_init)
        # Obtain samples from HMC using optimized parameters
        self.run_hmc(check_point, position_current, momentum_current)
        self.trace = self.trace[::self.tune_params['thinning_factor']]

    def get_posterior(self, n_samples):
        """take last n_sample weights"""
        posterior_samples = self.trace[-n_samples]
        return posterior_samples
