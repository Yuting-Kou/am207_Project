from abc import ABC

from autograd import grad
from autograd import numpy as np
from autograd.misc.optimizers import adam, sgd

from model import Model


class Inference(ABC):
    """
    An inference instance which has inference method and get_posterior methods.
    Conduct inference over subspace z.
    """
    submethods = {}

    @classmethod
    def register_submethods(cls, inference_type):
        def decorator(subclass):
            cls.submethods[inference_type] = subclass
            return subclass

        return decorator

    def __init__(self):
        self.Sigma_Z_det = self.Sigma_Z_inv = self.Mean_Z = None
        self.P = self.w_hat = None

    def get_posterior(self, n_samples):
        """
        return `n_samples` trace/samples of subspace weights.
        """
        pass

    def train(self, X, y, warm_start=False):
        """
        start to fit inference method to approximate the posterior.
        :param warm_start: bool, optional, default False When set to True, reuse the solution of the previous call
                            to fit as initialization, otherwise, just erase the previous solution.
        """
        pass

    def update_P_w_hat(self, P, w_hat):
        """ update P and w_hat"""
        self.P = P
        self.w_hat = w_hat

    def log_prior(self, z):
        """return log likelihood of prior distribution."""
        D = self.P.shape[1]
        z = z.reshape((-1, D))
        constant_W = -0.5 * (D * np.log(2 * np.pi) + np.log(self.Sigma_Z_det))
        exponential_W = -0.5 * np.diag(np.dot(np.dot(z - self.Mean_Z, self.Sigma_Z_inv), (z - self.Mean_Z).T))
        return constant_W + exponential_W

    def update_prior(self, Sigma_Z=None, Mean_Z=None):
        """update Sigma_Z"""
        D = self.P.shape[1]
        if Sigma_Z is None:
            Sigma_Z = np.eye(D)
        else:
            if len(Sigma_Z.shape) > 1:
                assert Sigma_Z.shape[0] == Sigma_Z.shape[1] == D
            else:
                # if sigma_Z is a number, turn it into (1,1)
                Sigma_Z = np.copy(Sigma_Z).reshape(1, 1)
        self.Sigma_Z_inv = np.linalg.inv(Sigma_Z)
        self.Sigma_Z_det = np.linalg.det(Sigma_Z)
        if Mean_Z is None:
            Mean_Z = np.zeros((1, D))
        else:
            Mean_Z = Mean_Z.reshape((1, D))
        self.Mean_Z = Mean_Z

    @classmethod
    def create(cls, inference_type, **kwargs):
        if inference_type not in cls.submethods:
            raise ValueError('Not implemented inference method: {}'.format(submethod))
        return cls.submethods[inference_type](**kwargs)


@Inference.register_submethods('BBB')
class BBB(Inference):
    """
    Implement BBVI using https://github.com/HIPS/autograd/blob/master/examples/black_box_svi.py

    === Usage ===

    # instantiate a BBB sampler
    BBB_sampler = BBB(model, P, w_hat, Sigma_Z, Mean_Z, Sigma_Y, random=0)

    #sample from the bayesian neural network posterior
    BBB_sampler.train( X, y, warm_start=False, S=None, max_iteration=None, step_size=None, verbose=None, init_mean=None,
              init_var=None, checkpoint=None, analytic_entropy=True, softplus=True, optimizer='adam', random_restart=1):

    # get posterior of z
    BBB_sampler.get_posterior(n_samples=100)
    === ===

    """

    def __init__(self, model: Model, P, w_hat, Sigma_Z=None, Mean_Z=None, Sigma_Y=None,
                 random=None, tune_params=None):
        """
        model: instance of our model object, which containing log_likelihood
        P, w_hat: subspace projection matrix P and shift vector w_hat.
        Sigma_Z: prior variance of subweights. Default is Identity Matrix
        Mean_Z: prior mean of subweights. Default is 0
        Sigma_Y: prior variance of observation(noise). Default is Identity matrix

        (default) tune_params: a disk of following info:
            {'step_size': 0.1,
            'S':2000,
            'max_iteration':20000,
           'verbose': True,
           'checkpoint':200,
           'init_mean': None,
           'init_log_std': None}
        self.variational_mu, self.variational_Sigma is the best result of BBVI.
        """
        if random is not None:
            if isinstance(random, int):
                self.random = np.random.RandomState(random)
            else:
                self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.model = model
        self.P = P
        self.D = self.P.shape[1]  # dimension of lower dim z.
        self.variational_mu = np.zeros(self.D)
        self.variational_Sigma = np.eye(self.D)
        self.w_hat = w_hat
        self.update_prior(Sigma_Z=Sigma_Z, Mean_Z=Mean_Z)
        self.ELBO = np.empty((1, 1))  # elbo values over optimization
        self.variational_params = np.empty((1, 2 * self.D))  # variational parmeeters over optimization

        if Sigma_Y is not None:
            model.update_Sigma_Y(Sigma_Y=Sigma_Y)

        self.log_density = None

        if tune_params is None:
            self.tune_params = {'step_size': 0.1,
                                'S': 100,
                                'max_iteration': 1000,
                                'checkpoint': 200,
                                'verbose': True,
                                'init_mean': None,
                                'init_var': None}
        else:
            self.tune_params = tune_params

        if 'verbose' not in self.tune_params.keys():
            self.tune_params['verbose'] = True
        if 'checkpoint' not in self.tune_params.keys():
            self.tune_params['checkpoint'] = 200
        if 'init_var' not in self.tune_params.keys():
            self.tune_params['init_var'] = None
        if 'init_mean' not in self.tune_params.keys():
            self.tune_params['init_mean'] = None

    def train(self, X, y, warm_start=False, S=None, max_iteration=None, step_size=None, verbose=None, init_mean=None,
              init_var=None, checkpoint=None, analytic_entropy=True, softplus=True, optimizer='adam', random_restart=1):
        """
        train the inference sampler
        :param optimizer: default is 'adam', could be 'adam', 'sgd'
        :param warm_start: bool, optional, default False When set to True, reuse the solution of the previous call
                            to fit as initialization, otherwise, just erase the previous solution.
        :param random_restart: randomly initialize the fits, and get the best results. default is 1.
            if random_restart >1, then the warm_start=False.
        """
        # set objective function
        self.log_density = lambda z, t: self.model.get_likelihood(X=X, y=y, z=z, P=self.P, w_hat=self.w_hat) \
                                        + self.log_prior(z).reshape(-1, 1)

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
            self.tune_params['init_mean'] = init_mean.reshape(self.D, )
        if init_var is not None:
            self.tune_params['init_var'] = init_var.reshape(self.D, )
        if checkpoint is not None:
            self.tune_params['checkpoint'] = checkpoint

        # randomly start to get best results
        optimal_ELBO = -1e16

        if random_restart <= 1:
            self.variational_inference(analytic_entropy=analytic_entropy, softplus=softplus, optimizer=optimizer,
                                       warm_start=warm_start)
            opt_param_index = np.argmax(self.ELBO[-100:])
            optimal_var_params = self.variational_params[-100:][opt_param_index]
            self.variational_mu = optimal_var_params[:self.D]
            self.variational_Sigma = np.diag(optimal_var_params[self.D:])
            self.optimal_trace=self.variational_params
        else:
            for i in range(1, random_restart):
                self.variational_inference(analytic_entropy=analytic_entropy, softplus=softplus, optimizer=optimizer,
                                           warm_start=False)
                local_opt = np.max(self.ELBO[-100:])

                if local_opt > optimal_ELBO:
                    optimal_ELBO = local_opt
                    opt_param_index = np.argmax(self.ELBO[-100:])
                    self.optimal_trace = self.variational_params
                    optimal_var_params = self.variational_params[-100:][opt_param_index]
                    self.variational_mu = optimal_var_params[:self.D]
                    self.variational_Sigma = np.diag(optimal_var_params[self.D:])

    def make_variational_objective(self, analytic_entropy, softplus, log_probability):
        """Notes: this is an extension of BBVI(http://arxiv.org/abs/1401.0118, and uses the
        local reparameterization trick from http://arxiv.org/abs/1506.02557), which is softplus=False,
        analytic_entropy=False"""

        if softplus:
            def unpack_params(params):
                mean, parametrized_var = params[:self.D], params[self.D:]
                var = np.log(1 + np.exp(parametrized_var))
                return mean, var
        else:
            def unpack_params(params):
                mean, parametrized_var = params[:self.D], params[self.D:]
                var = np.exp(parametrized_var)
                return mean, var

        if analytic_entropy:
            def entropy(var):
                ''' Gaussian entropy '''
                return 0.5 * np.sum(np.log(2 * np.pi * np.e) + np.log(var))

            def variational_objective(params, t):
                ''' Varational objective = H[q] + E_q[target] '''
                # unpack var parameters
                mean, var = unpack_params(params)
                # sample from q using reparametrization
                samples = self.random.randn(self.tune_params['S'], self.D) * var + mean
                # ELBO
                lower_bound = entropy(var) + np.mean(log_probability(samples, t))
                return -lower_bound
        else:
            def log_gaussian_pdf(samples, mean, var):
                assert samples.shape == (self.tune_params['S'], self.D)
                assert mean.shape == (1, self.D)

                Sigma = np.diag(var)
                Sigma_det = np.linalg.det(Sigma)
                Sigma_inv = np.linalg.det(Sigma)

                constant = -0.5 * (self.D * np.log(2 * np.pi) + np.log(Sigma_det))
                dist_to_mean = samples - mean
                exponential = -0.5 * np.diag(np.dot(np.dot(dist_to_mean, Sigma_inv), dist_to_mean.T))
                return constant + exponential

            def variational_objective(params, t):
                # unpack var parameters
                mean, var = unpack_params(params)
                # sample from q using reparametrization
                samples = self.random.randn(self.tune_params['S'], self.D) * var + mean
                # ELBO
                lower_bound = np.mean(log_probability(samples, t)
                                      - log_gaussian_pdf(samples, mean.reshape((1, self.D)), var))
                return -lower_bound

        return unpack_params, variational_objective, grad(variational_objective)

    def variational_inference(self, analytic_entropy=True, softplus=True, optimizer='adam', warm_start=False):
        # build variational objective.
        unpack_params, variational_objective, gradient = self.make_variational_objective(analytic_entropy, softplus,
                                                                                         self.log_density)

        def callback(params, iteration, g):
            ''' Actions per optimization step '''
            elbo = -variational_objective(params, iteration)
            self.ELBO = np.vstack((self.ELBO, elbo))
            mean, var = unpack_params(params)
            self.variational_params = np.vstack((self.variational_params, np.hstack((mean, var)).reshape((1, -1))))
            if self.tune_params['verbose'] and iteration % self.tune_params['checkpoint'] == 0:
                print(params)
                print("Iteration {} lower bound {}; gradient mag: {}".format(iteration, elbo, np.linalg.norm(
                    gradient(params, iteration))))



        # initialize variational parameters
        if self.tune_params['init_mean'] is None:
            self.tune_params['init_mean'] = self.random.normal(0, 0.1, size=self.D)
        if self.tune_params['init_var'] is None:
            self.tune_params['init_var'] = self.random.normal(0, 0.1, size=self.D)
        init_params = np.concatenate([self.tune_params['init_mean'], self.tune_params['init_var']])
        assert len(init_params) == 2 * self.D

        if not warm_start:
            # reset parameters
            self.ELBO = np.empty((1, 1))
            self.variational_params = np.empty((1, 2 * self.D))

        # perform gradient descent
        if optimizer == 'adam':
            # using adam (a type of gradient-based optimizer)
            adam(gradient, init_params, step_size=self.tune_params['step_size'],
                 num_iters=self.tune_params['max_iteration'], callback=callback)
        elif optimizer == 'sgd':
            # using sgd
            sgd(gradient, init_params, step_size=self.tune_params['step_size'],
                num_iters=self.tune_params['max_iteration'], callback=callback)

        self.variational_params = self.variational_params[1:]
        self.ELBO = self.ELBO[1:]

    def get_posterior(self, n_samples):
        """return posterior z of model"""
        return self.random.multivariate_normal(self.variational_mu, self.variational_Sigma,
                                               size=n_samples).reshape((-1, self.D))


@Inference.register_submethods('HMC')
class HMC(Inference):
    """
    Implement HMC using hw7 solutions.

    === Usage ===

    # instantiate an HMC sampler
    HMC_sampler = HMC(model, P, w_hat, Sigma_Z, Mean_Z, Sigma_Y, random=0)

    #sample from the bayesian neural network posterior
    HMC_sampler.train(position_init=position_init,
                       step_size=step_size,
                       leapfrog_steps=leapfrog_steps,
                       total_samples=total_samples,
                       burn_in=burn_in,
                       thinning_factor=thinning_factor)
    === ===
    """

    def __init__(self, model, P, w_hat, potential_energy=None, kinetic_energy=None, kinetic_energy_distribution=None,
                 Sigma_Z=None, Mean_Z=None, Sigma_Y=None, random=None, diagnostic_mode=False, tune_params=None):
        """
        (default) potential_energy = -(log_lklhd + log_prior)
        (default) kinetic_energy = lambda W: np.sum(W**2) / 2.0
        (default) kinetic_energy_distribution = lambda D: random.normal(0, 1, size=D)

        params is a dictionary includes at least following information:
            - D: dimension of weights
            - Sigma_Z: prior variance of weights

        (default) tuning_params: a disk of following info:
            {'step_size': 0.1,
           'leapfrog_steps': 10,
           'total_samples': 1000,
           'burn_in': 0.1,
           'thinning_factor': 1,
           'warm_start':True,
           'diagnostic_mode': diagnostic_mode}
        """
        if random is not None:
            if isinstance(random, int):
                self.random = np.random.RandomState(random)
            else:
                self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.model = model
        self.P = P
        self.D = self.P.shape[1]  # dimension of lower dim z.
        self.w_hat = w_hat
        self.update_prior(Sigma_Z=Sigma_Z, Mean_Z=Mean_Z)
        if Sigma_Y is not None:
            model.update_Sigma_Y(Sigma_Y=Sigma_Y)

        self.potential_energy = None  # need X,y
        self.kinetic_energy = lambda z: np.sum(z ** 2) / 2.0
        self.total_energy = None
        self.sample_momentum = lambda n: self.random.normal(0, 1, size=self.D).reshape((1, self.D))
        self.grad_potential_energy = None

        if tune_params is None:
            self.tune_params = {'step_size': 0.1,
                                'leapfrog_steps': 10,
                                'total_samples': 1000,
                                'burn_in': 0.1,
                                'warm_start': True,
                                'thinning_factor': 1,
                                'diagnostic_mode': diagnostic_mode}
        else:
            self.tune_params = tune_params
        if 'burn_in' not in self.tune_params.keys():
            self.tune_params['burn_in'] = 0.1
        if 'thinning_factor' not in self.tune_params.keys():
            self.tune_params['thinning_factor'] = 1
        if 'check_point' not in self.tune_params.keys():
            self.tune_params['check_point'] = 200

        self.accepts = 0.
        self.iterations = 0.
        self.trace = np.empty((1, self.D))
        self.potential_energy_trace = np.empty((1,))

        assert self.sample_momentum(1).shape == (1, self.D)
        assert isinstance(self.kinetic_energy(self.sample_momentum(1)), float)

    def train(self, X, y, warm_start=True, position_init=None, step_size=None, leapfrog_steps=None,
              total_samples=None, burn_in=None, thinning_factor=None, check_point=200, alpha=None,
              diagnostic_mode=None):
        self.potential_energy = lambda z: -1 * (
                    self.model.get_likelihood(X=X, y=y, z=z, P=self.P, w_hat=self.w_hat).reshape(-1)
                    + self.log_prior(z))[0]
        self.total_energy = lambda position, momentum: self.potential_energy(position) + self.kinetic_energy(momentum)
        self.grad_potential_energy = grad(self.potential_energy)

        assert self.grad_potential_energy(self.sample_momentum(1)).shape == (1, self.D)
        assert isinstance(self.potential_energy(self.sample_momentum(1)), float)
        assert isinstance(self.total_energy(self.sample_momentum(1), self.sample_momentum(1)), float)

        # Sample random initial momentum
        momentum_init = self.sample_momentum(1)

        # Set model parameters
        if position_init is None:
            position_init = self.random.normal(0, 1, size=momentum_init.shape)
        else:
            position_init = position_init.reshape(momentum_init.shape)

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
        if check_point is not None:
            self.tune_params['check_point'] = check_point
        if warm_start is not None:
            self.tune_params['warm_start'] = warm_start

        # Tune parameters during burn-in period
        burn_in_period = int(self.tune_params['burn_in'] * self.tune_params['total_samples'])
        if not self.tune_params['warm_start'] or self.trace.shape[0] == 1:
            position_current, momentum_current = self.tuning(burn_in_period, position_init)
        else:
            position_current = self.trace[-1]
        # Obtain samples from HMC using optimized parameters
        self.run_hmc(self.tune_params['check_point'], position_current)
        self.trace = self.trace[::self.tune_params['thinning_factor']]

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

    def hmc(self, position_current):
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

    def tuning(self, burn_in_period, position_init):
        # Determine check point
        if self.tune_params['check_point'] is None:
            check_point = 10
        else:
            check_point = self.tune_params['check_point']

        # Initialize position and momentum
        position_current = position_init

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
            position_current, momentum_current = self.hmc(position_current)

        # Reset number of accepts
        self.accepts = 0

        return position_current, momentum_current

    def run_hmc(self, check_point, position_init):
        # Initialize position and momentum
        position_current = position_init

        # Perform multiple HMC steps
        for i in range(self.tune_params['total_samples']):
            self.iterations += 1
            # output accept rate at check point iterations
            if i % check_point == 0 and i > 0:
                accept_rate = self.accepts * 100. / i
                print('HMC {}: accept rate of {}'.format(i, accept_rate))

            position_current, momentum_current = self.hmc(position_current)

            # add sample to trace
            if i % self.tune_params['thinning_factor'] == 0:
                self.trace = np.vstack((self.trace, position_current))
                self.potential_energy_trace = np.vstack((self.potential_energy_trace,
                                                         self.potential_energy(position_current)))

        self.trace = self.trace[1:]

    def get_posterior(self, n_samples):
        """take last n_sample weights"""
        posterior_samples = self.trace[-n_samples:]
        return posterior_samples
