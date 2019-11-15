from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

import pandas as pd
from src.Inference import HMC, BBB

import matplotlib.pyplot as plt


class Feedforward:
    def __init__(self, architecture, random=None, weights=None):
        self.params = {'H': architecture['width'],
                       'L': architecture['hidden_layers'],
                       'D_in': architecture['input_dim'],
                       'D_out': architecture['output_dim'],
                       'activation_type': architecture['activation_fn_type'],
                       'activation_params': architecture['activation_fn_params']}

        self.D = ((architecture['input_dim'] * architecture['width'] + architecture['width'])
                  + (architecture['output_dim'] * architecture['width'] + architecture['output_dim'])
                  + (architecture['hidden_layers'] - 1) * (architecture['width'] ** 2 + architecture['width'])
                  )

        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.h = architecture['activation_fn']

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            self.weights = weights

        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))

    def forward(self, weights, x):
        ''' Forward pass given weights and input '''
        H = self.params['H']
        D_in = self.params['D_in']
        D_out = self.params['D_out']

        assert weights.shape[1] == self.D

        if len(x.shape) == 2:
            assert x.shape[0] == D_in
            x = x.reshape((1, D_in, -1))
        else:
            assert x.shape[1] == D_in

        weights = weights.T

        # input to first hidden layer
        W = weights[:H * D_in].T.reshape((-1, H, D_in))
        b = weights[H * D_in:H * D_in + H].T.reshape((-1, H, 1))
        input = self.h(np.matmul(W, x) + b)
        index = H * D_in + H

        assert input.shape[1] == H

        # additional hidden layers
        for _ in range(self.params['L'] - 1):
            before = index
            W = weights[index:index + H * H].T.reshape((-1, H, H))
            index += H * H
            b = weights[index:index + H].T.reshape((-1, H, 1))
            index += H
            output = np.matmul(W, input) + b
            input = self.h(output)

            assert input.shape[1] == H

        # output layer
        W = weights[index:index + H * D_out].T.reshape((-1, D_out, H))
        b = weights[index + H * D_out:].T.reshape((-1, D_out, 1))
        output = np.matmul(W, input) + b
        assert output.shape[1] == self.params['D_out']

        return output

    def make_objective(self, x_train, y_train, reg_param=None):
        ''' Make objective functions: depending on whether or not you want to apply l2 regularization '''

        if reg_param is None:

            def objective(W, t):
                squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1) ** 2
                sum_error = np.sum(squared_error)
                return sum_error

            return objective, grad(objective)

        else:

            def objective(W, t):
                squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1) ** 2
                mean_error = np.mean(squared_error) + reg_param * np.linalg.norm(W)
                return mean_error

            return objective, grad(objective)

    def fit(self, x_train, y_train, params, reg_param=None):
        ''' Wrapper for MLE through gradient descent '''
        assert x_train.shape[0] == self.params['D_in']
        assert y_train.shape[0] == self.params['D_out']

        ### make objective function for training
        self.objective, self.gradient = self.make_objective(x_train, y_train, reg_param)

        ### set up optimization
        step_size = 0.01
        max_iteration = 5000
        check_point = 100
        weights_init = self.weights.reshape((1, -1))
        mass = None
        optimizer = 'adam'
        random_restarts = 5

        if 'step_size' in params.keys():
            step_size = params['step_size']
        if 'max_iteration' in params.keys():
            max_iteration = params['max_iteration']
        if 'check_point' in params.keys():
            self.check_point = params['check_point']
        if 'init' in params.keys():
            weights_init = params['init']
        if 'call_back' in params.keys():
            call_back = params['call_back']
        if 'mass' in params.keys():
            mass = params['mass']
        if 'optimizer' in params.keys():
            optimizer = params['optimizer']
        if 'random_restarts' in params.keys():
            random_restarts = params['random_restarts']

        def call_back(weights, iteration, g):
            ''' Actions per optimization step '''
            objective = self.objective(weights, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.weight_trace = np.vstack((self.weight_trace, weights))
            if iteration % check_point == 0:
                print("Iteration {} lower bound {}; gradient mag: {}".format(iteration, objective, np.linalg.norm(
                    self.gradient(weights, iteration))))

        ### train with random restarts
        optimal_obj = 1e16
        optimal_weights = self.weights

        for i in range(random_restarts):
            if optimizer == 'adam':
                adam(self.gradient, weights_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
            local_opt = np.min(self.objective_trace[-100:])

            if local_opt < optimal_obj:
                opt_index = np.argmin(self.objective_trace[-100:])
                self.weights = self.weight_trace[-100:][opt_index].reshape((1, -1))
            weights_init = self.random.normal(0, 1, size=(1, self.D))

        self.objective_trace = self.objective_trace[1:]
        self.weight_trace = self.weight_trace[1:]


if __name__ == '__main__':
    data = pd.read_csv(r'C:\Users\Lenovo\Downloads\HW7_solutions\HW7_data.csv')
    x = data['x'].values
    y = data['y'].values

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

    # set random state to make the experiments replicable
    rand_state = 127
    random = np.random.RandomState(rand_state)

    # initialize with a MSE weights
    nn = Feedforward(architecture=architecture, random=random)

    # change prior over the weights
    weights = nn.random.normal(0, 5, size=(1, nn.D))
    nn.weights = weights

    # initialize with MLE
    params = {'step_size': 1e-3,
              'max_iteration': 2000,
              'random_restarts': 1}

    # fit my neural network to minimize MSE on the given data
    nn.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)

    sigma_y = 0.5
    N = len(x)
    sigma_W = np.eye(nn.D) * 25


    def log_lklhd(w, x=x, y=y):
        assert len(w.shape) == 2 and w.shape[1] == nn.D
        S = w.shape[0]
        constant = (-np.log(sigma_y) - 0.5 * np.log(2 * np.pi)) * N
        exponential = -0.5 * sigma_y ** -2 * np.sum((y.reshape((1, 1, N)) - nn.forward(w, x.reshape(1, -1))) ** 2,
                                                    axis=2).flatten()
        assert exponential.shape == (S,)
        return constant + exponential


    # test HMC
    # instantiate an HMC sampler
    position_init = nn.weights.reshape((1, nn.D))
    # leap-frog step size
    step_size = 1e-2
    # leap-frog steps
    leapfrog_steps = 20
    # number of total samples after burn-in
    total_samples = 1000
    # percentage of samples to burn
    burn_in = 0.1
    # thinning factor
    thinning_factor = 1

    HMC_sampler = HMC(log_lklhd=log_lklhd, D=nn.D, Sigma_W=sigma_W, random=random)

    # sample from the bayesian neural network posterior
    HMC_sampler.sample(position_init=position_init,
                       step_size=step_size,
                       leapfrog_steps=leapfrog_steps,
                       total_samples=total_samples,
                       burn_in=burn_in,
                       thinning_factor=thinning_factor)

    # # visualize the traceplots for all of the 16 neural network parameters
    # fig, ax = plt.subplots(4, 4, figsize=(20, 20))
    # for i in range(nn.D):
    #     row = i // 4
    #     col = i % 4
    #     ax[row, col].plot(range(total_samples), HMC_sampler.trace[:, i], color='gray', alpha=0.7)
    # fig.suptitle('Trace plots for all neural network parameters | HMC', y=0.9)
    # plt.show()

    # test BBB
    BBB_sampler = BBB(log_lklhd=log_lklhd, D=nn.D, Sigma_W=sigma_W, random=random)
    BBB_sampler.variational_inference(step_size=5e-3,
                                      S=2000, max_iteration=1000, init_mean=nn.weights.reshape(-1),
                                      init_log_std=-10 * np.ones(nn.D))

    # Visualization
    n_sample = 100
    BBB_post_sample = BBB_sampler.get_posterior(n_sample).reshape(-1, nn.D)
    HMC_post_sample = HMC_sampler.get_posterior(n_sample).reshape(-1, nn.D)
    x_test = np.linspace(-8, 8, 100)
    y_test_hmc = nn.forward(HMC_post_sample, x_test.reshape(1,-1)).reshape((n_sample, -1)) + np.random.normal(0, sigma_y ** 0.5,
                                                                                                size=(
                                                                                                    n_sample,
                                                                                                    len(x_test)))
    y_test_bbb = nn.forward(BBB_post_sample, x_test.reshape(1,-1)).reshape((n_sample, -1)) + np.random.normal(0, sigma_y ** 0.5,
                                                                                                size=(
                                                                                                    n_sample,
                                                                                                    len(x_test)))

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(x, y, color='black', label='data')
    plt.grid()
    plt.title('Posterior Predictive of Bayesian NN | HMC')
    plt.ylim(-15, 15)
    for i in range(n_sample):
        plt.plot(x_test, y_test_hmc[i], color='red', alpha=0.2)
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.scatter(x, y, color='black')
    plt.plot(x_test, y_test_hmc.mean(0), color='red', label='posterior predictive mean')
    plt.fill_between(x_test, np.percentile(y_test_hmc, 0.25, axis=0), np.percentile(y_test_hmc, 97.5, axis=0),
                     color='red', label='95% CI', alpha=0.5)
    plt.legend(loc='best')
    plt.title('Posterior Predictive of Bayesian NN with 95% CI |HMC')
    plt.grid()
    plt.ylim(-15, 15)
    plt.subplot(2, 2, 3)
    plt.scatter(x, y, color='black', label='data')
    plt.grid()
    plt.title('Posterior Predictive of Bayesian NN | BBVI')
    for i in range(100):
        plt.plot(x_test, y_test_bbb[i], color='red', alpha=0.2)
    plt.legend()
    plt.ylim(-15, 15)
    plt.subplot(2, 2, 4)
    plt.scatter(x, y, color='black')
    plt.plot(x_test, y_test_bbb.mean(0), color='red', label='posterior predictive mean')
    plt.fill_between(x_test, np.percentile(y_test_bbb, 0.25, axis=0), np.percentile(y_test_bbb, 97.5, axis=0),
                     color='red',
                     label='95% CI', alpha=0.5)
    plt.legend(loc='best')
    plt.title('Posterior Predictive of Bayesian NN with 95% CI |BBVI')
    plt.ylim(-15, 15)
    plt.grid()
    plt.show()
