from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

import pandas as pd
from src.Inference import HMC, BBB

import matplotlib.pyplot as plt
from src.model import Model


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
    nn = Model.create(submodel_type="Feedforward", architecture=architecture, random=random)

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
