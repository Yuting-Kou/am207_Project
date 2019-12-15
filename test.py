import matplotlib.pyplot as plt
from autograd import numpy as np

from src.Inference import Inference
from src.Subspace import Subspace
from src.model import Model

if __name__ == '__main__':
    # test
    # print("All possible models:{}".format(Model.modeltype))
    data = np.load(r'.\example\data.npy')
    # data = np.load('data.npy')
    x, y = data[:, 0], data[:, 1]

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

    # ---------------- Core thing! ------------------ #
    # set up model, subspace, inference
    my_nn = Model.create(submodel_type="Feedforward", architecture=architecture)
    my_subspace = Subspace.create(subspace_type="random", model=my_nn, n_subspace=2)
    my_subspace.collect_vector(X=x, y=y)
    P, w = my_subspace.get_space()
    # my_inference = Inference.create(inference_type="HMC", model=my_nn, P=P, w_hat=w)
    my_inference = Inference.create(inference_type="BBB", model=my_nn, P=P, w_hat=w)

    # use MSE result as params_init
    params = {'step_size': 1e-3,
              'max_iteration': 5000,
              'random_restarts': 1}

    # fit my neural network to minimize MSE on the given data
    #my_nn.fit(x_train=x.reshape((1, -1)), y_train=y.reshape((1, -1)), params=params)

    # get initial weights (in subspace dimension!!)
    position_init = my_nn.get_z_from_W(weights=my_nn.weights, P=P, w_hat=w)
    # # notice since P from PCA is too small (almost 1e-16), the inversed z is too large! Maybe sth wrong at PCA!
    # so if P is too small, it is better to start with position_init=None

    # train
    # for HMC
    # my_inference.train(X=x, y=y, warm_start=True, position_init=position_init)
    # for BBB
    my_inference.train(X=x, y=y, warm_start=True, position_init=position_init)

    # get posterior z
    n_sample = 10
    post_sample = my_inference.get_posterior(n_samples=n_sample).reshape(-1, 2)
    x_test = np.linspace(-8, 8, 100)
    y_test = np.reshape(
        [my_nn.forward(P=P, w_hat=w, z=post_sample[i], X=x_test.reshape(1, -1)) for i in range(n_sample)],
        (n_sample, -1)) \
             + np.random.normal(0, my_nn.Sigma_Y_det ** 0.5, size=(n_sample, len(x_test)))
    # because here Sigma_Y is 1-D, so determinants=its value

    # plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.title('Posterior Predictive of Bayesian NN ')
    plt.ylim(-15, 15)
    for i in range(n_sample):
        plt.plot(x_test, y_test[i], color='red', alpha=max(1 / n_sample, 0.1))
    plt.scatter(x, y, color='black', label='data')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color='black')
    plt.plot(x_test, y_test.mean(0), color='red', label='posterior predictive mean')
    plt.fill_between(x_test, np.percentile(y_test, 0.25, axis=0), np.percentile(y_test, 97.5, axis=0),
                     color='red', label='95% CI', alpha=0.5)
    plt.legend(loc='best')
    plt.title('Posterior Predictive of Bayesian NN with 95% CI')
    plt.grid()
    plt.ylim(-15, 15)
    plt.show()
