from Subspace import Subspace
from model import Model
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.utils import _pair
import util
from collections import OrderedDict

@Subspace.register_subclass('curve')
class CurveSpace(Subspace):
    def __init__(self, curve_net_gen, params_curve, loader, criterion,
                 nn=None, net=None, **kwargs):
        """
        Generate basis of curve subspace
        :param net: BaseNet
        :param loader:
        :param params_base: epochs
        :param optimizer:
        :param criterion:
        :param n_parameters:
        :param curve_net_gen: class of CurveNet
        :param params_curve: epochs, sample_size
        """
        self.base_type = -1

        if nn is not None:
            self.nn = nn
            self.params_base = kwargs['params_base']
            self.X = kwargs['X']
            self.y = kwargs['y']
            self.base_type = 0
        else:
            self.params_base = kwargs['params_base']
            self.optimizer = kwargs['optimizer']
            self.base_type = 1

        self.net = net
        self.loader = loader
        self.criterion = criterion
        self.curve_net_gen = curve_net_gen
        self.params_curve = params_curve
        self.curve_net = None

    def get_endpoint(self, callback = 0):
        endpoint = []

        if self.base_type == 0:
            self.params_base['callback'] = callback
            self.params_base['check_point'] = callback
            for i in range(2):
                self.nn.weights = np.random.normal(0, 1, size=self.nn.weights.shape)
                self.nn.fit(x_train=self.X.reshape((1, -1)), y_train=self.y.reshape((1, -1)),
                            params=self.params_base)
                #print()
                endpoint.append(self.copy_weight(self.nn.weights.flatten().copy()))
        else:
            for i in range(2):
                endpoint.append(util.get_initialization(self.net, self.loader,
                                                        self.optimizer, self.criterion,
                                                        self.params_base, i, callback))
        return endpoint

    def copy_weight(self, w):
        res = OrderedDict()
        cnt = 0
        for key, val in self.net.state_dict().items():
            if val.dim() == 1:
                length = len(val)
            else:
                length = len(val.squeeze())
            res[key] = torch.Tensor(w[cnt:cnt+length].reshape(val.shape))
            cnt += length
        return res

    def train_midpoint(self, endpoint, epochs, callback):
        w1, w2 = endpoint[0], endpoint[1]
        if self.curve_net is None:
            self.curve_net = self.curve_net_gen(w1, w2)

        opt = optim.Adam(self.curve_net.parameters(), lr = 0.001)

        #callback = 10
        for epoch in range(epochs):

            running_loss = 0
            for i, data in enumerate(self.loader, 0):
                X, y = data

                opt.zero_grad()
                loss = 0
                for k in range(self.params_curve['sample_size']):
                    outputs = self.curve_net(X)
                    loss += self.criterion(outputs, y)
                loss /= self.params_curve['sample_size']

                loss.backward()
                opt.step()

                running_loss += loss
            if callback != 0 and epoch % callback == 0:
                print('[epoch %d] loss: %.3f' %
                      (epoch + 1, running_loss / callback / len(self.loader)))

    def collect_vector(self, epochs, X = None, y = None, restart = False, callback = 0):
        """
        :param epochs: for training midpoint
        :param X:
        :param y:
        :param restart: True - get new endpoints
        :param callback: printing frequency
        :return:
        """
        if self.curve_net is None or restart:
            self.endpoint = self.get_endpoint()
        self.train_midpoint(self.endpoint, epochs, callback)

        self.e1 = self.flatten_weights(self.endpoint[0])
        self.e2 = self.flatten_weights(self.endpoint[1])
        self.m = self.flatten_weights(self.curve_net.state_dict())
        #self.m = self.flatten_weights(self.endpoint[2])

        w0 = (self.e1 + self.e2) / 2
        v1 = (self.e1 - w0) / np.linalg.norm(self.e1 - w0)
        v2 = (self.m - w0) / np.linalg.norm(self.m - w0)

        self.basis = np.vstack((v1, v2)).T
        self.w_hat = w0

    def get_space(self):
        return self.basis, self.w_hat

    @staticmethod
    def flatten_weights(para):
        w = np.array([])
        for val in para.values():
            w = np.append(w, val.numpy().flatten())
        return w


class Linear(nn.Module):
    def __init__(self, in_features, out_features, start_point, end_point, bias = True):
        """
        Linear layer with self-customized parameters
        :param in_features:
        :param out_features:
        :param start_point: list of tensors
        :param end_point: list of tensors
        :param bias:
        """
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parameter_names = ('weight', 'bias')
        self.start_point = start_point
        self.end_point = end_point

        self.register_parameter('weight',
                                nn.Parameter(torch.Tensor(out_features, in_features)))
        if bias:
            self.register_parameter('bias',
                                    nn.Parameter(torch.Tensor(out_features)))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.in_features)
        getattr(self, 'weight').data.uniform_(-stdv, stdv)
        bias = getattr(self, 'bias')
        if bias is not None:
            bias.data.uniform_(-stdv, stdv)

    def forward(self, input, t):
        weight = getattr(self, 'weight')
        bias = getattr(self, 'bias')
        #p1, p2 = list(self.start_point.values()), list(self.end_point.values())
        p1, p2 = self.start_point, self.end_point
        if t <= 0.5:
            weight_t = 2 * (t * weight + (0.5-t) * p1[0])
        else:
            weight_t = 2 * ((t-0.5)*p2[0] + (1-t) * weight)

        if bias is not None:
            if t <= 0.5:
                bias_t = 2 * (t * bias + (0.5 - t) * p1[1])
            else:
                bias_t = 2 * ((t - 0.5) * p2[1] + (1 - t) * bias)
        else:
            bias_t = None
        return F.linear(input, weight_t, bias_t)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, start_point, end_point,
                 stride = 1, padding = 0, dilation = 1, groups = 1, bias = True):
        """
        Conv2d layer with self-customized parameters
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param start_point: list of tensors
        :param end_point: list of tensors
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """
        super(Conv2d, self).__init__()
        self.parameter_names = ('weight', 'bias')

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.start_point = start_point
        self.end_point = end_point

        self.register_parameter(
            'weight',
            nn.Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size)
            )
        )
        if bias:
            self.register_parameter(
                'bias',
                nn.Parameter(torch.Tensor(out_channels))
            )
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1./ np.sqrt(n)
        getattr(self, 'weight').data.uniform_(-stdv, stdv)
        bias = getattr(self, 'bias')
        if bias is not None:
            bias.data.uniform_(-stdv, stdv)

    def forward(self, input, t):
        weight = getattr(self, 'weight')
        bias = getattr(self, 'bias')
        #p1, p2 = list(self.start_point.values()), list(self.end_point.values())
        p1, p2 = self.start_point, self.end_point
        if t <= 0.5:
            weight_t = 2 * (t * weight + (0.5 - t) * p1[0])
        else:
            weight_t = 2 * ((t - 0.5) * p2[0] + (1 - t) * weight)

        if bias is not None:
            if t <= 0.5:
                bias_t = 2 * (t * bias + (0.5 - t) * p1[1])
            else:
                bias_t = 2 * ((t - 0.5) * p2[1] + (1 - t) * bias)
        else:
            bias_t = None
        return F.conv2d(input, weight_t, bias_t, self.stride,
                        self.padding, self.dilation, self.groups)


"""Example base net class"""
class BaseNet(nn.Module):
    def __init__(self, input_size, width, hidden_layer=1):
        super(BaseNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, width))
        for i in range(hidden_layer - 1):
            self.layers.append(nn.Linear(width, width))
        self.layers.append(nn.Linear(width, 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.exp(-layer(x)**2)
        x = self.layers[-1](x)
        return x

"""Example curve net class"""
class CurveNet(nn.Module):
    def __init__(self, input_size, width, hidden_layer, start_point, end_point):
        super(CurveNet, self).__init__()
        self.layers = nn.ModuleList()

        def get_weight(i):
            coeff = names[int(2 * i)]
            bias = names[int(2 * i) + 1]
            return [start_point[coeff], start_point[bias]], [end_point[coeff], end_point[bias]]

        names = list(start_point.keys())
        self.layers.append(Linear(input_size, width, *get_weight(0)))
        for i in range(hidden_layer - 1):
            self.layers.append(Linear(width, width, *get_weight(i + 1)))
        self.layers.append(Linear(width, 1, *get_weight(hidden_layer)))

    def forward(self, x, t=None):
        if t is None:
            t = x.data.new(1).uniform_()
        for layer in self.layers[:-1]:
            x = torch.exp(-layer(x, t)**2)
        x = self.layers[-1](x, t)
        return x
