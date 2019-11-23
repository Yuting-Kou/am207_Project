import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np


def get_initialization(init_type, net, loader, optimizer, criterion, train_params):
    def apply_weights_init(type):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                n = m.in_features
                y = 1.0 / np.sqrt(n)
                if type == 0:
                    m.weight.data.uniform_(-y, y)
                elif type == 1:
                    m.weight.data.normal_(0.0, 1 / np.sqrt(y))
                m.bias.data.fill_(0)

        return weights_init

    net.apply(apply_weights_init(init_type))
    for epoch in range(train_params['epochs']):
        running_loss = 0
        for k, data in enumerate(loader, 0):
            X, y = data

            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss
            if k % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, k + 1, running_loss / 2000))
                running_loss = 0.0
    return net.state_dict()