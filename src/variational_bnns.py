import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.layers import BBBLayer, MNFLayer, BBHLayer, MVGLayer

import math


def get_activation(activation_type='relu'):
    if activation_type=='relu':
        return F.relu
    elif activation_type=='tanh':
        return torch.tanh
    else:
        print('activation not recognized')

def get_layer(layer_type='bbb'):
    if layer_type == 'bbb':
        return BBBLayer
    if layer_type == 'mnf':
        return MNFLayer
    if layer_type == 'bbh':
        return BBHLayer
    if layer_type == 'mvg':
        return MVGLayer
    else:
        print('layer not recognized')

class BNN(nn.Module):
    """
    Fully connected BNN with constant width
    """
    def __init__(self, dim_in, dim_out, dim_hidden=50, n_layers=1, activation_type='relu', layer_type='bbb', sigma_y=.1, **kwargs):
        super(BNN, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.n_layers = n_layers
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.sigma_y = torch.tensor([sigma_y])

        self.activation = get_activation(self.activation_type)
        layer = get_layer(self.layer_type)

        # layers
        self.fc_hidden = []
        self.fc1 = layer(self.dim_in, self.dim_hidden)
        self.fc_hidden = nn.ModuleList([layer(self.dim_hidden, self.dim_hidden) for _ in range(self.n_layers-1)])
        self.fc_out = layer(self.dim_hidden, self.dim_out)

    def forward(self, x, sample=True):

        x = self.fc1(x, sample=sample)
        x = self.activation(x)

        for layer in self.fc_hidden:
            x = layer(x, sample=sample)
            x = self.activation(x)

        return self.fc_out(x, sample=sample)

    def kl_divergence(self):
        kld = self.fc1.kl_divergence() + self.fc_out.kl_divergence()
        for layer in self.fc_hidden:
            kld += layer.kl_divergence()
        return kld

    def neg_log_prob(self, y_observed, y_pred, mask=None):
        v = self.sigma_y ** 2

        N = y_observed.shape[0]

        #log_prob = -0.5 * N * math.log(2 * math.pi) - 0.5 * N * torch.log(v) + - 0.5 * (y_observed - y_pred)**2 / v

        const_term = -0.5 * N * (math.log(2*math.pi) + torch.log(v))
        if mask is None:
            dist_term = -torch.sum((y_observed - y_pred).pow(2)) / (2.*v)
        else:
            dist_term = -torch.sum((y_observed - y_pred).pow(2) * mask) / (2.*v)
        log_prob = const_term + dist_term

        return -log_prob


class VariationalBNN(object):
    """Implements an approximate Bayesian NN using Variational Inference."""
    def __init__(self, dim_in, dim_out, learning_rate, device=torch.device('cpu'), **args_bnn):
        self.bnn = BNN(dim_in, dim_out, **args_bnn).to(device)
        self.optimizer = torch.optim.Adam(self.bnn.parameters(), lr=learning_rate)

    def get_n_parameters(self):
        n_param=0
        for p in self.bnn.parameters():
            n_param+=np.prod(p.shape)
        return n_param

    def train(self, x, y, n_epochs, mask=None, print_freq=None):

        #print("Training for {} epochs...".format(n_epochs))

        elbo = torch.zeros(n_epochs)
        for epoch in range(n_epochs):

            # forward
            y_pred = self.bnn(x)

            kl_divergence = 1/x.shape[0]*self.bnn.kl_divergence()
            neg_log_prob = 1/x.shape[0]*self.bnn.neg_log_prob(y, y_pred, mask)
            loss = neg_log_prob + kl_divergence
            #loss = kl_divergence

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            elbo[epoch] = -loss.item()

            if print_freq is not None:
                if (epoch + 1) % print_freq == 0:
                    print('Epoch[{}/{}], log_prob: {:.6f}, kl: {:.6f}, elbo: {:.6f}'\
                        .format(epoch+1, n_epochs, -neg_log_prob.item(), kl_divergence.item(), -loss.item()))

        return elbo

