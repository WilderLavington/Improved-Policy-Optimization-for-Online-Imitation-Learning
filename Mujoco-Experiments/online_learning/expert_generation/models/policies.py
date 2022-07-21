import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
import numpy as np
from copy import deepcopy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None,
                    model_type='nn', bandwidth=0., transform_rv=True, nonlin='relu',
                    clamp=True):
        super(GaussianPolicy, self).__init__()
        self.model_type = model_type
        self.transform_rv = transform_rv
        self.clamp = clamp
        # nn model
        if self.model_type == 'nn':
            self.nonlin = nonlin
            self.linear1 = nn.Linear(num_inputs, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.mean_linear = nn.Linear(hidden_dim, num_actions)
            self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        # linear / rbf model
        elif self.model_type == 'rbf':
            self.hidden_dim = hidden_dim
            self.state_size = num_inputs
            self.linear_model_mean = nn.Linear(hidden_dim, num_actions)
            self.linear_model_sd = nn.Linear(hidden_dim, num_actions)
            self.bandwidth = torch.exp(torch.tensor(bandwidth))
        elif self.model_type == 'linear':
            self.linear_model_mean = nn.Linear(num_inputs, num_actions)
            self.linear_model_sd = nn.Linear(num_inputs, num_actions)
        else:
            raise Exception()

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        # generate random features transforms
        if self.model_type == 'rbf':
            self.scale = torch.randn((self.hidden_dim, self.state_size))
            self.shift = torch.FloatTensor(self.hidden_dim, 1).uniform_(-np.pi, np.pi)

    def forward(self, state):
        if self.model_type == 'nn':
            if self.nonlin == 'relu':
                x = F.relu(self.linear1(state))
                x = F.relu(self.linear2(x))
            elif self.nonlin == 'tanh':
                x = torch.tanh(self.linear1(state))
                x = torch.tanh(self.linear2(x))
            mean = self.mean_linear(x)
            log_std = self.log_std_linear(x)
            if self.clamp:
                log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        elif self.model_type == 'linear':
            mean = self.linear_model_mean(state)
            log_std = self.linear_model_sd(state)
            if self.clamp:
                log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        elif self.model_type == 'rbf':
            y = torch.matmul(self.scale.to(state.device), state.t()) / self.bandwidth
            y += self.shift.to(state.device)
            y = torch.sin(y).detach()
            mean = self.linear_model_mean(y.t())
            log_std = self.linear_model_sd(y.t())
            if self.clamp:
                log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        else:
            raise Exception()

        return mean, log_std

    def sample(self, state, reparam=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        if reparam == False:
            x_t = normal.sample()
        else:
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        if self.transform_rv:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = x_t
            log_prob = normal.log_prob(x_t).sum(dim=1, keepdim=True)
        return action, log_prob, mean

    def log_prob(self, state, action):

        # get dist
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        # get log prob
        #assert not self.transform_rv
        if self.transform_rv:
            u = torch.atanh((action - self.action_bias) / self.action_scale)
            log_prob = normal.log_prob(u).sum(1, keepdim=True)
            log_prob -= torch.log(self.action_scale * (1 - u.pow(2)) + epsilon).sum(1, keepdim=True)
        else:
            log_prob = normal.log_prob(action).sum(1, keepdim=True)
        # return
        return log_prob

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        # generate random features transforms
        if self.model_type == 'rbf':
            self.scale = self.scale.to(device)
            self.shift = self.shift.to(device)
        return super(GaussianPolicy, self).to(device)
