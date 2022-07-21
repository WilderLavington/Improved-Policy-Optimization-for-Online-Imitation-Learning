import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
import numpy as np
from copy import deepcopy
import  pyrfm

LOG_SIG_MAX = 2
LOG_SIG_MIN = -16
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def select_policy(num_inputs, num_actions, hidden_dim, action_space=None,
                  model_type='nn', bandwidth=0., transform_rv=True, nonlin='relu',
                  clamp=True, static_cov=0):
    # assert action_space is None
    if model_type=='nn':
        return NNPolicy(num_inputs, num_actions, transform_rv, clamp, static_cov, hidden_dim, nonlin)
    elif model_type=='rff':
        return RFFPolicy(num_inputs, num_actions, transform_rv, clamp, static_cov, hidden_dim, bandwidth)
    elif model_type=='linear':
        return LinearPolicy(num_inputs, num_actions, transform_rv, clamp, static_cov)
    else:
        raise Exception

class Policy(nn.Module):

    def __init__(self, num_inputs, num_actions, transform_rv, clamp, static_cov):
        super(Policy, self).__init__()

        # something to compute it all
        self.transform_rv = transform_rv
        self.clamp = clamp
        self.static_cov = static_cov

        # compute target means and sd
        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

        # something for stability
        self.epsilon = 1e-20
        self.force_log_std = static_cov

    def sample(self, state, reparam=True):

        # grab info
        mean, log_std = self.forward(state)

        # force requirements on std
        if self.clamp:
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        if self.force_log_std:
            log_std = -25 * torch.ones(log_std.detach().size()).to(log_std.device) + 0.*log_std

        # compute distrbution
        normal = Normal(mean, log_std.exp())

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
        std = log_std.exp() + self.epsilon
        normal = Normal(mean, std)

        # get log prob
        if self.transform_rv:
            u = torch.atanh((action - self.action_bias) / self.action_scale)
            log_prob = normal.log_prob(u).sum(1, keepdim=True)
            log_prob -= torch.log(self.action_scale * (1 - u.pow(2)) + epsilon).sum(1, keepdim=True)
        else:
            log_prob = normal.log_prob(action).sum(1, keepdim=True)

        if self.clamp:
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        if self.force_log_std:
            log_std = -6 * torch.ones(log_std.size()).to(log_std.device) + 0.*log_std.detach()

        # return
        return log_prob

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Policy, self).to(device)

    def forward(self, state):
        raise Expectation('')

class LinearPolicy(Policy):
    def __init__(self, num_inputs, num_actions, transform_rv, clamp, static_cov):
        super(LinearPolicy,self).__init__(num_inputs, num_actions, transform_rv, clamp, static_cov)

        #
        self.mean_linear = nn.Linear(num_inputs, num_actions)
        self.log_std_linear = nn.Linear(num_inputs, num_actions)
        self.model_type = 'LinearPolicy'

        # force to mean zero
        self.mean_linear.weight.data.mul_(0.0)
        self.mean_linear.bias.data.mul_(0.0)
        self.log_std_linear.weight.data.mul_(0.0)
        self.log_std_linear.bias.data.mul_(0.0)

    # function required for closed form soln
    def transform_state(self, state):
        return state

    # define forward
    def forward(self, state):
        mean = self.mean_linear(state)
        log_std = self.log_std_linear(state)
        return mean, log_std

class RFFPolicy(Policy):
    def __init__(self, num_inputs, num_actions, transform_rv, clamp, static_cov, hidden_dim, bandwidth, kernel_type='rbf'):
        super(RFFPolicy,self).__init__(num_inputs, num_actions, transform_rv, clamp, static_cov)

        #
        self.hidden_dim = hidden_dim
        self.state_size = num_inputs
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.bandwidth = torch.exp(torch.tensor(bandwidth))
        self.model_type = 'RFFPolicy'

        # force to mean zero
        self.mean_linear.weight.data.mul_(0.0)
        self.mean_linear.bias.data.mul_(0.0)
        self.log_std_linear.weight.data.mul_(0.0)
        self.log_std_linear.bias.data.mul_(0.0)

        # generate random features
        self.transform_map = pyrfm.random_feature.RandomFourier(n_components=hidden_dim, kernel='rbf', gamma=self.bandwidth)
        self.transform_map.fit((np.random.rand(1,num_inputs)))

        # self.scale = torch.randn((self.hidden_dim, self.state_size))
        # self.shift = torch.FloatTensor(self.hidden_dim, 1).uniform_(-np.pi, np.pi)

    # required for closed form solution
    def transform_state(self, state):
        # y = torch.matmul(self.scale.to(state.device), state.t()) / self.bandwidth
        # y += self.shift.to(state.device)
        # y = torch.sin(y).detach()
        # return y.t()
        return torch.FloatTensor(self.transform_map.transform(state.to('cpu').numpy())).to(state.device)

    # define forward
    def forward(self, state):
        y = self.transform_state(state)
        mean = self.mean_linear(y)
        log_std = self.log_std_linear(y)
        return mean, log_std

    # make sure to push other random data in
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device) 
        return super(RFFPolicy, self).to(device)

class NNPolicy(Policy):
    def __init__(self, num_inputs, num_actions, transform_rv, clamp, static_cov, hidden_dim, nonlin):
        super(NNPolicy,self).__init__(num_inputs, num_actions, transform_rv, clamp, static_cov)

        #
        self.nonlin = nonlin
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.model_type = 'NNPolicy'

        # force to mean zero
        self.mean_linear.weight.data.mul_(0.0)
        self.mean_linear.bias.data.mul_(0.0)
        self.log_std_linear.weight.data.mul_(0.0)
        self.log_std_linear.bias.data.mul_(0.0)

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

    # define forward
    def forward(self, state):
        if self.nonlin == 'relu':
            x = self.relu1(self.linear1(state))
            x = self.relu2(self.linear2(x))
        elif self.nonlin == 'tanh':
            x = torch.tanh(self.linear1(state))
            x = torch.tanh(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        return mean, log_std

class SACPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None,
                    model_type='nn', bandwidth=0., transform_rv=True, nonlin='relu',
                    clamp=True, static_cov=0):
        super(SACPolicy, self).__init__()
        self.model_type = model_type
        self.transform_rv = transform_rv
        self.clamp = clamp
        self.static_cov = static_cov
        # nn model
        if self.model_type == 'nn':
            self.nonlin = nonlin
            self.linear1 = nn.Linear(num_inputs, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.mean_linear = nn.Linear(hidden_dim, num_actions)
            self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        #  rbf model
        elif self.model_type == 'rbf':
            self.hidden_dim = hidden_dim
            self.state_size = num_inputs
            self.mean_linear = nn.Linear(hidden_dim, num_actions)
            self.log_std_linear = nn.Linear(hidden_dim, num_actions)
            self.bandwidth = torch.exp(torch.tensor(bandwidth))
        # linear
        elif self.model_type == 'linear':
            self.mean_linear = nn.Linear(num_inputs, num_actions)
            self.log_std_linear = nn.Linear(num_inputs, num_actions)
        else:
            raise Exception()

        #
        self.mean_linear.weight.data.mul_(0.0)
        self.mean_linear.bias.data.mul_(0.0)
        self.log_std_linear.weight.data.mul_(0.0)
        self.log_std_linear.bias.data.mul_(0.0)

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
            mean = self.mean_linear(state)
            log_std = self.log_std_linear(state)
            if self.clamp:
                log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        elif self.model_type == 'rbf':
            y = torch.matmul(self.scale.to(state.device), state.t()) / self.bandwidth
            y += self.shift.to(state.device)
            y = torch.sin(y).detach()
            mean = self.mean_linear(y.t())
            log_std = self.log_std_linear(y.t())
            if self.clamp:
                log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        else:
            raise Exception()
        if self.static_cov:
            log_std = -8 * torch.ones(log_std.size()).to(log_std.device) + 0.*log_std

        return mean, log_std

    def sample(self, state, reparam=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std+1e-20)

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
        return super(SACPolicy, self).to(device)

class PPOPolicy(nn.Module):
    def __init__(self, env_name):
        super(PPOPolicy, self).__init__()
        # load models
        env = gym.make(env_name)
        model = PPO("MlpPolicy", env, verbose=0)
        model.load('./expert_models/ppo_actorcritic_'+env_name.replace('-','_')+'.pt')
    def forward(self, state):
        raise Exception()

    def sample(self, state, reparam=False):
        return self.SB3_model.predict(state, deterministic=False)[0], None, self.SB3_model.predict(state, deterministic=True)[0]

    def log_prob(self, state, action):

        raise Exception()
