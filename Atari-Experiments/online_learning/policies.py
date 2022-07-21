import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
import numpy as np
from copy import deepcopy
import  pyrfm
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -16
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)

def select_policy(num_inputs, num_actions, feature_transformer,
                  hidden_dim=512, model_type='nn', bandwidth=0., nonlin='relu'):
    # assert action_space is None
    if model_type=='nn':
        return NNPolicy(num_inputs, num_actions, feature_transformer, False, hidden_dim, nonlin)
    elif model_type=='rff':
        return RFFPolicy(num_inputs, num_actions, feature_transformer, False, hidden_dim, bandwidth)
    elif model_type=='linear':
        return LinearPolicy(num_inputs, num_actions, feature_transformer, False)
    elif model_type == 'end2end':
        return End2EndPolicy(num_inputs, num_actions, feature_transformer, True)
    else:
        raise Exception

class Policy(nn.Module):

    def __init__(self, num_inputs, num_actions, feature_transformer, learn_fe):
        super(Policy, self).__init__()
        # set info
        self.num_inputs = num_actions
        self.num_actions = num_actions
        self.feature_transformer = feature_transformer
        self.ft_out = 512
        # shuffle weights around
        if learn_fe:
            self.feature_transformer.apply(weights_init_)

    def sample(self, state):
        logits = self.forward(state)
        logits = torch.clamp(logits, min=-20,max=2)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return dist.sample(), dist.log_prob(action), torch.argmax(logits,dim=-1)

    def log_prob(self, state, action):
        if not torch.is_tensor(state):
            state = torch.tensor(state).to(self.device)
            action = torch.tensor(action).to(self.device)
        logits = self.forward(state)
        logits = torch.clamp(logits, min=-20,max=20)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action)

    def to(self, device): 
        self.feature_transformer = self.feature_transformer.to(device)
        return super(Policy, self).to(device)

    def forward(self, state):
        raise Expectation('return logits')


class LinearPolicy(Policy):
    def __init__(self, num_inputs, num_actions, feature_transformer, learn_fe):
        super(LinearPolicy,self).__init__(num_inputs, num_actions, feature_transformer, learn_fe)
        self.output_linear = nn.Linear(self.ft_out, num_actions)
        self.model_type = 'LinearPolicy'
        self.learn_ft = False
        self.output_linear.weight.data.mul_(0.0)
        self.output_linear.bias.data.mul_(0.0)
    def transform_state(self, state):
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        state = self.state_transform(state)
        return self.feature_transformer(state)[0].detach()
    def forward(self, state):
        state = self.transform_state(state)
        logits = self.output_linear(state)
        return logits

class RFFPolicy(Policy):
    def __init__(self, num_inputs, num_actions, feature_transformer, learn_fe, hidden_dim, bandwidth):
        super(RFFPolicy,self).__init__(num_inputs, num_actions, feature_transformer, learn_fe)
        self.hidden_dim = hidden_dim
        self.state_size = self.ft_out
        self.output_linear = nn.Linear(hidden_dim, num_actions)
        self.output_linear.weight.data.mul_(0.0)
        self.output_linear.bias.data.mul_(0.0)
        self.bandwidth = torch.exp(torch.tensor(bandwidth))
        self.model_type = 'RFFPolicy'
        self.learn_ft = False
        self.transform_map = pyrfm.random_feature.\
            RandomFourier(n_components=hidden_dim, kernel='rbf', gamma=self.bandwidth)
        self.transform_map.fit((np.random.rand(1,self.ft_out)))
    def transform_state(self, state):
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        state = self.state_transform(state)
        state = self.feature_transformer(state)[0].detach()
        return torch.FloatTensor(self.transform_map.\
            transform(state.to('cpu').numpy())).to(state.device)
    def forward(self, state):
        y = self.transform_state(state)
        logits = self.output_linear(y)
        return logits

class NNPolicy(Policy):
    def __init__(self, num_inputs, num_actions, feature_transformer, learn_fe, hidden_dim, nonlin):
        super(NNPolicy,self).__init__(num_inputs, num_actions, feature_transformer, learn_fe)
        self.nonlin = nonlin
        self.linear1 = nn.Linear(self.ft_out, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, num_actions)
        self.output_linear.weight.data.mul_(0.0)
        self.output_linear.bias.data.mul_(0.0)
        self.model_type = 'NNPolicy'
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.learn_ft = False
    def transform_state(self, state):
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        state = self.state_transform(state)
        return self.feature_transformer(state)[0].detach()
    def forward(self, state):
        state = self.transform_state(state)
        if self.nonlin == 'relu':
            x = self.relu1(self.linear1(state))
            x = self.relu2(self.linear2(x))
        elif self.nonlin == 'tanh':
            x = torch.tanh(self.linear1(state))
            x = torch.tanh(self.linear2(x))
        logit = self.output_linear(x)
        return logit

class End2EndPolicy(Policy):
    def __init__(self, num_inputs, num_actions, feature_transformer, learn_fe):
        super(End2EndPolicy,self).__init__(num_inputs, num_actions, feature_transformer, learn_fe)
        self.output_linear = nn.Linear(self.ft_out, num_actions)
        self.output_linear.weight.data.mul_(0.0)
        self.output_linear.bias.data.mul_(0.0)
        self.learn_ft = True
    def transform_state(self, state):
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        state = self.state_transform(state)
        return self.feature_transformer(state)[0]
    def forward(self, state):
        state = self.transform_state(state)
        logit = self.output_linear(state)
        return logit
