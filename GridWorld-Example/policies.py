import torch
import numpy as np
from copy import deepcopy
import warnings
import wandb
warnings.filterwarnings('ignore')
import os
import csv
import torch.nn.functional as F

class Policy(torch.nn.Module):
    """
    POLICY CLASS: CONTAINS ALL USABLE FUNCTIONAL FORMS FOR THE AGENTS POLICY, ALONG WITH
    THE CORROSPONDING HELPER FUNCTIONS
    """
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear = torch.nn.Linear(state_size, action_size)
        self.linear.weight.data.fill_(0.001)
        self.linear.bias.data.fill_(1.0)
        self.softmax = torch.nn.Softmax()

    def sample_action(self, state):
        logits = self.forward(state)
        probs = logits.exp() / logits.exp().sum(dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample()

    def logprob_action(self, state, action):
        logits = self.forward(state)
        log_sum = torch.log(torch.exp(logits).sum(-1)).unsqueeze(-1)
        logprob = logits - log_sum
        onehot_action = F.one_hot(action.squeeze(-1).long(), num_classes=5)
        num = torch.gather(logprob, 1, action.long())
        return num

    def forward(self, state):
        probabilities = self.linear(torch.FloatTensor(state))
        return probabilities #self.softmax(probabilities)

class ExpertPolicy(torch.nn.Module):
    """
    POLICY CLASS: CONTAINS ALL USABLE FUNCTIONAL FORMS FOR THE AGENTS POLICY, ALONG WITH
    THE CORROSPONDING HELPER FUNCTIONS
    """
    def __init__(self, action_grid):
        super(ExpertPolicy, self).__init__()
        self.action_grid = action_grid
        self.grid_dim = self.action_grid.shape
    def sample_action(self, state):
        idx = np.nonzero(state == 1)[0][0]
        x_1 = np.floor(idx / self.grid_dim[1])
        x_2 = idx - x_1 * self.grid_dim[1]
        return self.action_grid[int(x_1),int(x_2)]
