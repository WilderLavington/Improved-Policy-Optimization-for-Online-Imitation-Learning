# general imports
import os
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from copy import deepcopy
import wandb
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torch.distributions import Normal
import datetime
import gym
import itertools

# my imports
from .algorithms import OnlineLearningAlgo
from .utils import select_optimizer
from online_learning.parser import get_args

class BC(OnlineLearningAlgo):

    def __init__(self, env, args):
        super(BC,self).__init__(env, args)
        self.lr, args.lr = 10**args.log_lr, 10**args.log_lr
        self.algo = 'BC'
        self.loss_type = args.loss_type
        self.optimizer = select_optimizer(self.policy, args, args.policy_optim) #torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.samples = args.samples_per_update
        self.replay_size = self.args.samples_per_update
        self.interactions = self.args.samples_per_update
        # grab examples
        _, (self.states, self.expert_actions, _, _) = self.gather_examples(None, self.samples, use_expert=True)
        # move them to the device
        self.states = self.states.to(self.device)
        self.expert_actions = self.expert_actions.to(self.device)
        self.lambda_reg = 10**self.args.log_lambda
        # generate memory permutation
        if self.args.use_sgd:
            dataset = torch.utils.data.TensorDataset(self.states, self.expert_actions)
            self.data_generator = torch.utils.data.DataLoader(dataset, batch_size=self.args.mini_batch_size, shuffle=True)

    def update_parameters(self, use_expert=True):
        # pick the update check
        if self.args.use_exact:
            assert self.args.episodes == 1
            loss = self.exact_solve()
        elif not self.args.use_sgd:
            loss = self.gradient_decent()
        else:
            loss = self.stochastic_gradient_decent()
        # return
        return loss.detach().item()

    def stochastic_gradient_decent(self):
        # iterate over the entire memory
        for states_batch, expert_actions_batch in self.data_generator:
            # compute loss
            loss = self.compute_loss(states_batch, expert_actions_batch)  / self.args.mini_batch_size
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # backprop through computation graph
            loss.backward()
            grad_norm = torch.cat([param.grad.view(-1) for param in self.policy.parameters()]).data.detach().pow(2).mean()
            # step optimizer
            self.optimizer.step()
        # store
        self.info = {'sgd_loss':  self.compute_loss(self.states, self.expert_actions),
                     'grad_norm': grad_norm}
        # return computed loss
        return loss

    def gradient_decent(self):
        # compute loss
        loss = self.compute_loss(self.states, self.expert_actions) # / self.replay_size
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # backprop through computation graph
        loss.backward()
        grad_norm = torch.cat([param.grad.view(-1) for param in self.policy.parameters()]).data.detach().pow(2).mean()
        # step optimizer
        self.optimizer.step()
        self.updates += 1
        # store
        self.info = {'gd_loss':  self.compute_loss(self.states, self.expert_actions),
                     'grad_norm': grad_norm}
        # return computed loss
        return loss

    def exact_solve(self):
        # make sure we are in the correct regime
        assert args.model_type != 'NNPolicy'
        assert args.static_cov == 1
        assert args.loss_type == 'l2'
        # transform state
        state = self.policy.transform_state(self.states).to('cpu').double()
        state = torch.cat((state,torch.ones(state.size()[0], 1).double()), dim=1)
        # compute ls solution
        proj = torch.inverse(torch.mm(state.t(),state) + self.lambda_reg * torch.eye(state.size()[-1]).double())
        w = torch.mm(proj, torch.mm(state.t(),self.expert_actions.to('cpu').double()))
        weight, bias = w[:-1,:], w[-1,:]
        # set the weights
        with torch.no_grad():
            self.policy.mean_linear.weight.data = weight.t().float().to(self.states.device)
            self.policy.mean_linear.bias.data = bias.float().to(self.states.device)
        # compute loss check
        lse_loss = (torch.mm(state[:,:-1],weight).to(self.states.device) + bias.to(self.states.device) - self.expert_actions).pow(2).sum()
        network_loss = (self.policy(self.states)[0] - self.expert_actions).pow(2).sum()
        lib_loss = self.compute_loss(self.states, self.expert_actions)
        grad = torch.mm(torch.mm(state.t(),state),w) + self.lambda_reg * w - torch.mm(state.t(), self.expert_actions.to('cpu').double())
        print(grad.pow(2).mean().item(), lse_loss.item(), network_loss.item(), lib_loss.item())
        # store
        self.info = {'rlse_loss': network_loss,
                     'grad_norm': grad.pow(2).sum()}
        # return computed loss
        return network_loss

def train_bc_agent():
    # get args
    args = get_args()
    # check world model will hold (for vizualiztion)
    assert args.env_name in ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    # Environment
    env = gym.make(args.env_name)
    # mae a copy
    args.env_copy = deepcopy(env)
    # presets
    args.expert_params_path = 'expert_models/sac_actor_'+args.env_name.replace('-','_')+'_expert'
    # Agent
    trainer = BC(env, args)
    # init project
    wandb.init(project='BCTests', group=args.env_name)
    # train model
    trainer.train_agent()

if __name__ == "__main__":
    train_bc_agent()
