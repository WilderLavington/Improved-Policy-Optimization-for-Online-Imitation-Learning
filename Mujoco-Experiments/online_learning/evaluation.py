
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

# oil imports
from .utils import timer, load_expert, select_optimizer
from .utils import parameters_to_vector, avg_dicts#, secant_algorithm, steffensen_algorithm
from .policies import select_policy
from .replay_memory import ReplayMemory

class AlgoEvaluator():

    #
    def __init__(self, env, args):
        super(AlgoEvaluator, self).__init__()

        # hyper-parameters / policy info
        self.args = args
        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

        # environment info
        self.num_inputs = env.observation_space.shape[0]
        self.action_space = env.action_space

        # set expert policy type and load expert parameters
        self.expert = load_expert(self.num_inputs, self.action_space, args, self.device)
        self.expert.to(self.device)

        # set inclass policy type
        self.inclass_policy = select_policy(self.num_inputs, self.action_space.shape[0], args.hidden_size, \
                action_space=self.action_space, model_type=args.model_type, bandwidth=args.bandwidth, \
                transform_rv=args.transform_dist, nonlin=args.nonlin, clamp=args.clamp, \
                static_cov=args.static_cov).to(self.device)
        self.inclass_policy.to(self.device)

        # set expert policy type and load expert parameters
        self.optimal_policy = load_expert(self.num_inputs, self.action_space, args, self.device)
        self.optimal_policy.to(self.device)

        # using FTL to compute best in hindsight
        self.max_epochs_per_eval = 1000
        self.lr = 10**(-3.)
        self.loss_type = args.loss_type
        self.early_stop_crit = 1e-8
        self.inner_policy_optim = 'Adam'
        self.interactions = 0
        self.updates = 0
        self.inclass_loss = 0.
        self.optimal_loss = 0.

        # set optimization info
        self.args.lr = self.lr
        self.inclass_optimizer = select_optimizer(self.inclass_policy, self.args, 'Adam')
        self.optimal_optimizer = select_optimizer(self.optimal_policy, self.args, 'Adam')

    # different losses for use in model update
    def compute_loss(self, policy, state_batch, expert_action_batch, use_expert=False):
        if self.args.loss_type == 'l2':
            if use_expert:
                _, lp, policy_action = self.expert.sample(state_batch, reparam=False)
            else:
                _, lp, policy_action = policy.sample(state_batch, reparam=False)
            return (policy_action - expert_action_batch.detach()).pow(2).sum() + 0. * lp.sum()
        elif self.args.loss_type == 'l1':
            if use_expert:
                _, lp, policy_action = self.expert.sample(state_batch, reparam=False)
            else:
                _, lp, policy_action = policy.sample(state_batch, reparam=False)
            return (policy_action - expert_action_batch.detach()).abs().sum() + 0. * lp.sum()
        elif self.args.loss_type == 'bc':
            if use_expert:
                policy_action, _, _ = self.expert.sample(state_batch, reparam=True)
            else:
                policy_action, _, _ = policy.sample(state_batch, reparam=True)
            return (policy_action - expert_action_batch.detach()).pow(2).sum()
        elif self.args.loss_type == 'mle':
            if use_expert:
                log_liklihood = self.expert.log_prob(state_batch, expert_action_batch)
            else:
                log_liklihood = policy.log_prob(state_batch, expert_action_batch)
            return -1 * log_liklihood.sum()
        else:
            raise Exception()

    def compute_grad(self, policy):
        # compute grad norm for posterity
        grad = torch.cat([p.grad.view(-1) if p.grad is not None\
                               else p.new_zeros(p.size()).view(-1)\
                               for p in policy.parameters()]).data
        return grad

    # just use iterative steps
    def iterative_step(self, policy, optim, tensor_states, tensor_expert_actions):

        #
        def closure(verbose=False, call_backward=False):
            # zero the parameter gradients
            optim.zero_grad()
            # grab ftl loss
            loss = self.compute_loss(policy, tensor_states, tensor_expert_actions) / self.replay_size
            # backprop through computation graph
            if call_backward:
                loss.backward()
            # do we want other info
            return loss

        # step optimizer (do we need a closure?)
        if self.inner_policy_optim == 'LSOpt':
            # direct step
            loss = optim.step(closure)
        else:
            # backprop through computation graph
            optim.step()

        # compute grad norm for posterity
        loss = closure(call_backward=True)
        grad_norm = self.compute_grad(policy).detach().pow(2).mean()

        # return
        return loss.detach(), grad_norm.detach()

    # something to update params
    def update_parameters(self, new_examples):

        # get sampled trajectories and push to memory
        self.memory, _, self.avg_return = new_examples
        self.replay_size = self.memory.__len__()
        self.interactions += self.args.samples_per_update

        # gather examples from memory
        states, expert_actions, _, _ = map(np.stack, zip(*self.memory.buffer))
        tensor_states = torch.stack([torch.tensor(s) for s in states]).float().detach()
        tensor_expert_actions = torch.stack([torch.tensor(s) for s in expert_actions]).float().detach()

        # move to device
        tensor_states = tensor_states.to(self.device)
        tensor_expert_actions = tensor_expert_actions.to(self.device)

        # otherwise run iterative
        for epoch in range(self.max_epochs_per_eval):
            loss, grad_norm = self.iterative_step(self.inclass_policy, self.inclass_optimizer, tensor_states, tensor_expert_actions)
            loss, grad_norm = self.iterative_step(self.optimal_policy, self.optimal_optimizer, tensor_states, tensor_expert_actions)
            self.updates += 1
            # early stopping crit
            if grad_norm.item() < self.early_stop_crit:
                break

        # return loss
        return loss

    # get return
    def evaluate_return(self, policy, env, duplicates=10):
        avg_reward = 0.
        policy = policy.to(self.device)
        for _  in range(duplicates):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                tensor_state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                tensor_action = policy.sample(tensor_state)[2]
                action = tensor_action.detach().cpu().numpy()[0]
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
            avg_reward += episode_reward
        avg_reward /= duplicates
        return avg_reward

    #
    def evaluate(self, env, memory, new_examples):
        # set memory
        self.memory = memory
        # update model parameters
        self.update_parameters(new_examples)
        # evaluate performance
        with torch.no_grad():
            self.memory, (new_states, new_expert_actions, _, _), self.avg_return = new_examples
            self.inclass_policy = self.inclass_policy.to('cpu')
            self.optimal_policy = self.optimal_policy.to('cpu')
            new_states = new_states.to('cpu')
            new_expert_actions = new_expert_actions.to('cpu')
            self.inclass_loss += self.compute_loss(self.inclass_policy, new_states, new_expert_actions)
            self.optimal_loss += self.compute_loss(self.optimal_policy, new_states, new_expert_actions)
            self.inclass_return = self.evaluate_return(self.inclass_policy, env)
            self.optimal_return = self.evaluate_return(self.optimal_policy, env)
        # nothing to return
        return None
