
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
import gym
#
from .utils import timer, select_optimizer
from .utils import parameters_to_vector, avg_dicts#, secant_algorithm, steffensen_algorithm
from .policies import select_policy
from .replay_memory import ReplayMemory
from .load_expert import pretrain_args, get_expert

class AlgoEvaluator():

    #
    def __init__(self, env, args):
        super(AlgoEvaluator, self).__init__()

        # hyper-parameters / policy info
        self.args = args
        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

        # environment info
        self.expert = get_expert(self.args)
        self.env = self.expert.get_env()
        self.avg_return, self.regret = None, 0.
        self.current_state = self.env.reset()
        self.memory = ReplayMemory(args.replay_size, args.seed, args)
        # set expert policy type and load expert parameters
        self.features_extractor = torch.nn.Sequential(\
                deepcopy(self.expert.model.policy.features_extractor),
                deepcopy(self.expert.model.policy.mlp_extractor))

        # grab the wrapper
        obs = self.env.reset()
        self.obs_size = obs.shape[1:]
        obs = self.expert.model.policy.obs_to_tensor(obs)[0].float()
        self.num_inputs = self.features_extractor(obs)[0].shape[-1]
        self.num_outputs = self.env.action_space.n

        # set inclass policy type
        self.inclass_policy = select_policy(self.num_inputs, self.num_outputs,
                self.features_extractor, args.hidden_size, \
                model_type=args.model_type, bandwidth=args.bandwidth, \
                nonlin=args.nonlin).to(self.device)
        self.inclass_policy.device = self.device
        self.inclass_policy.state_transform = lambda obs: self.expert.model.policy.obs_to_tensor(obs)[0].float()

        # set inclass policy type
        self.optimal_policy = select_policy(self.num_inputs, self.num_outputs,
                self.features_extractor, args.hidden_size, \
                model_type='end2end', bandwidth=args.bandwidth, \
                nonlin=args.nonlin).to(self.device)
        self.optimal_policy.device = self.device
        self.optimal_policy.state_transform = lambda obs: self.expert.model.policy.obs_to_tensor(obs)[0].float()

        # control mixing
        self.beta = self.args.beta
        self.beta_update = self.args.beta_update
        self.stochastic_interaction = self.args.stochastic_interaction
        self.episode = 1

        # logging info
        self.updates, self.interactions = 0, 0
        self.policy_return, self.policy_loss = None, None
        self.start = time.time()

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
        self.inclass_optimizer = select_optimizer(self.inclass_policy, self.args, self.inner_policy_optim)
        self.optimal_optimizer = select_optimizer(self.optimal_policy, self.args, self.inner_policy_optim)

    # different losses for use in model update
    def compute_loss(self, policy, state_batch, expert_action_batch, use_expert=False):
        state_batch = state_batch.reshape(state_batch.shape[0],*self.obs_size).cpu().numpy()
        expert_action_batch = expert_action_batch.cpu().numpy()
        if self.args.loss_type == 'mle':
            if use_expert:
                log_liklihood = self.expert.log_prob(state_batch, expert_action_batch.reshape(-1))
            else:
                log_liklihood = policy.log_prob(state_batch, expert_action_batch.reshape(-1))
            return -1 * log_liklihood.sum()
        else:
            raise Exception()


    def compute_grad(self, policy):
        # compute grad norm for posterity
        grad = torch.cat([p.grad.view(-1) if p.grad is not None\
                               else p.new_zeros(p.size()).view(-1)\
                               for p in policy.parameters()]).data
        return grad

    def gd_step(self, policy, optim, tensor_states, tensor_expert_actions):
        # create closure
        def closure(verbose=False, call_backward=False,
                s=tensor_states, a=tensor_expert_actions):
            # zero the parameter gradients
            optim.zero_grad()
            # grab ftl loss
            s = s.reshape(-1, *self.obs_size)
            a = a.reshape(-1, 1)
            loss = self.compute_loss(policy, s, a) / self.replay_size
            # backprop through computation graph
            if call_backward:
                loss.backward()
            # do we want other info
            return loss
        # step optimizer (do we need a closure?)
        if self.args.inner_policy_optim == 'LSOpt':
            # direct step
            loss = optim.step(closure)
        else:
            # backprop through computation graph
            optim.zero_grad()
            loss = closure(call_backward=True)
            optim.step()
        # compute grad norm for posterity
        loss = closure(call_backward=True)
        grad_norm = self.compute_grad(policy).detach().pow(2).mean()
        # return
        return loss.detach(), grad_norm.detach()

    # just use iterative steps
    def iterative_step(self, policy, optim, tensor_states, tensor_expert_actions):

        # iterate over the entire memory
        for states_batch, expert_actions_batch in self.data_generator:
            # take gradient over data subset
            loss, grad_norm = self.gd_step(policy, optim, states_batch, expert_actions_batch)

        # return
        return loss.detach(), grad_norm.detach()

    # something to update params
    def update_parameters(self, new_examples):

        # get sampled trajectories and push to memory
        self.memory, _, self.avg_return = new_examples
        self.replay_size = self.memory.__len__()
        self.interactions += self.args.samples_per_update

        # gather examples from current los
        states, expert_actions, _, _ = map(np.stack, zip(*self.memory.buffer))
        tensor_states = torch.tensor(states).float().detach()
        tensor_expert_actions = torch.tensor(expert_actions).float().detach()

        # move to device
        tensor_states = tensor_states.to(self.device).reshape(-1,*self.obs_size)
        tensor_expert_actions = tensor_expert_actions.to(self.device)
        self.inclass_policy.to(self.device)
        self.optimal_policy.to(self.device)
        # make a generater
        dataset = torch.utils.data.TensorDataset(tensor_states, tensor_expert_actions)
        self.data_generator = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
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

    # evaluation
    def evaluate_return(self, policy, env,  duplicates=3, use_random=False ):
        avg_reward = 0.
        assert not (use_random and use_expert)
        for _  in range(duplicates):
            state = env.reset()
            episode_reward = 0
            t = 0
            reset_flag = False
            while (not reset_flag):
                action = policy.sample(state)[2]
                if use_random:
                    action = [env.action_space.sample()]
                next_state, reward, done, infos = env.step(action)
                state = next_state
                episode_reward += reward
                t += 1
                if infos is not None:
                    episode_infos = infos[0].get("episode")
                    if episode_infos is not None:
                        # print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                        # print("Atari Episode Length", episode_infos["l"])
                        episode_reward = episode_infos['r']
                        reset_flag = True
                        avg_reward += episode_reward
            # print(t, episode_reward)
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
            self.inclass_policy.to('cpu')
            self.optimal_policy.to('cpu') 
            self.inclass_loss += self.compute_loss(self.inclass_policy, new_states.to('cpu'), new_expert_actions.to('cpu'))
            self.optimal_loss += self.compute_loss(self.optimal_policy, new_states.to('cpu'), new_expert_actions.to('cpu'))
            self.inclass_return = self.evaluate_return(self.inclass_policy, env)
            self.optimal_return = self.evaluate_return(self.optimal_policy, env)
        # nothing to return
        return None
