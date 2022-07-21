import torch
import numpy as np
from copy import deepcopy
import warnings
import wandb
import gym
import os
import csv
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')
from collections import deque
import wandb
import argparse
from copy import deepcopy

from policies import Policy, ExpertPolicy

# online learning algorithms
class OnlineLearningAlgo():

    def __init__(self, env, args):
        super(OnlineLearningAlgo, self).__init__()
        self.env = env
        self.other_env = deepcopy(env)
        self.state_space = len(self.env.reset())
        self.action_space = self.env.action_space.n
        self.policy = Policy(self.env.grid_dim[0]*self.env.grid_dim[1], 5)
        self.hindsight_policy = Policy(self.env.grid_dim[0]*self.env.grid_dim[1], 5)
        self.expert = ExpertPolicy(env.expert_actions_grid)
        self.log_dir = args.log_dir
        self.csv_log = args.csv_log
        self.use_wandb = args.use_wandb
        self.episodes = args.episodes
        self.samples = args.samples_per_episode
        self.expert_steps = args.expert_steps
        assert self.expert_steps <= self.episodes
        self.cum_loss = 0.
        self.memory = deque([], maxlen=args.maxmem)
        self.lr = args.lr
        self.epochs_per_update = args.epochs_per_update

    def simulate_trajectory(self, samples, use_expert=False):
        states = torch.zeros((samples, self.state_space))
        actions = torch.zeros((samples, 1))
        expert_actions = torch.zeros((samples, 1))
        rewards = torch.zeros((samples, 1))
        dones = torch.zeros((samples, 1))
        state = self.env.reset()
        for i in range(samples):
            expert_actions[i,:] = deepcopy(self.env.get_expert_action())
            states[i,:] = torch.tensor(state)
            actions[i,:] = self.policy.sample_action(state)
            if not use_expert:
                state, reward, done, info = self.env.step(actions[i,:].detach().numpy())
            else:
                state, reward, done, info = self.env.step(expert_actions[i,:].long().detach().numpy()[0])
            rewards[i,:] = reward
        return states, actions, expert_actions, rewards, dones

    # def best_in_hindsight(self):
        # set optimization info
        optimizer = torch.optim.Adam(self.hindsight_policy.parameters(), lr=self.lr)
        # train model
        for epoch in range(self.epochs_per_update):
            # generate memory permutation
            batch_ids = np.random.permutation(len(self.memory))
            loss = - self.hindsight_policy.logprob_action(self.memory[batch_ids[0]][0], self.memory[batch_ids[0]][2]).mean()
            # iterate over the entire memory
            for i in range(1,len(batch_ids)):
                # gather examples from memory
                states_batch, _, expert_actions_batch, _, _ = self.memory[batch_ids[i]]
                # compute loss
                loss = loss - self.hindsight_policy.logprob_action(states_batch, expert_actions_batch).mean()
            # step optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #
        self.hindsight_loss = loss.detach()
        #
        return self.hindsight_loss

    def info(self, samples, use_expert=False):
        states, actions, expert_actions, rewards, dones = samples
        # print('what are we even evaluating')
        # print(states, actions)
        if not use_expert:
            loss = - self.policy.logprob_action(states, expert_actions).sum()
        else:
            loss = torch.tensor(0.)
        # print(loss)
        return loss, 1 #self.evaluate(max_samples=1000, use_expert=True)

    def train_policy(self):

        # logging
        if self.csv_log:
            try:
                os.makedirs(self.log_dir, exist_ok = True)
            except OSError as error:
                pass
            f = open(os.path.join(self.log_dir, 'results.csv'), 'w', encoding='UTF8')
            writer = csv.writer(f)
            writer.writerow(['algo',  'env', 'episodes', 'time-steps',
                             'loss',
                             # 'expert_loss',
                             'reward',
                             # 'expert_reward',
                             'cum_loss',
                             'avg_cum_loss'])
        if self.use_wandb:
            wandb.init(project="AdaptiveFTRL", group=self.env.type)

        # get sampled trajectories and push to memory
        states, actions, expert_actions, rewards, dones = self.simulate_trajectory(int(self.samples), use_expert=False)
        self.memory.append((states, actions, expert_actions, rewards, dones))
        self.last_sample = states, actions, expert_actions, rewards, dones
        reward = sum(rewards) / max(1,sum(dones))

        # # grab info for logging
        # hindsight_loss = self.best_in_hindsight()
        policy_loss = self.info(self.last_sample)[0]
        self.cum_loss = policy_loss

        # logging initial loss
        if self.csv_log:
            writer.writerow([self.algo, self.env.type, 0, 0,
                         # hindsight_loss.item(),
                         policy_loss.item(),
                         # expert_loss.item(),
                         reward,
                         # expert_reward,
                         policy_loss.item(),
                         policy_loss.item() ])
        if self.use_wandb:
             wandb.log({"algo": self.algo, #"policy": self.policy.type,
                        "env": self.env.type,
                        "epoch": 0,
                        "interactions": 0,
                        # 'hindsight_loss': hindsight_loss,
                        'loss': policy_loss.item(),
                        # 'expert_loss': expert_loss.item(),
                        # "expert_reward": expert_reward,
                        "reward": reward,
                        "cum_loss": self.cum_loss.item(),
                        "avg_cum_loss": self.cum_loss.item() })

        # train model
        for epoch in range(self.episodes):

            # take remaining environment steps with agent
            self.update(self.last_sample)

            # get sampled trajectories and push to memory
            # self.samples = self.samples * 2
            states, actions, expert_actions, rewards, dones = self.simulate_trajectory(int(self.env.args.T), use_expert=False)
            self.memory.append((states, actions, expert_actions, rewards, dones))
            self.last_sample = states, actions, expert_actions, rewards, dones
            reward = sum(rewards)
            policy_loss = self.info(self.last_sample, use_expert=False)[0]
            self.cum_loss += policy_loss
            #
            print('=================================================')
            print(['algo', 'episodes', 'time-steps', 'loss', 'reward'])
            print([self.algo,  epoch, (epoch+1)*self.samples, self.cum_loss/(epoch+1), reward])
            # logging
            if self.csv_log:
                writer.writerow([self.algo, self.env.type, epoch, (epoch+1)*self.samples,
                             policy_loss.item(),
                             # expert_loss.item(),
                             reward,
                             # expert_reward,
                             self.cum_loss, self.cum_loss/(epoch+1)])
            if self.use_wandb:
                 wandb.log({"algo": self.algo, #"policy": self.policy.type,
                            "env": self.env.type, "epoch": epoch,
                            "interactions": (epoch+1)*self.samples,
                            'loss': policy_loss.item(),  "reward": reward,
                            # 'expert_loss': expert_loss.item(),
                            # "expert_reward": expert_reward,
                            "cum_loss": self.cum_loss,
                            "avg_cum_loss": self.cum_loss/(epoch+1)})

        # display
        print('Experiment Complete')
        print(['algo', 'policy', 'env', 'episodes', 'time-steps', 'loss', 'reward'])
        print([self.algo, self.policy.type,  self.env.type, epoch, (epoch+1)*self.samples, policy_loss.item(), reward])

        # return it
        return self.policy

    def update(self, samples, use_expert=False):
        ...
        return loss

class FTRL(OnlineLearningAlgo):

    def __init__(self, env, args):
        super(FTRL,self).__init__(env, args)

        self.inner_lr = args.inner_lr
        self.outer_lr = args.outer_lr
        assert self.outer_lr > 0.
        self.algo = 'FTRL'
        self.sum_of_gradients = 0 * deepcopy(torch.nn.utils.parameters_to_vector(self.policy.parameters()).detach())
        self.grad_sum = None
        self.eta = None
        self.epsilon = args.epsilon
        self.round = 0

    def current_grad_sum(self):
        # generate memory permutation
        batch_ids = np.random.permutation(len(self.memory))
        loss = - self.policy.logprob_action(self.memory[batch_ids[0]][0], self.memory[batch_ids[0]][2]).mean()
        # iterate over the entire memory
        for i in range(1,len(batch_ids)):
            # gather examples from memory
            states_batch, _, expert_actions_batch, _, _ = self.memory[batch_ids[i]]
            # compute loss
            loss = loss - self.policy.logprob_action(states_batch, expert_actions_batch).mean()
        # compute gradient for the sum of those function evals
        self.policy.zero_grad()
        loss.backward()
        # return it
        return torch.cat([param.grad.view(-1) for param in self.policy.parameters()]).detach().data

    def linearized_term(self):
        # linearized function defined in algo
        return torch.dot(self.get_last_grad(), torch.nn.utils.parameters_to_vector(self.policy.parameters()))

    def trust_region_term(self):
        # get squared difference between current and previous parameters
        diff = (self.prev_parameter_vec-torch.nn.utils.parameters_to_vector(self.policy.parameters())).pow(2).sum()
        # scale importance by eta^-1
        return (self.outer_lr / 2) * self.round ** (1/2) * (diff)

    def get_last_grad(self):
        # generate memory permutation
        batch_ids = np.random.permutation(len(self.memory))
        ftl_loss = - self.policy.logprob_action(self.memory[batch_ids[0]][0], self.memory[batch_ids[0]][2]).mean()
        # iterate over the entire memory
        for i in range(1,len(batch_ids)):
            # gather examples from memory
            states_batch, _, expert_actions_batch, _, _ = self.memory[batch_ids[i]]
            # compute loss
            ftl_loss = ftl_loss - self.prev_policy.logprob_action(states_batch, expert_actions_batch).mean()
        # compute gradient for the sum of those function evals
        self.prev_policy.zero_grad()
        ftl_loss.backward()
        # update to new sum of gradients
        self.sum_of_gradients = torch.cat([param.grad.view(-1) for param in self.prev_policy.parameters()]).data.detach().pow(2)
        # return for use in eta update
        return self.sum_of_gradients

    def update_eta(self, new_examples):
        # compute parameterwise eta update (1/eta)
        self.eta = self.outer_lr * (self.round)**(0.5)
        # nothing to return
        return None

    def update(self, samples, use_expert=False):
        # set optimization info
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.inner_lr)
        # get sampled trajectories and push to memory
        states, actions, expert_actions, rewards, dones = samples
        # set prev parameters
        self.round += 1
        self.update_eta((states, actions, expert_actions, rewards, dones))
        self.prev_policy = deepcopy(self.policy)
        self.prev_parameter_vec = deepcopy(torch.nn.utils.parameters_to_vector(self.prev_policy.parameters()).detach())
        # train model
        for epoch in range(self.epochs_per_update):
            # generate memory permutation
            batch_ids = np.random.permutation(len(self.memory))
            ftl_loss = - self.policy.logprob_action(self.memory[batch_ids[0]][0], self.memory[batch_ids[0]][2]).mean()
            # iterate over the entire memory
            for i in range(1,len(batch_ids)):
                # gather examples from memory
                states_batch, _, expert_actions_batch, _, _ = self.memory[batch_ids[i]]
                # compute loss
                ftl_loss = ftl_loss - self.policy.logprob_action(states_batch, expert_actions_batch).mean()
            #
            ftl_loss = ftl_loss / len(batch_ids)
            # add other terms
            if self.round == 1:
                lin_loss = torch.tensor(0.)
            else:
                lin_loss = self.linearized_term() / len(batch_ids)

            #
            tr_loss = self.trust_region_term()  / len(batch_ids)
            # step optimizer
            optimizer.zero_grad()
            (tr_loss-lin_loss+ftl_loss).backward()
            optimizer.step()
 
        #
        return (tr_loss+ftl_loss+lin_loss).detach()

class FTL(OnlineLearningAlgo):

    def __init__(self, env, args):
        super(FTL,self).__init__(env, args)
        self.epochs_per_update = args.epochs_per_update
        self.algo = 'FTL'

    def update(self, samples, use_expert=False):
        # set optimization info
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        # get sampled trajectories and push to memory
        states, actions, expert_actions, rewards, dones = samples
        # train model
        for epoch in range(self.epochs_per_update):
            # generate memory permutation
            batch_ids = np.random.permutation(len(self.memory))
            loss = - self.policy.logprob_action(self.memory[batch_ids[0]][0], self.memory[batch_ids[0]][2]).mean()
            # iterate over the entire memory
            for i in range(1,len(batch_ids)):
                # gather examples from memory
                states_batch, _, expert_actions_batch, _, _ = self.memory[batch_ids[i]]
                # compute loss
                loss = loss - self.policy.logprob_action(states_batch, expert_actions_batch).mean()
            # step optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # iterate over the entire memory
        loss = - self.policy.logprob_action(self.memory[batch_ids[0]][0], self.memory[batch_ids[0]][2]).sum()
        for i in range(1,len(batch_ids)):
            states_batch, _, expert_actions_batch, _, _ = self.memory[batch_ids[i]]
            # compute loss
            loss = loss - self.policy.logprob_action(states_batch, expert_actions_batch).sum()
        #
        print(torch.cat([param.grad.view(-1) for param in self.policy.parameters()]).data.detach().pow(2).mean())

        return loss

class OGD(OnlineLearningAlgo):

    def __init__(self, env, args):
        super(OGD,self).__init__(env, args)
        self.lr = args.lr
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=self.lr)
        self.algo = 'OGD'

    def update(self, samples, use_expert=False):
        # grab examples
        states, actions, expert_actions, rewards, dones = samples
        # compute loss
        loss = - self.policy.logprob_action(states, expert_actions).mean()
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # backprop through computation graph
        (loss).backward()
        # step optimizer
        self.optimizer.step()
        #
        return loss
