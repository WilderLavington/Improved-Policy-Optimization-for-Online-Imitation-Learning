
import torch
import numpy as np
from copy import deepcopy
import warnings
import wandb
warnings.filterwarnings('ignore')
import os
import csv
import gym

class GridWorld():
    """
    GRID WORLD CLASS: HELPER CLASS TO INSTANTIATE A SMALL MDP PROBLEM TO TEST POLICY GRADIENT
    METHODS ON TO ENSURE CORRECTNESS, ALSO ALOWS FOR FOR TESTING OF NEW TYPES OF ENVIROMENTS
    WITH DIFFERENT STATESPACE DISTRIBUTIONS ECT.
    """
    def __init__(self, args):
        super(GridWorld, self).__init__()
        self.grid = None
        self.start_state = [args.start_state_x, args.start_state_y]
        self.state = self.start_state
        self.grid_dim = [args.grid_dim_x, args.grid_dim_y]
        self.action_noise = args.action_noise
        self.T = args.T
        self.args = args
        assert (self.action_noise >= 0.) and (self.action_noise <= 1.)
        self.action_space = gym.spaces.Discrete(5)

    def reset(self):
        ...

    def step(self, action):
        ...

    def env_step(self, action):
        # state = [row, column]
        if action == 1: # up
            # if we are in top row no change
            if self.state[0] == 0:
                None
            # otherwise increase row position
            else:
                self.state[0] -= 1
        elif action == 2: # down
            if self.state[0] == self.grid_dim[0]-1:
                None
            # otherwise increase row position
            else:
                self.state[0] += 1
        elif action == 3: # right
            if self.state[1] == self.grid_dim[1]-1:
                None
            # otherwise increase row position
            else:
                self.state[1] += 1
        elif action == 4: # left
            if self.state[1] == 0:
                None
            # otherwise increase row position
            else:
                self.state[1] -= 1
        elif action == 0: # dont_move
            None
        else:
            print("key error for movement in env_step")
        self.one_hot_state = self.one_hot(self.state)
        return self.state

    def one_hot(self, state):
        array_state = np.zeros(self.grid_dim[0]*self.grid_dim[1])
        array_state[int(state[0]*(self.grid_dim[1]) + state[1])] = 1
        return array_state

    def invert_one_hot(self, state):
        idx = np.nonzero(state == 1)[0][0]
        x_1 = np.floor(idx / self.grid_dim[1])
        x_2 = idx - x_1 * self.grid_dim[1]
        return [int(x_1),int(x_2)]

    def display_position(self, state):
        x = np.zeros(self.grid_dim)
        x[state[0],state[1]] = 1
        print(x)
        print(self.expert_actions_grid)

class RandomActionGridWorld(GridWorld):
    """
    ...
    """
    def __init__(self, args):
        super(RandomActionGridWorld, self).__init__(args)
        # random initial state
        self.start_state = [np.random.randint(0,self.grid_dim[0]),np.random.randint(0,self.grid_dim[1])]
        self.state = self.start_state
        self.one_hot_state = self.one_hot(self.state)
        # random set of expert actions
        self.expert_actions_grid = np.random.randint(5, size=self.grid_dim)
        self.type = 'RandomActionWorld'
        self.best_loss = 0.0 * self.args.T * self.args.samples_per_episode

    def reset(self):
        # random initial state
        self.start_state = [np.random.randint(0,self.grid_dim[0]),np.random.randint(0,self.grid_dim[1])]
        self.state = self.start_state
        # reset time
        self.t = 1
        # return 1-hot encoded state
        return self.one_hot(self.start_state)

    def step(self, action):
        # add noise to action?
        if self.action_noise >= np.random.rand():
            action = np.random.randint(0,5)
        # update time
        self.t += 1
        # check end conditions
        done = 1 if self.t > self.T else 0
        # get reward
        reward = 1 if action == self.expert_actions_grid[self.state[0],self.state[1]] else 0
        # grab state
        state = self.env_step(action)
        # return
        return self.one_hot(state), reward, done, {}

class StairCaseGridWorld(GridWorld):
    """
    ...
    """
    def __init__(self, args):
        super(StairCaseGridWorld, self).__init__(args)
        self.expert_actions_grid = self.staircase_expert(self.grid_dim)
        self.norm = self.grid_dim[0] * self.grid_dim[1]
        self.type = 'StairCaseGridWorld'
        self.best_loss = 0.0 * self.args.T * self.args.samples_per_episode

    def reset(self):
        # random initial state
        self.start_state = [np.random.randint(0,self.grid_dim[0]),np.random.randint(0,self.grid_dim[1])]
        self.state = self.start_state
        self.one_hot_state = self.one_hot(self.state)
        # reset time
        self.t = 1
        # return 1-hot encoded state
        return self.one_hot(self.start_state)

    def step(self, action):
        # add noise to action?
        if self.action_noise >= np.random.rand():
            action = np.random.randint(0,5)
        # update time
        self.t += 1
        # check end conditions
        done = 1 if self.t > self.T else 0
        # get reward
        reward = 1 if action == self.expert_actions_grid[self.state[0],self.state[1]] else 0
        # grab state
        state = self.env_step(action)
        # return
        return self.one_hot(state), reward, done, {}

    def staircase_expert(self, grid_dim):
        grid = np.random.randint(5, size=grid_dim)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if i < j:
                    grid[i,j] = 1
                elif i == j:
                    grid[i,j] = 3
                elif i > j:
                    grid[i,j] = 2
        return grid

class AdversarialGridWorld_1(GridWorld):
    """
    ...
    """
    def __init__(self, args):
        super(AdversarialGridWorld_1, self).__init__(args)
        # random initial state
        self.start_state = [self.grid_dim[0]-1, int(self.grid_dim[1]/2)]
        self.state = [self.grid_dim[0]-1, int(self.grid_dim[1]/2)]
        self.one_hot_state = self.one_hot(self.state)
        # random set of expert actions
        self.expert_actions_grid = self.imperfect_expert_grid()
        self.type = 'AdversarialGridWorld_1'
        self.alternator = 0
        self.best_loss = 0.6 * self.args.T * self.args.samples_per_episode

    def reset(self):
        # random initial state
        self.state = [np.random.randint(0,int(self.grid_dim[1])), np.random.randint(0,int(self.grid_dim[1]))]

        # reset grid points
        if self.alternator:
            self.expert_actions_grid[:,:int(self.grid_dim[1]/2)] = 4
            self.expert_actions_grid[:,int(self.grid_dim[1]/2)+1:] = 4
            self.expert_actions_grid[:,int(self.grid_dim[1]/2)] = 1
            # self.state = [self.grid_dim[0]-1, np.random.randint(0,int(self.grid_dim[1]/2))]
            self.alternator = 0
        else:
            self.expert_actions_grid[:,:int(self.grid_dim[1]/2)] = 3
            self.expert_actions_grid[:,int(self.grid_dim[1]/2)+1:] = 3
            self.expert_actions_grid[:,int(self.grid_dim[1]/2)] = 0
            # self.state = [self.grid_dim[0]-1, np.random.randint(int(self.grid_dim[1]/2),self.grid_dim[1])]
            self.alternator = 1

        # reset time
        self.t = 1
        return self.one_hot(self.state)

    def get_expert_action(self):
        return self.expert_actions_grid[self.state[0],self.state[1]]

    def step(self, action):

        # update time
        self.t += 1
        # check end conditions
        done = 1 if self.t > self.T else 0
        # get reward
        reward = 1 if action == self.expert_actions_grid[self.state[0],self.state[1]]  else 0

        state = self.env_step(action) 
        # return
        return self.one_hot(self.state), reward, done, {'true_state':self.state}

    def imperfect_expert_grid(self):
        grid = np.zeros(self.grid_dim)
        grid[:,:int(self.grid_dim[1]/2)] = 4
        grid[:,int(self.grid_dim[1]/2)+1:] = 4
        grid[:,int(self.grid_dim[1]/2)] = 1
        return grid

    def map_states(self, state):
        return [state[0],0]
