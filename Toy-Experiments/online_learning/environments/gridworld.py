import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
torch.manual_seed(1)
from tqdm import tqdm

# gridworld stuff
class GridWorld():
    """
    GRID WORLD CLASS: HELPER CLASS TO INSTANTIATE A SMALL MDP PROBLEM TO TEST POLICY GRADIENT
    METHODS ON TO ENSURE CORRECTNESS, ALSO ALOWS FOR FOR TESTING OF NEW TYPES OF ENVIROMENTS
    WITH DIFFERENT STATESPACE DISTRIBUTIONS ECT.
    """
    def __init__(self, start_state, grid_dim, T ):
        super(GridWorld, self).__init__()
        self.grid = None
        start_state_x,start_state_y = start_state
        grid_dim_x,grid_dim_y = grid_dim
        self.start_state = [start_state_x, start_state_y]
        self.state = self.start_state
        self.grid_dim = [grid_dim_x, grid_dim_y]
        self.action_noise = 0.1
        self.T = T
        self.t = 0
        self.obs_size = self.grid_dim[0]*self.grid_dim[1]
        assert (self.action_noise >= 0.) and (self.action_noise <= 1.)
#         self.action_space = gym.spaces.Discrete(5)

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
            self.env_step(1+np.random.randint(4))
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
    def __init__(self, start, grid_dims, T):
        super(RandomActionGridWorld, self).__init__(start, grid_dims, T)
        # random initial state
        self.start_state = [np.random.randint(0,self.grid_dim[0]),np.random.randint(0,self.grid_dim[1])]
        self.state = self.start_state
        self.one_hot_state = self.one_hot(self.state)
        # random set of expert actions
        self.expert_actions_grid = np.random.randint(5, size=self.grid_dim)
        self.type = 'RandomActionWorld'
        self.best_loss = 0.0

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
