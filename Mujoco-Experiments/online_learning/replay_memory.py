import random
import numpy as np
from collections import deque
from torch.optim import SGD, Adam, RMSprop
from copy import deepcopy
import torch
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import os
from pathlib import Path

class ReplayMemory:
    def __init__(self, capacity, seed, args):
        # random.seed(seed)
        self.capacity = capacity
        self.args = args
        self.buffer = deque(maxlen=self.capacity)
        self.total_examples = 0
        self.traj_rew = 0.

    def push(self, state, action, reward, done):
        self.total_examples = min(self.total_examples + 1, self.capacity)
        if done:
            self.traj_rew = 0
        else:
            self.traj_rew += reward
        self.buffer.append((state, action, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size,  self.total_examples))
        return map(np.stack, zip(*batch))

    def __len__(self):
        return self.total_examples

    def save_examples(self, dir):
        state, action, reward, done = map(np.stack, zip(*self.buffer))
        state = torch.stack([torch.tensor(s) for s in state]).float().detach()
        action = torch.stack([torch.tensor(s) for s in action]).float().detach()
        reward = torch.stack([torch.tensor(s) for s in reward]).float().detach()
        done = torch.stack([torch.tensor(s) for s in done]).float().detach()
        save_dict = {'state':state, 'action': action, 'reward': reward, 'done': done}
        torch.save(save_dict, dir)
