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
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.args = args
        #
        self.traj_id, self.timestep = 0, -1
        self.traj_rew, self.unif_log_prob = 0., None
        #
        self.buffer = deque(maxlen=self.capacity)
        self.total_examples = 0
        self.avg_return = None


    def push(self, state, action, reward, next_state, done, log_prob,
            bad_mask=None, fstate=None, next_fstate=None):

        if self.unif_log_prob is None:
            self.unif_log_prob = len(action) * np.log((1/2))
            self.target_entropy = 0.
        # update strat sampling info
        self.timestep += 1
        self.total_examples = min(self.total_examples + 1, self.capacity)
        self.traj_rew += reward

        if type(log_prob) == np.ndarray:
            log_prob = log_prob.item()

        # only use uniform sampler
        self.buffer.append((state, action, reward, next_state, done, \
            self.traj_id, self.timestep, fstate, next_fstate))

        # more general resets at done
        if not done:
            # now reset it all
            self.traj_id += 1
            self.timestep = -1
            self.traj_rew = 0.
            self.strat_buffer = []

    def sample(self, batch_size, use_full_state = False, get_full_info = False):
        if get_full_info:
            batch = random.sample(self.buffer, min(batch_size,  self.total_examples))
            _, action, reward, _, done, traj_id, timestep, state, next_state = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done, traj_id, timestep
        if not use_full_state:
            batch = random.sample(self.buffer, min(batch_size,  self.total_examples))
            state, action, reward, next_state, done,_,_,_,_ = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done
        else:
            batch = random.sample(self.buffer, min(batch_size,  self.total_examples))
            _, action, reward, _, done, _, _, state, next_state = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done

    def __len__(self):
        return self.total_examples

    def viz_dist(self,x,sample_type):
        if x is not None:
            df = pd.DataFrame({
                'id': torch.tensor(x)[:,0].numpy(), 'return': torch.tensor(x)[:,1].numpy(),
                'log_prob': torch.tensor(x)[:,2].numpy(), 'T': torch.tensor(x)[:,3].numpy(),})
            fig = df.hist(bins=min(25,len(x)))
            plt.savefig(sample_type+'_samples_figure.pdf')
            plt.close('all')

    def save_examples(self, dir):
        state, action, reward, next_state, done, traj_id, timestep, fstate, next_fstate = map(np.stack, zip(*self.buffer))
        state =torch.stack([torch.tensor(s) for s in state]).float().detach()
        action = torch.stack([torch.tensor(s) for s in action]).float().detach()
        reward = torch.stack([torch.tensor(s) for s in reward]).float().detach()
        next_state = torch.stack([torch.tensor(s) for s in next_state]).float().detach()
        done = torch.stack([torch.tensor(s) for s in done]).float().detach()
        traj_id = torch.stack([torch.tensor(s) for s in traj_id]).float().detach()
        timestep = torch.stack([torch.tensor(s) for s in timestep]).float().detach()
        fstate = torch.stack([torch.tensor(s) for s in fstate]).float().detach()
        next_fstate = torch.stack([torch.tensor(s) for s in next_fstate]).float().detach()
        save_dict = {'state':state, 'action': action,
        'reward': reward, 'next_state': next_state, 'next_fstate': next_fstate,
        'done': done, 'timestep':timestep, 'traj_id':traj_id, 'fstate':fstate}
        torch.save(save_dict, dir)
        # print(save_dict)
        print('saved dataloader to '+dir)
        return None

    def tensor_dataset(self):
        state, action, reward, next_state, done, traj_id, timestep, fstate, next_fstate = zip(*self.buffer)
        state =torch.stack([torch.tensor(s) for s in state]).float()
        action = torch.stack([torch.tensor(s) for s in action]).float()
        reward = torch.stack([torch.tensor(s) for s in reward]).float()
        next_state = torch.stack([torch.tensor(s) for s in next_state]).float()
        done = torch.stack([torch.tensor(s) for s in done]).float()
        traj_id = torch.stack([torch.tensor(s) for s in traj_id]).float()
        timestep = torch.stack([torch.tensor(s) for s in timestep]).float()
        fstate = torch.stack([torch.tensor(s) for s in fstate]).float()
        next_fstate = torch.stack([torch.tensor(s) for s in next_fstate]).float()
        dataset = torch.utils.data.TensorDataset(state, action, reward, next_state, done, traj_id, timestep, fstate, next_fstate)
        return dataset, (state, action, reward, next_state, done, traj_id, timestep, fstate, next_fstate)

    def create_torch_dataset(self, tensor_data_set=True):
        if tensor_data_set:
            state, action, reward, next_state, done, traj_id, timestep,_,_ = zip(*self.buffer)
            state = torch.stack([torch.tensor(s) for s in state]).float()
            action = torch.stack([torch.tensor(s) for s in action]).float()
            reward = torch.stack([torch.tensor(s) for s in reward]).float()
            next_state = torch.stack([torch.tensor(s) for s in next_state]).float()
            done = torch.stack([torch.tensor(s) for s in done]).float()
            traj_id = torch.stack([torch.tensor(s) for s in traj_id]).float()
            timestep = torch.stack([torch.tensor(s) for s in timestep]).float()
            self.torch_dataset = torch.utils.data.TensorDataset(state, action, reward, next_state, done, traj_id, timestep)
        else:
            self.torch_dataset = torch.utils.data.Dataset(self.buffer)
        return self.torch_dataset

    def create_torch_dataloader(self, train_set, sampler='with_replacement'):
        if sampler=='with_replacement':
            rand_sampler = torch.utils.data.RandomSampler(train_set, num_samples=self.args.batch_size, replacement=True)
            self.torch_dataloader = DataLoader(train_set, drop_last=False,
                    sampler=rand_sampler, batch_size=self.args.batch_size)
        else:
            self.torch_dataloader = DataLoader(train_set, drop_last=True, shuffle=True,
                    sampler=None, batch_size=self.args.batch_size)
        self.data_generator = iter(self.torch_dataloader)
        return self.torch_dataloader
