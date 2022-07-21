
# sci-compute imports
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy import interpolate
from torch.distributions import Normal
from functools import reduce
import gym
import shutil

# general imports
import itertools
from copy import deepcopy
import os
import operator
import glob
import natsort
import pickle

# display and logging
import datetime
import time
import imageio
import wandb
import matplotlib.pyplot as plt

def fill_buffer(args, agent, memory, env, eval_expert=True):
    # Training Loop
    total_numsteps = 0
    updates = 1
    start = time.time()
    # fill the buffer
    avg_reward = 0.
    episodes = 0
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        full_state = np.concatenate([deepcopy(env.sim.data.qpos), deepcopy(env.sim.data.qvel)])
        episode_reward = 0.
        while not done:
            action, log_prob = agent.select_action(state, eval_expert=eval_expert)  # Sample action from policy
            next_state, reward, done, _ = env.step(action) # Step
            next_full_state = np.concatenate([deepcopy(env.sim.data.qpos), deepcopy(env.sim.data.qvel)])
            episode_steps += 1
            total_numsteps += 1
            mask = float(not done)
            # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask, log_prob,
                fstate=full_state, next_fstate=next_full_state, bad_mask=None) # Append transition to memory
            state = next_state
            full_state = deepcopy(next_full_state)
            episode_reward += reward
        avg_reward += episode_reward
        episodes += 1
        if total_numsteps >= len(memory):
            break
    avg_reward /= episodes
    memory.avg_return = avg_reward
    return memory

def eval_policy(agent, env, duplicates=10):
    avg_reward = 0.
    for _  in range(duplicates):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
    avg_reward /= duplicates
    return avg_reward

def compute_buffer_autocorr(memory, k_lag=250):
    # get full data-set
    _, (state, _, _, _, _, traj_id, timestep, _, _) = memory.tensor_dataset()
    sorted_states = deepcopy(state)
    # init
    auto_corr = torch.zeros((k_lag-1, state.size()[-1]))
    # iterate through trajectories
    for traj in torch.unique(traj_id):
        # get indexes where traj lies
        idx = torch.nonzero(traj_id == traj).flatten()
        # pull examples for this run
        traj_timesteps = timestep[idx]
        traj_states = state[idx]
        # sort based on time-step
        sorted_states[traj_timesteps.long()] = traj_states
        # compute mean
        mean_X = sorted_states.mean(dim=0)
        # shifted
        shifted_states = sorted_states-mean_X
        # compute auto-correlation-norm
        norm_X = (shifted_states*sorted_states).sum(dim=0)
        # compute (appriximate k-lag autocorrelation)
        for lag in range(1,k_lag):
            auto_corr[lag-1,:] += (shifted_states[:-lag,:]*shifted_states[lag:,:]).sum(dim=0) / (norm_X + 1e-8)
        # average
        auto_corr /= len(torch.unique(traj_id))
    # return target autocorrelation
    return auto_corr

def compute_buffer_marginal_kdes(memory, bins=25):
    # get full data-set
    _, (state, _, _, _, _, _, _, _, _) = memory.tensor_dataset()
    # init
    marginal_kdes = {}
    marginal_kdes['state_size'] = state.size()[1]
    # for each index
    for c in range(state.size()[1]):
        # generate hist info
        fig = plt.subplots(figsize=(10,7))
        marginal_info = plt.hist(state[:,c].numpy(), bins=bins, density=True)
        # compute the per-index kde-estimate
        marginal_vals = torch.tensor([list(marginal_info[0]), list(marginal_info[1][1:])])
        marginal_fxn = interpolate.interp1d(marginal_vals[1,:], marginal_vals[0,:], 'cubic')
        # figure out bounds
        ub = max(marginal_vals[1,:])
        lb = min(marginal_vals[1,:])
        #
        x = np.arange(lb,ub,0.0001)
        y = marginal_fxn(x)
        # now we have a kde estimate of the marginal
        marginal_kdes[str(c)] = marginal_fxn
        marginal_kdes[str(c)+'_x'] = x
        marginal_kdes[str(c)+'_y'] = y
    # now return the info
    return marginal_kdes
