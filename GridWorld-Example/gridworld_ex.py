
# generic imports
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

# from exp
from algorithms import FTRL, FTL, OGD
from environments import StairCaseGridWorld, RandomActionGridWorld, AdversarialGridWorld_1

def test(args):
    # 
    env = eval(args.grid_type)(args)
    # grab algorithm
    algo = eval(args.algo)
    # call train policy script
    algo(env, args).train_policy()
    #
    return None

def basic_args():
    parser = argparse.ArgumentParser(description='GridWorld Experiments.')
    # logging args
    parser.add_argument('--log_dir', type=str, default='./results/')
    parser.add_argument('--use_wandb', type=int, default=1)
    parser.add_argument('--csv_log', type=int, default=1)
    parser.add_argument('--seed', type=int, default=np.random.randint(1000))
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--action_noise', type=float, default=0. )
    # online learning args
    parser.add_argument('--algo', type=str, default='OGD', help='OGD,FTL,FTRL')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--inner_lr', type=float, default=1e-3)
    parser.add_argument('--outer_lr', type=float, default=5e-1)
    parser.add_argument('--expert_steps', type=int, default=1)
    parser.add_argument('--epochs_per_update', type=int, default=1000)
    parser.add_argument('--maxmem', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    # gridworld simulator args
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--samples_per_episode', type=int, default=5)
    parser.add_argument('--grid_dim_x', type=int, default=5)
    parser.add_argument('--grid_dim_y', type=int, default=5)
    parser.add_argument('--grid_type', type=str, default='AdversarialGridWorld_1', help='RandomActionGridWorld, StairCaseGridWorld, AdversarialGridWorld_1')
    parser.add_argument('--start_state_x', type=int, default=100)
    parser.add_argument('--start_state_y', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    print('begin test...')
    args = basic_args()
    test(args)
    print('test complete...')

if __name__ == "__main__":
    main()
