
# general imports
import datetime
import gym
import numpy as np
import itertools
import torch
import time
from copy import deepcopy
import wandb
import os

# project imports
from online_learning.parser import get_args
from online_learning.utils import avg_dicts, stat_dicts, timer
from online_learning.algorithms import OGD, AdaOGD, AdamOGD, FTL, FTRL, AFTRL, SFTRL
import subprocess

def generate_env(args): 
    # Environment
    env = gym.make(args.env_name)
    # # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # mae a copy
    args.env_copy = gym.make(args.env_name)
    return args, env

def initialization(args):
    # grab environment
    args, env = generate_env(args)
    # presets
    args.expert_params_path = 'online_learning/expert_generation/expert_models/sac_actor_'+args.env_name.replace('-','_')+'_expert'
    # Agent
    trainer = eval(args.algo)(env, args)
    # return
    return trainer

def evaluate_il_algorithm():
    # get args
    args,_ = get_args()
    # init project
    if args.offline_wandb:
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ["WANDB_MODE"] = "online"
    run = wandb.init(project=args.project, config=vars(args), entity=args.entity)
    # set trainer classes up with seperate seeds
    trainers = [initialization(args)]
    for seed in range(args.multi_seed_run-1):
        args.seed = args.seed * seed
        trainers.append(initialization(args))
    for episode in range(args.episodes):
        # step all models and store average info
        [t.agent_update() for t in trainers]
        # log if we dare
        if (episode % args.log_interval == 0):
            eval_info = stat_dicts([t.agent_info() for t in trainers])
            wandb.log(eval_info)
            print("=========================================")
            print('Iteration info:',eval_info)
            print('Elapsed time:', timer(trainers[0].start, time.time()))
            print("=========================================")
    # finish the current run
    save_dir = run.dir
    run.finish()

def train_il_agent(args=None):
    # get args
    if args is None:
        args,_ = get_args()
    # init project
    run = wandb.init(project=args.project, config=vars(args), entity='wilderlavington')
    # set trainer classes up with seperate seeds
    trainer = initialization(args)
    # train model
    trainer.train_agent()
    # finish the current run
    run.finish()

if __name__ == "__main__":
    main()
