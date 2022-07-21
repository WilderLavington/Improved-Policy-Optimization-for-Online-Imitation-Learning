
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
from online_learning.load_expert import pretrain_args
from online_learning.utils import stat_dicts, timer
from online_learning.algorithms import OGD, AdaOGD, FTL, FTRL, AFTRL, SFTRL


def generate_env(args):
    # Environment
    env = gym.make(args.env_name)
    # # set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # mae a copy
    args.env_copy = gym.make(args.env_name)
    return args, env

def initialization(args):
    # grab environment
    args, env = generate_env(args)
    # presets
    args.expert_params_path = 'expert_models/sac_actor_'+args.env_name.replace('-','_')+'_expert'
    # Agent
    trainer = eval(args.algo)(env, args)
    # return
    return trainer

def evaluate_il_algorithm():
    # get args
    args, parser = get_args()
    # init project
    if args.offline_wandb:
        os.environ["WANDB_MODE"] = "offline"
    # add ...
    args, parser = pretrain_args(parser=parser)
    # Environment
    env = gym.make(args.env_name)
    # set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # init project
    run = wandb.init(project=args.project, config=vars(args), entity='wilderlavington')
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
        args, parser = get_args()
        args, parser = pretrain_args(parser=parser)
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
