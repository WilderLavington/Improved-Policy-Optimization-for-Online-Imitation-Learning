
#
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
torch.manual_seed(1)
from tqdm import tqdm
import argparse
import os, sys
import wandb
#
sys.path.append(os.path.dirname(os.path.dirname('../../')))
sys.path.append(os.path.dirname(os.path.dirname('./')))

#
from online_learning.utils import *
from online_learning.plotting import *
from online_learning.run_experiments import *
from online_learning.environments.gridworld import *
from online_learning.problems import *
from online_learning.grads import *
from online_learning.losses import *
from online_learning.parsers import *
from online_learning.algorithms.optimal import *
from online_learning.algorithms.random import *

def initialize(info):
    # set some required info
    info.ACTION_DIM = 5
    info.STATE_DIM = info.ENV_DIMS_x * info.ENV_DIMS_y
    info.W_STAR = 2.5 * (0.5 - torch.rand((info.STATE_DIM,info.ACTION_DIM)))
    info.START_STATE = [info.START_STATE_x,info.START_STATE_y]
    info.ENV_DIMS = [info.ENV_DIMS_x, info.ENV_DIMS_y]
    info.loss_type = 'grid_world'
    info.param = 10**info.log_param
    info.lr = 10**info.log_lr
    #
    if info.problem_type == 'norm':
        info.sample_fxn = gridworld_mdp
    elif info.problem_type == 'adv':
        info.sample_fxn = adversarial_gridworld_mdp
    elif info.problem_type == 'stoch':
        info.sample_fxn = noisy_gridworld_mdp
    else:
        raise Exception('')
    # set info
    info.get_loss_grad = CSE_grad
    info.get_loss = CSE_loss
    info.OPTIMAL_step = OPTIMAL_step
    info.RANDOM_step = RANDOM_step
    #
    info.env = RandomActionGridWorld(info.START_STATE,info.ENV_DIMS,info.T)
    info.env.reset()
    return info

###
def main():
    info, parser = general_args()
    info, parser = grid_world_args(parser)
    info = initialize(info)
    #
    if info.exp_sweep:
        run_batch_experiments(info, name='gridworld_exp_'+info.exp_id)
        plot_experiments(name='gridworld_exp_'+info.exp_id)
    else:
        eval_algo(info, eval(info.algo))
if __name__ == "__main__":
    main()
