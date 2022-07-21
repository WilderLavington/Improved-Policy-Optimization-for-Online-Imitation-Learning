
#
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
torch.manual_seed(1)
from tqdm import tqdm
import argparse
import os, sys

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
    info.W_STAR = 2.5 * (0.5 - torch.rand((info.STATE_DIM,info.ACTION_DIM)))
    info.loss_type = 'l1'
    info.param = 10**info.log_param
    info.lr = 10**info.log_lr
    #
    if info.problem_type == 'norm':
        info.sample_fxn = online_reg
    elif info.problem_type == 'adv':
        info.sample_fxn = adversarial_online_reg
    elif info.problem_type == 'stoch':
        info.sample_fxn = noisy_online_reg
    else:
        raise Exception('')
    # set info
    info.get_loss_grad = L1_grad
    info.get_loss = L1_loss
    info.OPTIMAL_step = OPTIMAL_step
    info.RANDOM_step = RANDOM_step
    #
    return info

###
def main():

    info, parser = general_args()
    info = initialize(info)

    if info.exp_sweep:
        run_batch_experiments(info, name='l1_exp_'+info.exp_id)
        plot_experiments(name='l1_exp_'+info.exp_id)
    else:
        eval_algo(info, eval(info.algo))

if __name__ == "__main__":
    main()
