
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from online_learning.linesearch import *
from online_learning.utils import *

def OPTIMAL_step(info, k): #(X, b, w, k, param, )
    new_info = deepcopy(info)
    if info.problem_type != 'adv':
        new_info.w = info.W_STAR
    else:
        new_info.w = 0 * info.W_STAR
    # info added
    new_info.policy_loss_grad_norm_list.append(info.get_loss_grad(info.w, info.X, info.b).norm(2).pow(2))
    new_info.eta_list.append(0)
    new_info.inner_loop_steps_list.append(1)
    new_info.algo_loss_grad_norm_list.append(info.get_loss_grad(info.w, info.X, info.b).abs().sum())
    new_info.squared_grad_sum += info.get_loss_grad(info.w, info.X, info.b).norm(2).pow(2)
    return new_info
