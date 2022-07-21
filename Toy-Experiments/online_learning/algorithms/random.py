
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from online_learning.linesearch import *
from online_learning.utils import *


def RANDOM_step(info, k): #(X, b, w, k, param, )
    new_info = deepcopy(info)
    new_info.w = 1.5 * (0.5 - torch.rand((info.STATE_DIM,info.ACTION_DIM)))
    # info added
    new_info.squared_grad_sum += info.get_loss_grad(info.w, info.X, info.b).norm(2).pow(2)
    new_info.policy_loss_grad_norm_list.append(info.get_loss_grad(info.w, info.X, info.b).norm(2).pow(2))
    new_info.eta_list.append(0)
    new_info.inner_loop_steps_list.append(1)
    new_info.algo_loss_grad_norm_list.append(info.get_loss_grad(info.w, info.X, info.b).abs().sum())
    return new_info
