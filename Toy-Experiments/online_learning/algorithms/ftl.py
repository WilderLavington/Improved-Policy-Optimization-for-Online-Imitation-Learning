
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from online_learning.linesearch import *
from online_learning.utils import *

def FTL_step(info, k):
    w_k = info.w
    for iter in range(1,info.max_inner_steps):
        w_grad = FTL_grad(info) / (k+1)
        get_loss_ = lambda w_: info.get_loss(w_, info.X, info.b) / (k+1)
        ac_lr, _ = compute_step_size(get_loss_, w_grad, info.w, info.lr)
        info.w = info.w - ac_lr * w_grad
        if w_grad.abs().sum() < 1e-8:
            info.policy_loss_grad_norm_list.append(info.get_loss_grad(info.w, info.X, info.b).norm(2).pow(2))
            info.eta_list.append( k**0.5 )
            info.inner_loop_steps_list.append(iter)
            info.algo_loss_grad_norm_list.append(w_grad.abs().sum())
            return info
    # info added
    info.policy_loss_grad_norm_list.append(info.get_loss_grad(info.w, info.X, info.b).norm(2).pow(2))
    info.eta_list.append(0)
    info.inner_loop_steps_list.append(iter)
    info.algo_loss_grad_norm_list.append(w_grad.abs().sum())
    return info

def FTL_grad(info):
    w_grad = info.get_loss_grad(info.w, info.X, info.b)
    return w_grad
