
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from online_learning.linesearch import *
from online_learning.utils import *

from online_learning.linesearch import *
from online_learning.utils import *

def AFTRL_grad(info,X, b, w, w_k, k, squared_grad_sum, param):
    w_grad = info.get_loss_grad(w, X, b)
    if k > 1:
        w_grad -= info.get_loss_grad(w_k, X[info.SAMPLE_SIZE:,:], b[info.SAMPLE_SIZE:])
    w_grad += squared_grad_sum**0.5 * (param) * (w - w_k)
    return w_grad
def AFTRL_loss(info,X, b, w, w_k, k, squared_grad_sum, param):
    ftl_loss = info.get_loss(w, X, b)
    if k > 1:
        lin_loss = (w.reshape(-1) * info.get_loss_grad(w_k, X[info.SAMPLE_SIZE:,:], b[info.SAMPLE_SIZE:]).reshape(-1)).sum()
    else:
        lin_loss = 0.
    tr_loss = 0.5 * squared_grad_sum**0.5 * (param) * (w - w_k).pow(2).sum()
    return (ftl_loss - lin_loss + tr_loss)
def AFTRL_step(info, k):
    w_k = info.w
    info.squared_grad_sum += info.get_loss_grad(info.w, info.X, info.b).norm(2).pow(2)
    for iter_ in range(1,info.max_inner_steps):
        w_grad = AFTRL_grad(info, info.X, info.b, info.w, w_k, k, info.squared_grad_sum, info.param) / (k+1)
        get_loss_ = lambda w_: AFTRL_loss(info, info.X, info.b, w_, w_k, k, info.squared_grad_sum, info.param) / (k+1)
        ac_lr, num_iters = compute_step_size(get_loss_, w_grad, info.w, info.lr)
        info.w = info.w - ac_lr * w_grad
        if w_grad.abs().sum() < 1e-8:
            info.policy_loss_grad_norm_list.append(info.get_loss_grad(info.w, info.X, info.b).norm(2).pow(2))
            info.eta_list.append( k**0.5 )
            info.inner_loop_steps_list.append(iter_)
            info.algo_loss_grad_norm_list.append(w_grad.abs().sum())
            return info
    # info added
    info.policy_loss_grad_norm_list.append(info.get_loss_grad(info.w, info.X, info.b).norm(2).pow(2))
    info.eta_list.append((info.squared_grad_sum)**0.5)
    info.inner_loop_steps_list.append(iter_)
    info.algo_loss_grad_norm_list.append(w_grad.abs().sum())
    # return
    return info
