#
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from online_learning.linesearch import *
from online_learning.utils import *
def OGD_step(info, k):
    w_k = info.w
    w_grad = OGD_grad(info, info.X, info.b, info.w, w_k, k, info.param)
    info.w = info.w - (info.lr / k)**0.5 * w_grad
    # info added
    info.policy_loss_grad_norm_list.append(info.get_loss_grad(info.w, info.X, info.b).norm(2).pow(2))
    info.eta_list.append(0)
    info.inner_loop_steps_list.append(1)
    info.algo_loss_grad_norm_list.append(w_grad.abs().sum())
    return info
def OGD_grad(info, X, b, w, w_k, k=None,param=None):
    return info.get_loss_grad(w, X[:info.SAMPLE_SIZE,:], b[:info.SAMPLE_SIZE])
