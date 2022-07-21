
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

def check_armijo_conditions(step_size, loss, grad_norm,  loss_next, c=0.2, beta_b=0.5):
    found = 0
    # computing the new break condition
    break_condition = loss_next - \
        (loss - (step_size) * c * grad_norm**2)
    if (break_condition <= 0):
        found = 1
    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b
    return found, step_size
def compute_step_size(get_loss, w_grad, w_, lr):
    # set up initial step size
    step_size = lr
    # get current loss
    loss = get_loss(w_)
    for i in range(100):
        w_next = deepcopy(w_) - step_size * w_grad
        loss_next = get_loss(w_next)
        found, step_size = check_armijo_conditions(step_size, loss, w_grad.norm(2),
                          loss_next, c=0.02, beta_b=0.9)
        if found:
            break
        elif step_size <= 1e-12:
            break
    return step_size, i
