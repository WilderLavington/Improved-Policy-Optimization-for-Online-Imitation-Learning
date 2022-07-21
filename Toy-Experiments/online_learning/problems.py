
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from online_learning.utils import *

# discrete state, discrete action
def gridworld_mdp(k, info, beta=0):
    # sample action
    X_k = torch.tensor(info.env.one_hot(info.env.state)).unsqueeze(0)
    output = torch.mm(X_k.double(), info.w.double()) #+ (1 - 0.5 * torch.rand((SAMPLE_SIZE,3)))
    p_k = torch.softmax(output+1e-12,dim=1)
    a_k = torch.distributions.Categorical(p_k).sample()
    # compute true output
    output = torch.mm(X_k.double(), info.W_STAR.double()) #+ (1 - 0.5 * torch.rand((SAMPLE_SIZE,3)))
    p_k = torch.softmax(output+1e-12,dim=1)
    b_k = torch.distributions.Categorical(p_k).sample()
    # step-env
    if not beta:
        info.env.step(a_k.detach().numpy())
    else:
        info.env.step(b_k.detach().numpy())
    # return it all
    return X_k, b_k

def adversarial_gridworld_mdp(k, info, beta=0):

    #
    info.X_k = torch.tensor(info.env.one_hot(info.env.state)).unsqueeze(0)
    output = torch.mm(info.X_k.double(), info.w.double()) #+ (1 - 0.5 * torch.rand((SAMPLE_SIZE,3)))
    p_k = torch.softmax(output+1e-12,dim=1)
    a_k = torch.distributions.Categorical(p_k).sample()

    # compute true output
    output = torch.mm(info.X_k.double(), info.W_STAR.double()) #+ (1 - 0.5 * torch.rand((SAMPLE_SIZE,3)))
    if k % 2 == 0:
        p_k = torch.softmax(output,dim=1)+1e-12
        info.b_k = torch.distributions.Categorical(p_k).sample()
    else:
        p_k = torch.softmax(-output,dim=1)+1e-12
        info.b_k = torch.distributions.Categorical(p_k).sample()

    # step-env
    if not beta:
        info.env.step(a_k.detach().numpy())
    else:
        info.env.step(b_k.detach().numpy())

    # return it all
    return info.X_k, info.b_k

def noisy_gridworld_mdp(k, info, beta=0):

    #
    info.X_k = torch.tensor(info.env.one_hot(info.env.state)).unsqueeze(0)
    output = torch.mm(info.X_k.double(), info.w.double()) #+ (1 - 0.5 * torch.rand((SAMPLE_SIZE,3)))
    p_k = torch.softmax(output+1e-12,dim=1)
    a_k = torch.distributions.Categorical(p_k).sample()

    # compute true output
    noisy_W = info.W_STAR.double() + k * (0.5 - torch.rand((info.STATE_DIM,info.ACTION_DIM)))
    output = torch.mm(info.X_k.double(), noisy_W) #+ (1 - 0.5 * torch.rand((SAMPLE_SIZE,3)))
    p_k = torch.softmax(output,dim=1)+1e-12
    info.b_k = torch.distributions.Categorical(p_k).sample()

    # step-env
    if not beta:
        info.env.step(a_k.detach().numpy())
    else:
        info.env.step(b_k.detach().numpy())

    # return it all
    return info.X_k, info.b_k

# continous state, continous action
def online_reg(k, info):
    X_k = 15 * (0.5 - torch.rand((info.SAMPLE_SIZE,info.STATE_DIM)))
    output = torch.mm(X_k.double(), info.W_STAR.double()) #
    b_k = output + (1 - 0.5 * torch.rand((info.SAMPLE_SIZE, info.ACTION_DIM)))
    return X_k, b_k

def adversarial_online_reg(k, info):
    X_k = 15 * (0.5 - torch.rand((info.SAMPLE_SIZE,info.STATE_DIM)))
    output = torch.mm(X_k.double(), info.W_STAR.double()) #
    if k % 2 == 0:
        b_k = output + (1 - 0.5 * torch.rand((info.SAMPLE_SIZE,info.ACTION_DIM)))
    else:
        b_k = - output + (1 - 0.5 * torch.rand((info.SAMPLE_SIZE,info.ACTION_DIM)))
    return X_k, b_k

def noisy_online_reg(k, info):
    X_k = 15 * (0.5 - torch.rand((info.SAMPLE_SIZE,info.STATE_DIM)))
    noisy_W = info.W_STAR.double() + k * (0.5 - torch.rand((info.STATE_DIM,info.ACTION_DIM)))
    output = torch.mm(X_k.double(), noisy_W.double()) #
    b_k = - output + k * (1 - 0.5 * torch.rand((info.SAMPLE_SIZE,3)))
    return X_k, b_k

# continuous state, dicrete action
def logistic_reg(k, info, prob=0.5):
    X_k = 15 * (0.5 - torch.rand((info.SAMPLE_SIZE,info.STATE_DIM)))
    output = torch.mm(X_k.double(), info.W_STAR.double()) #+ (1 - 0.5 * torch.rand((SAMPLE_SIZE,3)))
    p_k = torch.softmax(output+1e-12,dim=1)
    b_k = torch.distributions.Categorical(p_k).sample()
    if torch.rand(1).item() < prob:
        b_k = torch.distributions.Categorical(torch.ones(p_k.size())).sample()
    return X_k, b_k

def adversarial_logistic_reg(k, info, prob=0.5):
    X_k = 15 * (0.5 - torch.rand((info.SAMPLE_SIZE,info.STATE_DIM)))
    if k % 2 == 0:
        output = torch.mm(X_k.double(), info.W_STAR.double()) #+ (1 - 0.5 * torch.rand((SAMPLE_SIZE,3)))
    else:
        output = torch.mm(X_k.double(), -info.W_STAR.double())
    p_k = torch.softmax(output+1e-12,dim=1)
    b_k = torch.distributions.Categorical(p_k).sample()
    if torch.rand(1).item() < prob:
        b_k = torch.distributions.Categorical(torch.ones(p_k.size())).sample()
    return X_k, b_k

def noisy_logistic_reg(k, info, prob=0.5):
    X_k = 15 * (0.5 - torch.rand((info.SAMPLE_SIZE,info.STATE_DIM)))
    noisy_W = info.W_STAR.double() + k * (0.5 - torch.rand((info.STATE_DIM,info.ACTION_DIM)))
    output = torch.mm(X_k.double(), info.W_STAR.double()) #+ (1 - 0.5 * torch.rand((SAMPLE_SIZE,3)))
    p_k = torch.softmax(output+1e-12,dim=1)
    b_k = torch.distributions.Categorical(p_k).sample()
    if torch.rand(1).item() < prob:
        b_k = torch.distributions.Categorical(torch.ones(p_k.size())).sample()
    return X_k, b_k
