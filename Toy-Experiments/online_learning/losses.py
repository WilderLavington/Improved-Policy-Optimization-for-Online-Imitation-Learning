
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from online_learning.utils import *

def CSE_loss(w, X, b):
    w_param = torch.tensor(w.numpy(), requires_grad=True)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    logit = torch.mm(X.double(), w_param.double())
    loss = criterion(logit, b.long().view(-1))
    return loss

def L2_loss(w, X, b):
    w_param = torch.tensor(w.numpy(), requires_grad=True)
    l2_loss = torch.nn.MSELoss(reduction='sum')
    output = torch.mm(X.double(), w_param.double())
    loss = l2_loss(output, b)
    return loss

def L1_loss(w, X, b):
    w_param = torch.tensor(w.numpy(), requires_grad=True)
    l1_loss = torch.nn.L1Loss(reduction='sum')
    output = torch.mm(X.double(), w_param.double())
    loss = l1_loss(output, b)
    return loss

def Huber_loss(w, X, b):
    w_param = torch.tensor(w.numpy(), requires_grad=True)
    l1_loss = torch.nn.HuberLoss(reduction='sum')
    output = torch.mm(X.double(), w_param.double())
    loss = l1_loss(output, b)
    return loss
