
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

def add_data(X_k,X):
    return torch.cat([X_k,X],dim=0)

class INFO:
    pass
