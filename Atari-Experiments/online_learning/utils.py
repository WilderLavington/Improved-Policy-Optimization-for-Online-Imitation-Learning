import math
import torch
import numpy as np
from typing import Iterable, Optional
# optimizers stuff
from torch.optim import Adam, SGD, ASGD, RMSprop, LBFGS, Adagrad
from .lsopt import LSOpt

def avg_dicts(dict_list):
    avg_dict = {}
    for key in dict_list[0].keys():
        avg_dict[key] = sum(item.get(key,0) for item in dict_list) / len(dict_list)
    return avg_dict

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
def replace_args(args, argsdict={}):
    """ forcably replace values in arguments. """
    for key in list(argsdict.keys()):
        args.__dict__[key] = argsdict[key]
    return args
def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p
def stat_dicts(dict_list):
    # mean
    stat_dict = {}
    for key in dict_list[0].keys():
        if (dict_list[0][key] is not None):
            vals = np.array([item.get(key,0).cpu().detach().numpy() \
                    if torch.is_tensor(item.get(key,0)) else item.get(key,0) \
                        for item in dict_list])
            stat_dict[key+'_mean'] = np.mean(vals)
            stat_dict[key+'_min'] = np.min(vals)
            stat_dict[key+'_max'] = np.max(vals)
            stat_dict[key+'_25_quant'] = np.quantile(vals, 0.25)
            stat_dict[key+'_75_quant'] = np.quantile(vals, 0.75)
            stat_dict[key+'_05_quant'] = np.quantile(vals, 0.05)
            stat_dict[key+'_95_quant'] = np.quantile(vals, 0.95)

    return stat_dict
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def select_optimizers(policy, critic, args):

    critic_optim = select_optimizer(critic, args, args.critic_optim)
    policy_optim = select_optimizer(policy, args, args.policy_optim)

    # return
    return policy_optim, critic_optim

def select_optimizer(model, args, optim_type):
    # critic
    if optim_type == 'Adam':
        model_optim = Adam(model.parameters(), lr=args.lr)
    elif optim_type == 'SGD':
        model_optim = SGD(model.parameters(), lr=args.lr)
    elif optim_type == 'ASGD':
        model_optim = ASGD(model.parameters(), lr=args.lr)
    elif optim_type == 'RMSprop':
        model_optim = RMSprop(model.parameters(), lr=args.lr)
    elif optim_type == 'LSOpt':
        model_optim = LSOpt(model.parameters(), init_step_size=args.lr)
    elif optim_type == 'Ssn':
        raise Exception
        # model_optim = Ssn(model.parameters(), init_step_size=args.lr, n_batches_per_epoch=1)
    elif optim_type == 'SlsEg':
        raise Exception
        # model_optim = SlsEg(model.parameters(), init_step_size=args.lr)
    elif optim_type == 'SlsAcc':
        raise Exception
        # model_optim = SlsAcc(model.parameters(), init_step_size=args.lr)
    elif optim_type == 'Sls':
        raise Exception 
    elif optim_type == 'Adagrad':
        model_optim = Adagrad(model.parameters(), lr=args.lr)
    elif optim_type == 'LBFGS':
        model_optim = LBFGS(model.parameters(), lr=args.lr)
    else:
        raise Exception('wtf'+optim_type)
    # return
    return model_optim

def check_grad(model):

    print('looking at named gradients.....')
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(name, torch.sum(param.grad))
        else:
            pass

    print('looking at all parameters.....')
    for k, v in model.state_dict().items():
        print(k, type(v))

    if set([k[0] for k in model.state_dict().items()]) == set([k[0] for k in model.named_parameters()]):
        print('all parameters accounted for.')
    else:
        print('some parameters not accounted for.')

    return None

def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    r"""Convert parameters to one vector
    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.reshape(-1))
    return torch.cat(vec)

def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.
    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.
    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device
