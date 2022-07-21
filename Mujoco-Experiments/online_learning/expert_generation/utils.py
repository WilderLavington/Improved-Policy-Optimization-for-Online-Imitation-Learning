import math
import torch
import numpy as np
# optimizers stuff
from torch.optim import Adam, SGD, ASGD, RMSprop, LBFGS, Adagrad

def load_expert(num_inputs, action_space, args, device):
    expert = GaussianPolicy(num_inputs, action_space.shape[0], 256, action_space).to(device)
    expert.load_state_dict(torch.load(args.expert_params_path,map_location=device))
    return expert
    
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

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
    elif optim_type == 'AdaptiveFirst':
        raise Exception
        # model_optim = AdaptiveFirst(model.parameters(), init_step_size=args.lr)
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
        # model_optim = Sls(model.parameters(), init_step_size=args.init_step_size,
        #     beta_b=args.sls_beta_b, c=args.sls_c, gamma = args.sls_gamma,
        #     beta_f=args.sls_beta_f, n_batches_per_epoch=args.n_batches_per_epoch,
        #     eta_max = args.sls_eta_max)
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
