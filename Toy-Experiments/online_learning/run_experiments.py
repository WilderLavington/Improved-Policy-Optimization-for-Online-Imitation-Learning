#
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import wandb
#
from online_learning.algorithms.ftl import *
from online_learning.algorithms.ftrl import *
from online_learning.algorithms.aftrl import *
from online_learning.algorithms.aoftrl import *
from online_learning.algorithms.ogd import *
from online_learning.algorithms.saftrl import *
from online_learning.algorithms.sftrl import *
from online_learning.algorithms.adagrad import *
from online_learning.algorithms.optimal import *
from online_learning.algorithms.random import *
from online_learning.utils import *
from online_learning.plotting import *
from online_learning.environments.gridworld import *
#
def create_stats(infos):
    stat_dict = {}
    stat_dict['eta_list'] = np.array([np.array(info.eta_list) for info in infos])
    stat_dict['inner_loop_steps'] = np.array([np.array(info.inner_loop_steps_list) for info in infos])
    stat_dict['policy_loss_grad_norm'] = np.array([np.array(info.policy_loss_grad_norm_list) for info in infos])
    stat_dict['algo_loss_grad_norm'] = np.array([np.array(info.algo_loss_grad_norm_list) for info in infos])
    stat_dict['w_star_diff'] = np.array([np.array(info.w_star_diff_list) for info in infos])
    stat_dict['performance'] = np.array([np.array(info.performance_list) for info in infos])
    stat_dict['loss_in_hindsight'] = np.array([np.array(info.loss_in_hindsight) for info in infos])
    keys = deepcopy(list(stat_dict.keys()))
    eval_dict = {}
    for key in keys:
        algo_eval = np.array(stat_dict[key])
        eval_dict[key+'_mean'], eval_dict[key+'_max'], eval_dict[key+'_min']  = np.mean(algo_eval, axis=0), np.quantile(algo_eval, 0.75, axis=0), np.quantile(algo_eval, 0.25, axis=0)
        algo_cumu_sum = np.cumsum(np.array(algo_eval), axis=1)
        eval_dict[key+'_cumu_mean'], eval_dict[key+'_cumu_max'], eval_dict[key+'_cumu_min'] = np.mean(algo_cumu_sum, axis=0), np.quantile(algo_cumu_sum, 0.75, axis=0), np.quantile(algo_cumu_sum, 0.25, axis=0)
    return eval_dict, keys
#
def train_policy(algo, info_):
    info = deepcopy(info_)
    info.w = torch.tensor(np.zeros((info.STATE_DIM, info.ACTION_DIM)))
    info.X, info.b = info.sample_fxn(k=1, info=info)
    info.X_k, info.b_k = info.X, info.b
    info.squared_grad_sum=0.
    info.squared_grad_sum_diff = 0.
    info.squared_grad_list = []
    # logging
    info.eta_list = []
    info.inner_loop_steps_list = []
    info.policy_loss_grad_norm_list = []
    info.algo_loss_grad_norm_list = []
    info.w_star_diff_list = []
    info.performance_list = []
    # train model over n rounds
    for k in tqdm(range(1, info.total_rounds+1)):
        # evaluate perf
        info.performance_list.append(info.get_loss(info.w, info.X_k, info.b_k).detach().numpy())
        info.w_star_diff_list.append((info.w-info.W_STAR).mean().detach().numpy())
        # call training algorithm
        info = algo(info, k)
        # sample new data eval
        info.X_k, info.b_k = info.sample_fxn(k=k, info=info)
        # add it to the total data
        info.X, info.b = add_data(info.X_k,info.X), add_data(info.b_k,info.b)
    # return info
    return info.w, info.performance_list, info
# regret_computation
def get_best_in_hindsight(info, k):
    info_k = deepcopy(info)
    info_k.X, info_k.b = info.X[:info.SAMPLE_SIZE * (k),:], info.b[:info.SAMPLE_SIZE * (k)]
    info_k = FTL_step(info_k, k)
    hindsight_loss = info.get_loss(info_k.w, info_k.X, info_k.b).detach().numpy()
    return hindsight_loss
#
def add_hindsight_info(infos):
    for info in tqdm(infos):
        info.loss_in_hindsight = []
        for k in tqdm(range(1,len(info.performance_list)+1),leave=False):
            hindsight_loss = get_best_in_hindsight(info, k)
            info.loss_in_hindsight.append(hindsight_loss)

    return infos
#
def eval_algo(info, algo):
    infos =[train_policy(algo, deepcopy(info))[2] for _ in range(5)]
    infos = add_hindsight_info(infos)
    stat_dict, keys = create_stats(infos)
    wandb.init(project=info.project, config=info)
    for i in tqdm(range(len(stat_dict['performance_mean']))):
        log_dict = {"round": i, "loss_type": str(info.loss_type)}
        log_dict.update({key:stat_dict[key][i] for key in stat_dict.keys()})
        wandb.log(log_dict)
    wandb.log({'policy_loss':stat_dict['performance_mean'][-1], 'cumu_policy_loss':stat_dict['performance_cumu_mean'][-1]})
    return stat_dict
#
def run_batch_experiments(info, name='test_img'):
    # experiments
    print('running ftl experiments...')
    ftl = [np.array(train_policy(FTL_step, info)[1]) for _ in range(10)]
    print('running ogd experiments...')
    ogd = [np.array(train_policy(OGD_step, info)[1]) for _ in range(10)]
    print('running ftrl experiments...')
    ftrl = [np.array(train_policy(FTRL_step, info)[1]) for _ in range(10)]
    print('running aftrl experiments...')
    aftrl = [np.array(train_policy(AFTRL_step, info)[1]) for _ in range(10)]
    print('running aoftrl experiments...')
    aoftrl = [np.array(train_policy(AOFTRL_step, info)[1]) for _ in range(10)]
    print('running saftrl experiments...')
    saftrl = [np.array(train_policy(SAFTRL_step, info)[1]) for _ in range(10)]
    print('running sftrl experiments...')
    sftrl = [np.array(train_policy(SFTRL_step, info)[1]) for _ in range(10)]
    print('running optimal experiments...')
    best = [np.array(train_policy(info.OPTIMAL_step, info)[1]) for _ in range(10)]
    print('running random experiments...')
    random = [np.array(train_policy(info.RANDOM_step, info)[1]) for _ in range(10)]
    # save em
    data = {'ftl':ftl, 'ogd': ogd, 'ftrl': ftrl, 'sftrl':sftrl, 'aftrl':aftrl,
            'aoftrl': aoftrl, 'best': best, 'saftrl': saftrl, 'random': random}
    torch.save(data, 'data/'+name+'.pt')
    return data

def plot_experiments(name='test_img'):
    # load data
    data = torch.load('data/'+name+'.pt')
    ftl, ogd, ftrl, aftrl, sftrl, aoftrl, best, saftrl = data['ftl'],data['ogd'],data['ftrl'],data['aftrl'],data['sftrl'],data['aoftrl'],data['best'],data['saftrl']
    # now plot it
    plot_loss(ogd, ftl, ftrl, aftrl, aoftrl, saftrl, best, name='plots/'+name)
    plot_cumuloss(ogd, ftl, ftrl, aftrl, aoftrl, saftrl, best, name='plots/'+'cumuloss_'+name)
    # make the log-scale version
    plot_loss(ogd, ftl, ftrl, aftrl, aoftrl, saftrl, best, name='plots/log_'+name, log_scale=True)
    plot_cumuloss(ogd, ftl, ftrl, aftrl, aoftrl, saftrl, best, name='plots/'+'log_cumuloss_'+name, log_scale=True)
