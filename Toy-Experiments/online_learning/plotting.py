
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

def plot_loss(ogd, ftl, ftrl, aftrl, aoftrl, saftrl, best, name='lossfig', log_scale=False):
    #
    # quantiles for plotting
    ogd_mean, ogd_max, ogd_min = np.mean(ogd, axis=0), np.quantile(ogd, 0.75, axis=0), np.quantile(ogd, 0.25, axis=0)
    ftl_mean, ftl_max, ftl_min = np.mean(ftl, axis=0), np.quantile(ftl, 0.75, axis=0), np.quantile(ftl, 0.25, axis=0)
    ftrl_mean, ftrl_max, ftrl_min = np.mean(ftrl, axis=0), np.quantile(ftrl, 0.75, axis=0), np.quantile(ftrl, 0.25, axis=0)
    aftrl_mean, aftrl_max, aftrl_min = np.mean(aftrl, axis=0), np.quantile(aftrl, 0.75, axis=0), np.quantile(aftrl, 0.25, axis=0)
    aoftrl_mean, aoftrl_max, aoftrl_min = np.mean(aoftrl, axis=0), np.quantile(aoftrl, 0.75, axis=0), np.quantile(aoftrl, 0.25, axis=0)
    saftrl_mean, saftrl_max, saftrl_min = np.mean(saftrl, axis=0), np.quantile(saftrl, 0.75, axis=0), np.quantile(saftrl, 0.25, axis=0)
    best_mean, best_max, best_min = np.mean(best, axis=0), np.quantile(best, 0.75, axis=0), np.quantile(best, 0.25, axis=0)
    #
    fig, ax = plt.subplots()
    ax.plot(ogd_mean, label='OGD')
    ax.fill_between(range(len(ogd_mean)), ogd_max, ogd_min, alpha = 0.5)
    ax.plot(ftl_mean, label='FTL')
    ax.fill_between(range(len(ftl_mean)), ftl_max, ftl_min, alpha = 0.5)
    ax.plot(ftrl_mean, label='FTRL')
    ax.fill_between(range(len(ftrl_mean)), ftrl_max, ftrl_min, alpha = 0.5)
    ax.plot(aftrl_mean, label='AFTRL')
    ax.fill_between(range(len(aftrl_mean)), aftrl_max, aftrl_min, alpha = 0.5)
    ax.plot(aoftrl_mean, label='AOFTRL')
    ax.fill_between(range(len(aoftrl_mean)), aoftrl_max, aoftrl_min, alpha = 0.5)
    ax.plot(saftrl_mean, label='SAFTRL')
    ax.fill_between(range(len(saftrl_mean)), saftrl_max, saftrl_min, alpha = 0.5)
    ax.plot(best_mean, label='BEST')
    ax.fill_between(range(len(ftrl_mean)), best_max, best_min, alpha = 0.5)
    ax.grid()
    plt.legend()
    plt.rcParams['figure.dpi'] = 400
    if log_scale:
        ax.set_yscale('log')
    # plt.show()
    plt.savefig(name+'.png')

def plot_cumuloss(ogd, ftl, ftrl, aftrl, aoftrl, saftrl, best, name='cumulossfig', log_scale=False):
    #
    ogd_ = np.cumsum(np.array(ogd), axis=1)
    ftl_ = np.cumsum(np.array(ftl), axis=1)
    ftrl_ = np.cumsum(np.array(ftrl), axis=1)
    aftrl_ = np.cumsum(np.array(aftrl), axis=1)
    aoftrl_ = np.cumsum(np.array(aoftrl), axis=1)
    saftrl_ = np.cumsum(np.array(saftrl), axis=1)
    best_ = np.cumsum(np.array(best), axis=1)
    #
    ogd_mean_cs, ogd_max, ogd_min = np.mean(ogd_, axis=0), \
                                    np.quantile(ogd_, 0.75, axis=0), \
                                    np.quantile(ogd_, 0.25, axis=0)
    ftl_mean_cs, ftl_max, ftl_min = np.mean(ftl_, axis=0), \
                                    np.quantile(ftl_, 0.75, axis=0), \
                                    np.quantile(ftl_, 0.25, axis=0)
    ftrl_mean_cs, ftrl_max, ftrl_min = np.mean(ftrl_, axis=0), \
                                       np.quantile(ftrl_, 0.75, axis=0), \
                                       np.quantile(ftrl_, 0.25, axis=0)
    aftrl_mean_cs, aftrl_max, aftrl_min = np.mean(aftrl_, axis=0), \
                                       np.quantile(aftrl_, 0.75, axis=0), \
                                       np.quantile(aftrl_, 0.25, axis=0)
    aoftrl_mean_cs, aoftrl_max, aoftrl_min = np.mean(aoftrl_, axis=0), \
                                           np.quantile(aoftrl_, 0.75, axis=0), \
                                           np.quantile(aoftrl_, 0.25, axis=0)
    saftrl_mean_cs, saftrl_max, saftrl_min = np.mean(saftrl_, axis=0), \
                                          np.quantile(saftrl_, 0.75, axis=0), \
                                          np.quantile(saftrl_, 0.25, axis=0)
    best_mean_cs, best_max, best_min = np.mean(best_, axis=0), \
                                           np.quantile(best_, 0.75, axis=0), \
                                           np.quantile(best_, 0.25, axis=0)
    # print(np.array(ogd) )
    fig, ax = plt.subplots()
    ax.plot(ogd_mean_cs, label='OGD')
    ax.fill_between(range(len(ogd_mean_cs)), ogd_max, ogd_min, alpha = 0.5)
    ax.plot(ftl_mean_cs, label='FTL')
    ax.fill_between(range(len(ogd_mean_cs)), ftl_max, ftl_min, alpha = 0.5)
    ax.plot(ftrl_mean_cs, label='FTRL')
    ax.fill_between(range(len(ogd_mean_cs)), ftrl_max, ftrl_min, alpha = 0.5)
    ax.plot(aftrl_mean_cs, label='AFTRL')
    ax.fill_between(range(len(aftrl_mean_cs)), aftrl_max, aftrl_min, alpha = 0.5)
    ax.plot(aoftrl_mean_cs, label='AOFTRL')
    ax.fill_between(range(len(ogd_mean_cs)), aoftrl_max, aoftrl_min, alpha = 0.5)
    ax.plot(saftrl_mean_cs, label='SAFTRL')
    ax.fill_between(range(len(ogd_mean_cs)), saftrl_max, saftrl_min, alpha = 0.5)
    ax.plot(best_mean_cs, label='BEST')
    ax.fill_between(range(len(ogd_mean_cs)), best_max, best_min, alpha = 0.5)
    ax.grid()
    plt.legend()
    plt.rcParams['figure.dpi'] = 400
    if log_scale:
        ax.set_yscale('log')
    plt.savefig(name+'.png')
    # plt.show()
