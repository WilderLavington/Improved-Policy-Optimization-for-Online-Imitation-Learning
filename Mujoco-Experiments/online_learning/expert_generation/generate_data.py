
#
import datetime
import gym
import numpy as np
import itertools
import torch
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import os

from cwm_lib.trainers.algos.bc import BC
from cwm_lib.trainers.memory.replay_memory import ReplayMemory
from cwm_lib.arg_parsers.generate_parser import get_args
from cwm_lib.trainers.utils import timer
from cwm_lib.trainers.memory.memory_utils import *

#
def check_args_bc(args):

    # presets
    env_name = args.env_name.replace('-','_')
    args.expert_params_path = args.expert_params_path+'sac_actor_'+env_name+'_expert'
    args.critic_sampler_type = None
    args.combined_update = None
    args.beta = torch.tensor(1.)
    args.beta_update = torch.tensor(1.)

    # me being a big dummy on my grid-search
    if args.use_log_lr:
        args.lr = np.exp(args.log_lr)
    args.init_step_size = np.exp(args.log_init_step_size)

    # set batch size
    args.n_batches_per_epoch = np.floor(args.replay_size / args.batch_size)

    # return em
    return args

#
def fill_buffer(args, agent, memory, env):
    # Training Loop
    total_numsteps = 0
    updates = 1
    start = time.time()
    # fill the buffer
    avg_reward = 0.
    episodes = 0
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        full_state = np.concatenate([deepcopy(env.sim.data.qpos), deepcopy(env.sim.data.qvel)])
        episode_reward = 0.
        while not done:
            action, log_prob = agent.select_action(state, evaluate=args.expert_mode, eval_expert=True)  # Sample action from policy
            next_state, reward, done, _ = env.step(action) # Step
            next_full_state = np.concatenate([deepcopy(env.sim.data.qpos), deepcopy(env.sim.data.qvel)])
            episode_steps += 1
            total_numsteps += 1
            mask = float(not done)
            # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask, log_prob,
                fstate=full_state, next_fstate=next_full_state, bad_mask=None) # Append transition to memory
            state = next_state
            full_state = deepcopy(next_full_state)
            episode_reward += reward
        avg_reward += episode_reward
        episodes += 1
        if total_numsteps > len(memory):
            break
    avg_reward /= episodes
    return memory, avg_reward

# update
def generate_data():

    start = time.time()

    args = get_args()

    args = check_args_bc(args)

    # make sure the expert params exist
    expert_params_dict = torch.load(args.expert_params_path)

    # Environment
    env = gym.make(args.env_name)
    args.env_copy = deepcopy(env)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = BC(env.observation_space.shape[0], env.action_space, args)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed, args)

    # get expert data
    memory, avg_reward = fill_buffer(args, agent, memory, env)

    # make sure dir exists
    try:
        os.makedirs(args.data_file_location)
    except FileExistsError:
        # directory already exists
        pass

    # now save the data loader
    memory.save_examples(args.data_file_location+"/expert_examples_"+args.env_name+'.pt')

    # make some plots
    data = torch.load(args.data_file_location+"/expert_examples_"+args.env_name+'.pt')

    # now plot the dist over state_error
    print('Generating expert k-lag distribution plots...')
    k_lag_coeffs = compute_buffer_autocorr(memory, k_lag=250)
    for dim in range(k_lag_coeffs.size()[1]):
        plt.plot(k_lag_coeffs[:,dim].numpy())
        plt.savefig('./plots/data_k_lag_'+str(dim)+'.png')
        plt.close()

    # now plot the dist over state_error
    print('Generating expert marginal distribution plots...')
    marginal_kdes = compute_buffer_marginal_kdes(memory)
    for dim in range(data['state'].shape[1]):
        plt.hist(data['state'][:,dim].numpy(), bins=25)
        plt.savefig('./plots/data_marginal_'+str(dim)+'.png')
        plt.close()

    print('Generating expert vizualization video...')
    visualize_dataset(env, memory, file_location='./videos/example.gif')

    print("----------------------------------------")
    print("Time-elapsed: {}, Data Avg. Reward: {}".format(timer(start,time.time()), round(avg_reward, 2)))
    print("----------------------------------------")


def load_data(env_name='HalfCheetah-v2', data_location='./data/expert_examples'):
    env_name = env_name.replace("-", "_")
    assert env_name in ['Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2']
    file_name = data_location + '/run_examples_'+env_name+'_.pt'
    data_dict = torch.load(file_name)
    return data_dict
