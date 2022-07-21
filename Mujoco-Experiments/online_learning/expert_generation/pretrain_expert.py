
import datetime
import gym
import numpy as np
import itertools
import torch
import wandb

from online_learning.expert_generation.pretrain_parser import get_args
from online_learning.expert_generation.sac import SAC
from online_learning.expert_generation.memory.replay_memory import ReplayMemory

def initialization(args):
    # Environment
    env = gym.make(args.env_name)
    # set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.combined_update = 1
    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    # Memory
    memory = ReplayMemory(args.replay_size, args.seed, args)
    # init project
    wandb.init(project=args.project, group='pre_train_'+args.env_name)
    # return
    return args, agent, memory, env

def step_trainer(args, agent, memory, updates):
    # Number of updates per step in environment
    for i in range(args.updates_per_step):
        # Update parameters of all the networks
        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
    # return las losses
    return critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha

def generate_action(args, agent, env, state, total_numsteps):
    if args.start_steps > total_numsteps:
        action = env.action_space.sample()  # Sample random action
        log_prob = np.log(1/env.action_space.shape[0])
    else:
        action, log_prob = agent.select_action(state)  # Sample action from policy
    return action, log_prob

def evaluate_return(agent, env, duplicates=10):
    avg_reward = 0.
    for _  in range(duplicates):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            tensor_state = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
            action, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
    avg_reward /= duplicates
    return avg_reward

def train_agent(args, agent, memory, log_interval=10):
    # init
    total_numsteps = 0
    updates = 0
    best_return = evaluate_return(agent, gym.make(args.env_name), duplicates=15)
    env = gym.make(args.env_name)
    # train loop
    for i_episode in itertools.count(1):
        # episode inits
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        # loop for one full trajectory
        while not done:
            # get action
            action, log_prob = generate_action(args, agent, env, state, total_numsteps)
            # step agent
            if len(memory) > args.batch_size:
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = step_trainer(args, agent, memory, updates)
                updates += args.updates_per_step
            # step environment
            next_state, reward, done, _ = env.step(action) # Step
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            # update logged stats
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            # add example to buffer
            memory.push(state, action, reward, next_state, mask, log_prob) # Append transition to memory
            # update state
            state = next_state
        # log and eval
        if i_episode % log_interval == 0 and args.eval is True:
            # run eval
            avg_reward = evaluate_return(agent, gym.make(args.env_name), duplicates=15)
            # display
            print("----------------------------------------")
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}"\
                .format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
            print("Test Episodes: {}, Avg. Reward: {}".format(i_episode, round(avg_reward, 2)))
            print("----------------------------------------")
            # save expert
            if avg_reward > best_return:
                best_return = avg_reward
                print('new best average reward: ', avg_reward)
                agent.save_model(env_name=args.env_name, suffix="expert")
            # log
            if len(memory) > args.batch_size:
                wandb.log({'loss_critic_1': critic_1_loss, 'loss_critic_2': critic_2_loss,
                'loss_policy': policy_loss, 'loss_entrop': ent_loss, 'alpha': alpha,
                'stoc_avg_reward': round(episode_reward, 2), 'time_steps': len(memory),
                'det_avg_reward': round(avg_reward, 2)}, step=updates)
        # stopping conditions
        if total_numsteps > args.num_steps:
            env.close()
            break

def pretrain_expert_policy():
    args = get_args()
    args, agent, memory, env = initialization(args)
    train_agent(args, agent, memory, log_interval=10)

if __name__ == "__main__":
    pretrain_expert_policy()
