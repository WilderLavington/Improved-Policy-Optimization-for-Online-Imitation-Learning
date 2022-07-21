

import gym
from stable_baselines3 import PPO


def train_experts():

    # train hopper
    env = gym.make("Hopper-v2")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=6000000)
    model.save('./expert_models/ppo_actorcritic_Hopper_v2.pt')

    # train walker
    env = gym.make("HalfCheetah-v2")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=6000000)
    model.save('./expert_models/ppo_actorcritic_HalfCheetah_v2.pt')

    # train walker
    env = gym.make("Walker2d-v2")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=6000000)
    model.save('./expert_models/ppo_actorcritic_Walker2d_v2.pt')

if __name__ == "__main__":
    train_experts()
