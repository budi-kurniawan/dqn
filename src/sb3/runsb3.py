import gymnasium as gym
import random
from itertools import count
import torch
import os
import time
from stable_baselines3 import DQN


"""
Setting amsgrad=True in the AdamW optimizer makes the results reproducible and better.
"""
os.system('cls' if os.name == 'nt' else 'clear')


def train_dqn(env, num_episodes: int, seed: int, device, draw_chart: bool = False):
    episode_durations = []


if __name__ == "__main__":
    seed = 42
    env = gym.make("CartPole-v1")
    random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state) #4
    num_episodes = 500
    draw_chart = False
    

    policy_kwargs = dict(
        net_arch=[128, 128], # 2 hidden layers with 128 units each
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-4)
    ) 

    model = DQN("MlpPolicy", env, 
                policy_kwargs=policy_kwargs, 
                buffer_size=10_000,
                batch_size=128,
                train_freq=1,
                gradient_steps=-1,
                target_update_interval=1,
                tau=0.005,
                gamma=0.99,
                learning_rate=3e-4,
                device="cpu",
                verbose=1)

    start = time.time()
    model.learn(total_timesteps=50_000)

    end = time.time()
    print(f"Execution time: {end - start:.4f} seconds")
