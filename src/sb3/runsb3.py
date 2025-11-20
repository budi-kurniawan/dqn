import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
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
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def train_dqn(env, num_episodes: int, seed: int, device, draw_chart: bool = False):
    episode_durations = []
    env = gym.make("CartPole-v1")
    policy_kwargs = dict(
        net_arch=[128, 128], # 2 hidden layers with 128 units each
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-4)
    ) 
    model = DQN("MlpPolicy", env, 
                policy_kwargs=policy_kwargs, 
                buffer_size=10_000,
                train_freq=1,
                gradient_steps=1,
                device="cuda",
                verbose=2)
    model.learn(total_timesteps=50_000)
    vec_env = model.get_env()
    # for i_episode in range(num_episodes):
    #     count = 0

    #     obs = vec_env.reset()
    #     while True:
    #         count += 1
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, reward, done, info = vec_env.step(action)
    #         if done:
    #             obs = vec_env.reset()
    #             episode_durations.append(count)
    #             print("episode ", i_episode, ", reward: ", count)
    #             if draw_chart:
    #                 plot_durations(episode_durations)
    #             break

    if draw_chart:
        plot_durations(episode_durations, True)
        plt.ioff()
        plt.show()
    return episode_durations      


if __name__ == "__main__":
    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    #device = torch.device("cpu")
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
    

    print("device:", device)
    start = time.time()

    results = train_dqn(env, num_episodes, seed, device, draw_chart)

    end = time.time()
    print("Total rewards:", sum(results))
    print(f"Execution time: {end - start:.4f} seconds")
