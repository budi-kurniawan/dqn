import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import random
from itertools import count

import torch

from agent.fastdqn.dqn_agent import DQNAgent
import os
import time


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

def train_dqn(env, dqn_agent, num_episodes: int, seed: int, device, draw_chart: bool = False):
    episode_durations = []

    for i_episode in range(num_episodes):
        state, _ = env.reset() # observation is tuple [4]
        for t in count():
            action = dqn_agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # next_state = numpy.ndarray, reward: float
            dqn_agent.update(state, action, next_state, reward, terminated, truncated)
            if terminated or truncated:
                episode_durations.append(t + 1)
                print("episode ", i_episode, ", reward: ", t + 1)
                if draw_chart:
                    plot_durations(episode_durations)
                break
            state = next_state
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
    dqn_agent = DQNAgent(n_observations, n_actions, env, device)
    draw_chart = False
    

    print("device:", device)
    start = time.time()

    results = train_dqn(env, dqn_agent, num_episodes, seed, device, draw_chart)

    end = time.time()
    print("Total rewards:", sum(results))
    print(f"Execution time: {end - start:.4f} seconds")
