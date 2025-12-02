import random
from itertools import count

import torch

from agent.dqn.dqn_agent import DQNAgent
from util.plot_util import plot_timesteps
import os
import time

os.system('cls' if os.name == 'nt' else 'clear')

def train_dqn(env, dqn_agent, num_episodes: int, seed: int, device):
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
                print("episode ", i_episode + 1, ", reward: ", t + 1)
                break
            state = next_state
    return episode_durations      


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    #device = torch.device("cpu")
    seed = 42
    from env.cartpole.adapted_cartpole_env import AdaptedCartpoleEnv
    env = AdaptedCartpoleEnv(seed, device)
    random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    #env.action_space.seed(seed)
    #env.observation_space.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state) #4
    num_episodes = 500
    dqn_agent = DQNAgent(n_observations, n_actions, env, device)
    

    print("device:", device)
    start = time.time()
    results = train_dqn(env, dqn_agent, num_episodes, seed, device)
    end = time.time()
    print("Total rewards:", sum(results))
    print(f"Execution time: {end - start:.4f} seconds")
    draw_chart = False
    if draw_chart:
        plot_timesteps(results, True)
