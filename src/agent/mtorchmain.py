from itertools import count
import torch

from agent.mtorchdqn.torch_dqn_agent import TorchDQNAgent
from env.mcartpole.cartpole_env import CartpoleEnv
from util.plot_util import plot_timesteps
import os
import time

os.system('cls' if os.name == 'nt' else 'clear')

def train_dqn(env, dqn_agent, num_episodes: int, seed: int, device):
    episode_durations = []

    for i_episode in range(num_episodes):
        state = env.reset()
        for t in count():
            action = dqn_agent.select_action(state)
            next_state, reward, terminated, truncated = env.step(action)
            # next_state: shape([4]), reward: shape([1])
            dqn_agent.update(state, action, next_state, reward, terminated, truncated)
            if torch.logical_or(terminated, truncated):
                episode_durations.append(t + 1)
                print("episode ", i_episode + 1, ", reward: ", t + 1)
                break
            state = next_state.clone() # must clone, else state and next_state will point to same tensor in agent.update()
    return episode_durations      


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    #device = torch.device("cpu")
    seed = 1
    env = CartpoleEnv(seed, device)
    torch.manual_seed(seed)
    #env.action_space.seed(seed)
    #env.observation_space.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    n_actions = env.action_space.n
    state = env.reset()
    n_observations = len(state) #4
    num_episodes = 500
    dqn_agent = TorchDQNAgent(n_observations, n_actions, env, device)
    

    print("device:", device)
    start = time.time()
    results = train_dqn(env, dqn_agent, num_episodes, seed, device)
    end = time.time()
    print("Total rewards:", sum(results))
    print(f"Execution time: {end - start:.4f} seconds")
    draw_chart = False
    if draw_chart:
        plot_timesteps(results, True)
