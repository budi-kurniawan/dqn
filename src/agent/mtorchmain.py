from itertools import count
import torch

from agent.mtorchdqn.mtorch_dqn_agent import MTorchDQNAgent
from env.mcartpole.mtorch_cartpole_env import MTorchCartpoleEnv
from util.plot_util import plot_timesteps
import os
import time

os.system('cls' if os.name == 'nt' else 'clear')

def train_dqn(env, agent, num_episodes: int, device, n_envs):
    episode_durations = []

    n_steps = 10_000

    state = env.reset() #shape(n_envs, n_observations)
    for i_step in range(n_steps):
        #state = state[0] # TODO, select_action needs to take 2-dim tensor
        action = agent.select_action(state[0])
        action = action.repeat(n_envs) #TODO, remove this
        next_state, reward, terminated, truncated = env.step(action)
        agent.update(state, action, next_state, reward, terminated, truncated)
        # if torch.logical_or(terminated, truncated): #TODO move this to env.step()
        #     break
        print("step:", i_step)
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
    n_envs = 2
    env = MTorchCartpoleEnv(device, n_envs)
    torch.manual_seed(seed)
    #env.action_space.seed(seed)
    #env.observation_space.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    n_actions = env.action_space.n
    state = env.reset()
    n_observations = len(state[0]) #4, state is 2-dim, so we need the length of a row
    num_episodes = 10
    dqn_agent = MTorchDQNAgent(n_observations, n_actions, env, device, n_envs)
    

    print("device:", device)
    start = time.time()
    results = train_dqn(env, dqn_agent, num_episodes, device, n_envs)
    end = time.time()
    print("Total rewards:", sum(results))
    print(f"Execution time: {end - start:.4f} seconds")
    draw_chart = False
    if draw_chart:
        plot_timesteps(results, True)
