from itertools import count
import torch

from agent.mtorchdqn.mtorch_dqn_agent import MTorchDQNAgent
from env.mcartpole.mtorch_cartpole_env import MTorchCartpoleEnv
from util.plot_util import plot_timesteps, plot_simple
import os
import time

os.system('cls' if os.name == 'nt' else 'clear')

# TODO use pinned memory to optimise
def train_dqn(env, agent, n_steps):
    state = env.reset() #shape(n_envs, n_observations)
    for i_step in range(n_steps):
        action = agent.select_action(state)
        # state already reset on terminal either at the first iteration or in env.step()
        # do not call reset() in this for loop
        next_state, reward, terminated, truncated = env.step(action)
        agent.update(state, action, next_state, reward, terminated, truncated)
        state = next_state.clone() # must clone next_state, else state and next_state will point to same tensor in agent.update()
    return env.get_rewards()      


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    #device = torch.device("cpu")
    seed = 1
    n_envs = 20
    mem_capacity = 1000
    batch_size = 32
    n_steps = 6_000

    env = MTorchCartpoleEnv(device, n_envs)
    torch.manual_seed(seed)
    #env.action_space.seed(seed)
    #env.observation_space.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    n_actions = env.action_space.n
    state = env.reset()
    n_observations = len(state[0]) #4, state is 2-dim, so we need the length of a row
    dqn_agent = MTorchDQNAgent(n_observations, n_actions, env, device, n_envs, mem_capacity, batch_size)
    print("device:", device)
    start = time.time()
    results = train_dqn(env, dqn_agent, n_steps)
    end = time.time()
    print("#episodes:", len(results))
    print(results)
    print("Total rewards:", sum(results))
    print(f"Execution time: {end - start:.4f} seconds")
    draw_chart = True
    if draw_chart:
        plot_simple(results.cpu())
