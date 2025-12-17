import jax
import gymnax
from gymnax.visualize import Visualizer
import jax.numpy as jnp

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
    n_envs = 100
    mem_capacity = 5_000
    env = MTorchCartpoleEnv(device, n_envs)
    n_steps = 10_000
    torch.manual_seed(seed)
    #env.action_space.seed(seed)
    #env.observation_space.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


    key = jax.random.key(0)
    key, key_reset, key_act, key_step = jax.random.split(key, 4)

    # Instantiate the environment & its settings.
    env, env_params = gymnax.make("CartPole-v1")

    # Reset the environment.
    obs, state = env.reset(key_reset, env_params)

    # Sample a random action.
    action = env.action_space(env_params).sample(key_act)

    # Perform the step transition.
    n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)
    print(n_obs)

    state_seq, reward_seq = [], []
    key, key_reset = jax.random.split(key)
    obs, env_state = env.reset(key_reset, env_params)
    while True:
        state_seq.append(env_state)
        key, key_act, key_step = jax.random.split(key, 3)
        action = env.action_space(env_params).sample(key_act)
        next_obs, next_env_state, reward, done, info = env.step(
            key_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        if done:
            break
        else:
            obs = next_obs
            env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate(f"anim.gif")
