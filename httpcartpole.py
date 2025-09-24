import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import json
import requests


class HttpCartpoleEnv(CartPoleEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.info = {}
        self.base_url = "http://localhost:8080"
        self.steps = 0
        self.req_session = requests.Session()

    def reset(self, seed=None, options=None):
        if seed is not None:
            url = self.base_url + "/seed"
            params = {"seed": seed}
            response = requests.get(url, params=params)

        url = self.base_url + "/reset"
        params = {}
        response = requests.get(url, params=params)
        observation = json.loads(response.text)["state"]
        return observation, self.info

    def step(self, action: int):
        url = self.base_url + "/step"
        params = {"action": action}
        response = self.req_session.get(url, params=params)
        result = json.loads(response.text)
        self.steps += 1
        observation, reward, terminated, truncated = result["state"], result["reward"], result["terminated"], result["truncated"]
        if self.steps >= 500:
            print("truncated:", truncated, ", terminated:", terminated)
        if truncated or terminated:
            self.steps = 0
        return observation, reward, terminated, truncated, self.info



seed = 42
env = HttpCartpoleEnv()
obs, info = env.reset(seed=seed)
print("response from env.reset():", obs, info)

step_results = env.step(1)
print("step results:", step_results)