import hill_racing_env
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import load_results, Monitor
from gymnasium.wrappers import FilterObservation
import pandas as pd
import numpy as np
import gymnasium as gym
import noise
import random

env_id = 'hill_racing_env/HillRacing-v0'


class RemoveIdleAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Set the new action space to be discrete(2) with options 1 and 2
        self.action_space = gym.spaces.Discrete(n=2, start=1)

    def action(self, action):
        return action


class ChangeActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Set the new action space to be discrete(2) with options 1 and 2
        self.action_space = gym.spaces.Box(low=-13, high=13, shape=(1,), dtype=np.float32)

    def action(self, action):
        return action


# Function to test environments
def test_env(env):
    episodes = 10
    print(f"Testing {episodes} episodes with random samples")
    for episode in range(1, episodes + 1):
        state = env.reset(seed=1)
        done = False
        truncated = False
        score = 0
        while not done and not truncated:
            env.render()
            action = 0
            obs, reward, done, truncated, info = env.step(action)
            print(info, score)
            score += reward


# Function that tests models
def test_model(model):
    vec_env = model.get_env()
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        print(rewards, info)
        vec_env.render("human")


if __name__ == "__main__":
    env = gym.make(env_id, render_mode="human", action_space="discrete_3", reward_type="distance")
    test_env(env)
    model = PPO.load("baseline_models/ppo_base_300_3.zip", env=env,
                     custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    # test_model(model)
    # flatLength = 500
    # distance = 100_000
    # ground_seed = random.uniform(0, 100000)  # Generates a random seed that will define the terrain
    # for i in range(0, distance, 15):
    #     steepness_Level = np.interp(i, [0, distance], [130, 250])
    #     noise_value = ground_seed + (i - flatLength) / (700 - steepness_Level)
    #     noisedY = noise.pnoise1(ground_seed + (i - flatLength) / (700 - steepness_Level), octaves=4,
    #                             lacunarity=2)
    #     maxHeight = -150 + np.interp(steepness_Level, [0, 200], [0, 320])
    #     print(f"i: {i}, steepness_Level: {steepness_Level}, noisedY:{noisedY}, maxHeight: {maxHeight}")
