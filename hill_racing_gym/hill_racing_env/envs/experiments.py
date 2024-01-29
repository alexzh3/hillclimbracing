import hill_racing_env
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import load_results, Monitor
from gymnasium.wrappers import FilterObservation
import pandas as pd

env_id = 'hill_racing_env/HillRacing-v0'


# The base environment case, all observations and all discrete actions (0,1,2),reward type distance
def exp_base(runs):
    for i in range(runs):
        env = gym.make(env_id)
        env = Monitor(env, f'ppo_base_soft_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_base_soft_{i}")


# The base environment case, all observations and all discrete actions (0,1,2), reward type action
def exp_base_reward_action(runs):
    for i in range(runs):
        env = gym.make(env_id, reward_type="action")
        env = Monitor(env, f'ppo_base_action_soft{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_base_action_soft{i}")


# The base environment case, all observations and all discrete actions (0,1,2), reward type wheel speed
def exp_base_reward_wheel_speed(runs):
    for i in range(runs):
        env = gym.make(env_id, reward_type="wheel_speed")
        env = Monitor(env, f'ppo_base_wheel_speed_soft_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_base_wheel_speed_soft_{i}")


#######################################################################################################################
# Action input experiments

# Only gas and reverse actions, all observations
def exp_action_discrete_2():
    env = gym.make(env_id, action_space_type="discrete_2")
    env = Monitor(env, 'ppo_action_discrete_2', info_keywords=("score",))
    model = PPO("MultiInputPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=1_000_000)
    model.save("baseline_models/ppo_action_discrete_2")


# Continuous motor speed as actions, all observations
def exp_action_continuous():
    env = gym.make(env_id, action_space_type="continuous")
    env = Monitor(env, 'ppo_action_continuous', info_keywords=("score",))
    model = PPO("MultiInputPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=1_000_000)
    model.save("baseline_models/ppo_action_continuous")


#######################################################################################################################
if __name__ == "__main__":
    # base do 5 runs
    # exp_base(5)
    # exp_base_reward_action(5)
    exp_base_reward_wheel_speed(5)
    # exp_base_reward_wheel_speed(5)
    # model = PPO.load("baseline_models/ppo_base_1000k")

