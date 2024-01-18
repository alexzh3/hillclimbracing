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


# The base environment case, all observations and all discrete actions (0,1,2), 1000k timesteps, ppo
def exp_base():
    env = gym.make(env_id)
    env = Monitor(env, 'ppo_base', info_keywords=("score",))
    model = PPO("MultiInputPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=1_000_000)
    model.save("baseline_models/ppo_base")


#######################################################################################################################
# Discrete experiments, actions are all (0,1,2)
# Only chassis_position experiment
def exp_obs_position():
    env = gym.make(env_id)
    env = FilterObservation(env, filter_keys=['chassis_position'])
    env = Monitor(env, 'ppo_position', info_keywords=("score",))
    model = PPO("MultiInputPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=1_000_000)
    model.save("baseline_models/ppo_position")


# Only angle experiment
def exp_obs_angle():
    env = gym.make(env_id)
    env = FilterObservation(env, filter_keys=['chassis_angle'])
    env = Monitor(env, 'ppo_angle', info_keywords=("score",))
    model = PPO("MultiInputPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=1_000_000)
    model.save("baseline_models/ppo_angle")


# Only wheel_speed experiment
def exp_obs_speed():
    env = gym.make(env_id)
    env = FilterObservation(env, filter_keys=['wheels_speed'])
    env = Monitor(env, 'ppo_speed', info_keywords=("score",))
    model = PPO("MultiInputPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=1_000_000)
    model.save("baseline_models/ppo_speed")


# Only on ground experiment
def exp_obs_on_ground():
    env = gym.make(env_id)
    env = FilterObservation(env, filter_keys=['on_ground'])
    env = Monitor(env, 'ppo_on_ground', info_keywords=("score",))
    model = PPO("MultiInputPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=1_000_000)
    model.save("baseline_models/ppo_on_ground")


#######################################################################################################################
# Multi observation input experiments

# Position + angle
def exp_obs_pos_angle():
    env = gym.make(env_id)
    env = FilterObservation(env, filter_keys=['chassis_position', 'chassis_angle'])
    env = Monitor(env, 'ppo_position_angle', info_keywords=("score",))
    model = PPO("MultiInputPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=1_000_000)
    model.save("baseline_models/ppo_position_angle")


# Position + wheel speed
def exp_obs_pos_speed():
    env = gym.make(env_id)
    env = FilterObservation(env, filter_keys=['chassis_position', 'wheels_speed'])
    env = Monitor(env, 'ppo_position_speed', info_keywords=("score",))
    model = PPO("MultiInputPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=1_000_000)
    model.save("baseline_models/ppo_position_speed")


# Position + on_ground
def exp_obs_pos_ground():
    env = gym.make(env_id)
    env = FilterObservation(env, filter_keys=['chassis_position', 'on_ground'])
    env = Monitor(env, 'ppo_position_ground', info_keywords=("score",))
    model = PPO("MultiInputPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=1_000_000)
    model.save("baseline_models/ppo_position_ground")


# Position + angle + speed
def exp_obs_pos_angle_speed():
    env = gym.make(env_id)
    env = FilterObservation(env, filter_keys=['chassis_position', 'chassis_angle', 'wheels_speed'])
    env = Monitor(env, 'ppo_position_angle_speed', info_keywords=("score",))
    model = PPO("MultiInputPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=1_000_000)
    model.save("baseline_models/ppo_position_angle_speed")


#######################################################################################################################
# Action input experiments
def exp_action_discrete():
    ...


if __name__ == "__main__":
    exp_obs_pos_angle()
    exp_obs_pos_speed()
    exp_obs_pos_ground()
    exp_obs_pos_angle_speed()
    # model = PPO.load("baseline_models/ppo_base_1000k")
