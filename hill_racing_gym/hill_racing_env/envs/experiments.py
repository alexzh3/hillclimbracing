import hill_racing_env
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import load_results, Monitor
from gymnasium.wrappers import FilterObservation
import pandas as pd

env_id = 'hill_racing_env/HillRacing-v0'


# Different reward functions tested with observations with discrete actions (0,1,2) (base)

# The base environment case, all observations and all discrete actions (0,1,2),reward type distance
def exp_base_reward_distance(runs, reward_type):
    for i in range(runs):
        env = gym.make(env_id, reward_type=reward_type)
        env = Monitor(env, f'ppo_base_diff_increasing_{reward_type}_1000_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_base_diff_increasing_{reward_type}_1000_{i}")


# The base environment case, all observations and all discrete actions (0,1,2), reward type action
def exp_base_reward_action(runs, reward_type):
    for i in range(runs):
        env = gym.make(env_id, reward_function="action", reward_type=reward_type)
        env = Monitor(env, f'ppo_base_action_{reward_type}_1000_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_base_action_{reward_type}_1000_{i}")


# The base environment case, all observations and all discrete actions (0,1,2), reward type wheel speed
def exp_base_reward_wheel_speed(runs, reward_type):
    for i in range(runs):
        env = gym.make(env_id, reward_function="wheel_speed", reward_type=reward_type)
        env = Monitor(env, f'ppo_base_wheel_speed_{reward_type}_1000_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_base_wheel_speed_{reward_type}_1000_{i}")


#######################################################################################################################

# Different reward functions tested with observations with continuous actions (0,1,2) (cont)

# Continuous, all observations and all discrete actions (0,1,2),reward type distance
def exp_cont_reward_distance(runs, reward_type):
    for i in range(runs):
        env = gym.make(env_id, action_space="continuous", reward_type=reward_type)
        env = Monitor(env, f'ppo_cont_{reward_type}_1000_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_cont_{reward_type}_1000_{i}")


# Continuous, all observations and all discrete actions (0,1,2), reward type wheel speed
def exp_cont_reward_wheel_speed(runs, reward_type):
    for i in range(runs):
        env = gym.make(env_id, reward_function="wheel_speed", reward_type=reward_type, action_space="continuous")
        env = Monitor(env, f'ppo_cont_wheel_speed_diff_increasing_{reward_type}_1000_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_cont_wheel_speed_diff_increasing_{reward_type}_1000_{i}")


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
    # Continuous action space with wheel speed based rewards  1000 aggressive, increasing difficulty, -180 difficulty
    exp_cont_reward_wheel_speed(5, "aggressive")
    # Discrete action space with distance-based rewards, 1000 soft, increasing difficulty, -180 difficulty
    exp_base_reward_distance(5, "soft")
