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


# Different reward functions tested with observations with discrete actions (0,1,2) (base)

# The base environment case, all observations and all discrete actions (0,1,2),reward type distance
def exp_base_reward_distance(runs):
    for i in range(runs):
        env = gym.make(env_id)
        env = Monitor(env, f'ppo_base_soft_300_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_base_soft_300_{i}")


# The base environment case, all observations and all discrete actions (0,1,2), reward type action
def exp_base_reward_action(runs):
    for i in range(runs):
        env = gym.make(env_id, reward_type="action")
        env = Monitor(env, f'ppo_base_action_soft_300_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_base_action_soft_300_{i}")


# The base environment case, all observations and all discrete actions (0,1,2), reward type wheel speed
def exp_base_reward_wheel_speed(runs):
    for i in range(runs):
        env = gym.make(env_id, reward_type="wheel_speed")
        env = Monitor(env, f'ppo_base_wheel_speed_soft_1000_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_base_wheel_speed_soft_1000_{i}")


#######################################################################################################################

# Different reward functions tested with observations with continuous actions (0,1,2) (cont)

# Continuous, all observations and all discrete actions (0,1,2),reward type distance
def exp_cont_reward_distance(runs):
    for i in range(runs):
        env = gym.make(env_id, action_space="continuous")
        env = Monitor(env, f'ppo_cont_soft_300_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_cont_soft_300_{i}")


# Continuous, all observations and all discrete actions (0,1,2), reward type wheel speed
def exp_cont_reward_wheel_speed(runs):
    for i in range(runs):
        env = gym.make(env_id, reward_type="wheel_speed", action_space="continuous")
        env = Monitor(env, f'ppo_cont_wheel_speed_soft_300_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_cont_wheel_speed_soft_300_{i}")


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
    # Discrete(3)/base experiments
    # exp_base_reward_distance(5)
    # exp_base_reward_action(5)
    # exp_base_reward_wheel_speed(5)
    # Continuous experiments
    # exp_cont_reward_distance(5)
    # exp_cont_reward_wheel_speed(5)
    # TBD experiment with 1000 for wheel speed soft to check whether the soft vs aggressive curve in 300 is correct
    exp_base_reward_wheel_speed(5)

