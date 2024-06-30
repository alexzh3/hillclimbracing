import hill_racing_env
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor

from gymnasium.wrappers import FilterObservation
import pandas as pd
from stable_baselines3.common.callbacks import ProgressBarCallback

env_id = 'hill_racing_env/HillRacing-v0'


# Different reward functions tested with observations with discrete actions (0,1,2) (base)

# The base environment case, all observations and all discrete actions (0,1,2),reward type distance
def exp_base_reward_distance(runs, reward_type):
    for i in range(runs):
        env = gym.make(env_id, reward_type=reward_type)
        env = Monitor(env, f'ppo_base_2_{reward_type}_1000_{i}', info_keywords=("score", "total_airtime"))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_base_2_{reward_type}_1000_{i}")


# The base environment case, all observations and all discrete actions (0,1,2), reward type action
def exp_base_reward_action(runs, reward_type):
    for i in range(runs):
        env = gym.make(env_id, reward_function="action", reward_type=reward_type)
        env = Monitor(env, f'ppo_base_action_diff_increasing_{reward_type}_1000_{i}', info_keywords=("score",))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_base_action_diff_increasing_{reward_type}_1000_{i}")


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
        env = Monitor(env, f'ppo_cont_2_{reward_type}_1000_{i}', info_keywords=("score", "total_airtime"))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_cont_2_{reward_type}_1000_{i}")


# Continuous, all observations and all discrete actions (0,1,2), reward type wheel speed
def exp_cont_reward_wheel_speed(runs, reward_type):
    for i in range(3, 5):
        env = gym.make(env_id, reward_function="wheel_speed", reward_type=reward_type, action_space="continuous")
        env = Monitor(env, f'ppo_cont_wheel_speed_2_{reward_type}_1000_{i}', info_keywords=("score", "total_airtime"))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_cont_wheel_speed_2_{reward_type}_1000_{i}")


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

# Airtime rewards

# Continuous distance
def exp_cont_reward_airtime_distance(runs, reward_type):
    for i in range(runs):
        env = gym.make(env_id, reward_function="airtime_distance", reward_type=reward_type, action_space="continuous")
        env = Monitor(env, f'ppo_cont_airtime_{reward_type}_1000_{i}', info_keywords=("score", "total_airtime"))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_cont_airtime_{reward_type}_1000_{i}")


# Continuous wheel speed
def exp_cont_reward_airtime_wheel_speed(runs, reward_type):
    for i in range(runs):
        env = gym.make(env_id, reward_function="airtime_wheel_speed", reward_type=reward_type,
                       action_space="continuous")
        env = Monitor(env, f'ppo_cont_wheel_speed_airtime_{reward_type}_1000_{i}',
                      info_keywords=("score", "total_airtime"))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_cont_wheel_speed_airtime_{reward_type}_1000_{i}")


# Discrete distance
def exp_base_reward_airtime_distance(runs, reward_type):
    for i in range(runs):
        env = gym.make(env_id, reward_function="airtime_distance", reward_type=reward_type)
        env = Monitor(env, f'ppo_base_airtime_{reward_type}_1000_{i}', info_keywords=("score", "total_airtime"))
        model = PPO("MultiInputPolicy", env, verbose=1, seed=i)
        model.learn(total_timesteps=1_000_000)
        model.save(f"baseline_models/ppo_base_airtime_{reward_type}_1000_{i}")


#######################################################################################################################
# Evaluation of best models
def eval_model(model_path, monitor_name, action_space, reward_type, reward_function, episodes):
    # env = gym.make(env_id, action_space=action_space, reward_type=reward_type,
    #                reward_function=reward_function)
    vec_env = make_vec_env(env_id, n_envs=20, seed=0, monitor_dir=monitor_name, vec_env_cls=SubprocVecEnv,
                           env_kwargs={'render_mode': None,
                                       'action_space': action_space,
                                       'reward_type': reward_type,
                                       'reward_function': reward_function},
                           monitor_kwargs={'info_keywords': ("score", "position_list")})
    # vec_env = VecMonitor(venv=vec_env, filename=monitor_name, info_keywords=("score",))
    # env = Monitor(env, monitor_name, info_keywords=("score", ))
    model = PPO.load(model_path, env=vec_env,
                     custom_objects={'observation_space': vec_env.observation_space,
                                     'action_space': vec_env.action_space})
    evaluate_policy(model=model, env=vec_env, n_eval_episodes=episodes)


#######################################################################################################################
# Random agent experiment (for control)
def random_model(runs, reward_type):
    for i in range(runs):
        env = gym.make(env_id, reward_function="action", action_space="discrete_3", reward_type=reward_type)
        env = Monitor(env, f'rando\m_agent_{reward_type}_1000_{i}', info_keywords=("score",))
        # Initialize variables
        current_timesteps = 0
        num_timesteps = 1_000_000
        # Run the random agent
        while current_timesteps < num_timesteps:
            obs = env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                # Select a random action
                action = env.action_space.sample()
                # Take the action in the environment
                obs, reward, terminated, truncated, info = env.step(action)
                current_timesteps += 1
            print(f"Run: {i}, Timesteps: {current_timesteps}")


#######################################################################################################################

if __name__ == "__main__":
    # Evaluation experiments

    # # Evaluation of Continuous action space with wheel speed based 1000 aggressive -150 difficulty
    # eval_model(model_path="baseline_models/ppo_cont_wheel_speed_aggressive_1000_0.zip",
    #            monitor_name="eval_ppo_cont_wheel_speed_aggressive_1000", action_space="continuous",
    #            reward_type="aggressive", reward_function="wheel_speed", episodes=100)
    #
    # # Evaluation of Continuous action space with wheel speed based rewards 1000 aggressive -150 difficulty
    # eval_model(model_path="baseline_models/ppo_cont_wheel_speed_aggressive_1000_0.zip",
    #            monitor_name="eval_ppo_cont_wheel_speed_aggressive_1000", action_space="continuous",
    #            reward_type="aggressive", reward_function="wheel_speed", episodes=1000)
    #
    # # Evaluation of Continuous action space with distance based rewards 1000 soft -150 difficulty
    # eval_model(model_path="baseline_models/ppo_cont_soft_1000_3.zip",
    #            monitor_name="eval_ppo_cont_distance_soft_1000", action_space="continuous",
    #            reward_type="soft", reward_function="distance", episodes=1000)
    # # Evaluation of discrete action space with distance based rewards 1000 soft -150 difficulty
    # eval_model(model_path="baseline_models/ppo_base_soft_1000_4.zip",
    #            monitor_name="eval_ppo_base_distance_soft_1000", action_space="discrete_3",
    #            reward_type="soft", reward_function="distance", episodes=1000)
    #
    # # Evaluation of discrete action space with action based rewards 1000 soft -150 difficulty
    # eval_model(model_path="baseline_models/ppo_base_action_soft_1000_4.zip",
    #            monitor_name="eval_ppo_discrete_action_soft_1000", action_space="discrete_3",
    #            reward_type="soft", reward_function="action", episodes=1000)

    # EXPERIMENTS TBD FOR AIRTIME VS ORIGINAL COMPARISON, THESE ARE ORIGINAL BUT WITH AIRTIME MONITORED
    # # Continuous action space with distance based rewards, 1000 soft, -150 difficulty
    # exp_cont_reward_distance(5, "soft")
    # Discrete action space with distance based rewards, 1000 soft, -150 difficulty
    # exp_base_reward_distance(5, "soft")
    # Continuous action space with wheel-speed based rewards, 1000 aggressive, -150 difficulty
    # exp_cont_reward_wheel_speed(5, "aggressive")

    # # Continuous action space with wheel speed based rewards  1000 aggressive, increasing difficulty, -184 difficulty
    # exp_cont_reward_wheel_speed(5, "aggressive")
    # # Continuous action space with distance-based rewards 1000 soft, increasing difficulty, -184 difficulty
    # exp_cont_reward_distance(5, "soft")
    # # Discrete action space with distance-based rewards, 1000 soft, increasing difficulty, -184 difficulty
    # exp_base_reward_distance(5, "soft")
    # # Discrete action space with action-based rewards, 1000 soft, increasing difficulty, -184 difficulty
    # exp_base_reward_action(5, "soft")

    # # Continuous action space with wheel speed based and airtime rewards 1000 aggressive -150 difficulty
    # exp_cont_reward_airtime_wheel_speed(5, "aggressive")
    # # Continuous action space with distance-based and airtime rewards 1000 soft -150 difficulty
    # exp_cont_reward_airtime_distance(5, "soft")
    # # Discrete action space with distance-based and airtime rewards 1000 soft -150 difficulty
    # exp_base_reward_airtime_distance(5, "soft")
    # random_model(5, "soft")
    # random_model(5, "aggressive")
