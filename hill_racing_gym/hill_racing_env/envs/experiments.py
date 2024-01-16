import hill_racing_env
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import load_results
import pandas as pd


# Discrete experiments, actions are all (0,1,2)
def exp_obs_position(env):
    ...


if __name__ == "__main__":
    env_id = 'hill_racing_env/HillRacing-v0'
    num_cpu = 18
    vec_env = make_vec_env(env_id, n_envs=num_cpu, seed=1, monitor_dir="monitor", vec_env_cls=SubprocVecEnv,
                           env_kwargs={'render_mode': None})
    model = PPO("MultiInputPolicy", vec_env, verbose=1, seed=1)
    model.learn(total_timesteps=500_000)
    # model.save("baseline_models/ppo_base_500k")
    # model = PPO.load("baseline_models/ppo_base_500k")
    # obs = vec_env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = vec_env.step(action)
    #     print(info, rewards)
    #     vec_env.render("human")
