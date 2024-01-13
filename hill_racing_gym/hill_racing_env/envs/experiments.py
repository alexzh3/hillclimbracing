import hill_racing_env
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

env_id = 'hill_racing_env/HillRacing-v0'
num_cpu = 1
vec_env = make_vec_env(env_id, n_envs=num_cpu, seed=1, vec_env_cls=SubprocVecEnv,
                       env_kwargs={'render_mode': 'human'})
model = PPO("MultiInputPolicy", vec_env, verbose=1, seed=1)
model.load('baseline_models/ppo_50d_500ts')
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    print(info, rewards)
    vec_env.render("human")
