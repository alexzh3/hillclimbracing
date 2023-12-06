from gymnasium.envs.registration import register

register(
     id="hillclimbracing/HillRacing-v0",
     entry_point="hillclimbracing.envs:HillRacingEnv",
     max_episode_steps=10000,
)