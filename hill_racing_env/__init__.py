from gymnasium.envs.registration import register

register(
    id="hill_racing_env/HillRacing-v0",
    entry_point="hill_racing_env.envs:HillRacingEnv",
)
