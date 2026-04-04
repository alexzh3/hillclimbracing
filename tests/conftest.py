import os
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))


@pytest.fixture
def discrete_env():
    """Create a discrete-action env, yield it, then close."""
    import gymnasium as gym
    import hill_racing_env  # noqa: F401 -- triggers registration

    env = gym.make(
        "hill_racing_env/HillRacing-v0",
        action_space="discrete_3",
        reward_function="distance",
        reward_type="aggressive",
    )
    yield env
    env.close()


@pytest.fixture
def continuous_env():
    """Create a continuous-action env, yield it, then close."""
    import gymnasium as gym
    import hill_racing_env  # noqa: F401

    env = gym.make(
        "hill_racing_env/HillRacing-v0",
        action_space="continuous",
        reward_function="distance",
        reward_type="soft",
    )
    yield env
    env.close()
