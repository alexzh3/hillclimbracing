"""Tests for environment creation, reset, and step."""

import pytest
import gymnasium as gym
import hill_racing_env  # noqa: F401


ENV_ID = "hill_racing_env/HillRacing-v0"

ACTION_SPACES = ["discrete_3", "continuous"]
REWARD_FUNCTIONS = ["distance", "action", "wheel_speed", "airtime_distance", "airtime_wheel_speed"]
REWARD_TYPES = ["aggressive", "soft"]


class TestEnvCreation:
    """Verify environment can be created with all valid configurations."""

    @pytest.mark.parametrize("action_space", ACTION_SPACES)
    def test_create_with_action_space(self, action_space):
        env = gym.make(ENV_ID, action_space=action_space)
        assert env is not None
        env.close()

    @pytest.mark.parametrize("reward_fn", REWARD_FUNCTIONS)
    def test_create_with_reward_function(self, reward_fn):
        env = gym.make(ENV_ID, reward_function=reward_fn)
        assert env is not None
        env.close()

    @pytest.mark.parametrize("reward_type", REWARD_TYPES)
    def test_create_with_reward_type(self, reward_type):
        env = gym.make(ENV_ID, reward_type=reward_type)
        assert env is not None
        env.close()

    def test_create_with_custom_max_steps(self):
        env = gym.make(ENV_ID, max_steps=500)
        assert env is not None
        env.close()

    def test_create_with_original_noise(self):
        env = gym.make(ENV_ID, original_noise=True)
        assert env is not None
        env.close()


class TestResetStep:
    """Verify reset and step produce correct outputs."""

    def test_reset_returns_obs_and_info(self, discrete_env):
        result = discrete_env.reset(seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_is_deterministic_with_seed(self, discrete_env):
        obs1, _ = discrete_env.reset(seed=123)
        obs2, _ = discrete_env.reset(seed=123)
        for key in obs1:
            assert (obs1[key] == obs2[key]).all(), f"Mismatch in {key}"

    def test_step_returns_five_tuple(self, discrete_env):
        discrete_env.reset(seed=0)
        result = discrete_env.step(1)
        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_info_keys(self, discrete_env):
        discrete_env.reset(seed=0)
        _, _, _, _, info = discrete_env.step(0)
        expected_keys = {
            "car_position", "prev_max_distance", "score", "dead",
            "steps_stuck", "airtime_counter", "total_airtime",
            "on_ground", "position_list",
        }
        assert expected_keys == set(info.keys())

    def test_multi_step_loop(self, discrete_env):
        """Run 50 steps without crashing."""
        discrete_env.reset(seed=42)
        for _ in range(50):
            action = discrete_env.action_space.sample()
            obs, reward, terminated, truncated, info = discrete_env.step(action)
            if terminated or truncated:
                discrete_env.reset()
                break

    def test_continuous_step(self, continuous_env):
        continuous_env.reset(seed=0)
        action = continuous_env.action_space.sample()
        obs, reward, terminated, truncated, info = continuous_env.step(action)
        assert isinstance(obs, dict)

    @pytest.mark.parametrize("reward_fn", REWARD_FUNCTIONS)
    def test_all_reward_functions_produce_numeric_reward(self, reward_fn):
        env = gym.make(ENV_ID, reward_function=reward_fn)
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(1)
        assert isinstance(reward, (int, float))
        env.close()
