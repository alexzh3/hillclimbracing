"""Tests for observation and action space contracts."""

import numpy as np
from gymnasium import spaces


class TestObservationSpace:
    """Verify observations conform to the declared observation space."""

    def test_obs_keys_match_space(self, discrete_env):
        obs, _ = discrete_env.reset(seed=0)
        assert set(obs.keys()) == set(discrete_env.observation_space.spaces.keys())

    def test_obs_in_space_after_reset(self, discrete_env):
        obs, _ = discrete_env.reset(seed=0)
        assert discrete_env.observation_space.contains(obs), (
            f"Observation after reset not in declared space: {obs}"
        )

    def test_obs_in_space_after_step(self, discrete_env):
        discrete_env.reset(seed=0)
        obs, _, _, _, _ = discrete_env.step(1)
        assert discrete_env.observation_space.contains(obs), (
            f"Observation after step not in declared space: {obs}"
        )

    def test_chassis_position_shape(self, discrete_env):
        obs, _ = discrete_env.reset(seed=0)
        assert obs["chassis_position"].shape == (2,)
        assert obs["chassis_position"].dtype == np.float32

    def test_chassis_angle_shape(self, discrete_env):
        obs, _ = discrete_env.reset(seed=0)
        assert obs["chassis_angle"].shape == (1,)
        assert obs["chassis_angle"].dtype == np.float32

    def test_wheels_speed_shape(self, discrete_env):
        obs, _ = discrete_env.reset(seed=0)
        assert obs["wheels_speed"].shape == (2,)
        assert obs["wheels_speed"].dtype == np.float32

    def test_on_ground_shape(self, discrete_env):
        obs, _ = discrete_env.reset(seed=0)
        assert obs["on_ground"].shape == (2,)


class TestDiscreteActionSpace:
    """Verify the discrete action space."""

    def test_is_discrete(self, discrete_env):
        assert isinstance(discrete_env.action_space, spaces.Discrete)

    def test_has_three_actions(self, discrete_env):
        assert discrete_env.action_space.n == 3

    def test_sample_in_range(self, discrete_env):
        for _ in range(20):
            action = discrete_env.action_space.sample()
            assert 0 <= action < 3


class TestContinuousActionSpace:
    """Verify the continuous action space."""

    def test_is_box(self, continuous_env):
        assert isinstance(continuous_env.action_space, spaces.Box)

    def test_shape(self, continuous_env):
        assert continuous_env.action_space.shape == (1,)

    def test_bounds(self, continuous_env):
        assert continuous_env.action_space.low[0] == -13
        assert continuous_env.action_space.high[0] == 13

    def test_sample_in_bounds(self, continuous_env):
        for _ in range(20):
            action = continuous_env.action_space.sample()
            assert -13 <= action[0] <= 13
