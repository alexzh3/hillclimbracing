"""Tests for the custom Perlin noise implementation."""

from hill_racing_env.envs.perlin import original_pnoise, scaled_cosine


class TestScaledCosine:
    def test_zero(self):
        assert scaled_cosine(0) == 0.0

    def test_one(self):
        assert abs(scaled_cosine(1) - 1.0) < 1e-9

    def test_half(self):
        assert abs(scaled_cosine(0.5) - 0.5) < 1e-9

    def test_range(self):
        for i in range(11):
            val = scaled_cosine(i / 10.0)
            assert 0.0 <= val <= 1.0


class TestOriginalPnoise:
    def test_returns_float(self):
        result = original_pnoise(0.5)
        assert isinstance(result, float)

    def test_deterministic(self):
        a = original_pnoise(1.234)
        b = original_pnoise(1.234)
        assert a == b

    def test_output_range(self):
        for x in [0.0, 0.5, 1.0, 10.0, 100.0, 1000.0]:
            val = original_pnoise(x)
            assert 0.0 <= val <= 1.0, f"pnoise({x}) = {val} out of [0, 1]"

    def test_different_inputs_differ(self):
        vals = {original_pnoise(x) for x in [0.1, 5.5, 50.0, 500.0]}
        assert len(vals) > 1, "All inputs produced identical output"

    def test_negative_input(self):
        val = original_pnoise(-3.7)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_three_dimensions(self):
        val = original_pnoise(1.0, 2.0, 3.0)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0
