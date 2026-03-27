"""
Test Environment — Phase 1 Hard Gate

Validates:
1. URDF loads with correct joint count
2. Parent CPG walks 500 steps without falling
3. Observation dimensions match spec
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def env():
    """Create environment for testing."""
    from environment.simulation import OISEnvironment
    environment = OISEnvironment(render=False)
    yield environment
    environment.close()


@pytest.fixture
def parent(env):
    """Create parent agent for testing."""
    from agents.parent_agent import ParentAgent
    import config as cfg
    return ParentAgent(num_joints=env.parent.num_actuated, timestep=cfg.TIMESTEP)


class TestEnvironmentSetup:
    """Test that the environment initializes correctly."""

    def test_humanoid_loads(self, env):
        """Humanoid URDF loads with actuatable joints."""
        assert env.parent.num_actuated > 0, "Parent has no actuated joints"
        assert env.infant.num_actuated > 0, "Infant has no actuated joints"
        print(f"  Parent joints: {env.parent.num_actuated}")
        print(f"  Infant joints: {env.infant.num_actuated}")

    def test_observation_shape(self, env):
        """Observations have correct dimensions."""
        obs = env.get_observation()
        assert obs["vision"].shape == (64, 64, 3), f"Vision shape: {obs['vision'].shape}"
        assert obs["vision_flat"].shape[0] == 12288, f"Vision flat: {obs['vision_flat'].shape}"
        assert obs["proprio"].shape[0] == env.infant.num_actuated * 2
        assert obs["touch"].shape[0] == 6
        print(f"  Vision: {obs['vision'].shape}")
        print(f"  Proprio: {obs['proprio'].shape}")
        print(f"  Touch: {obs['touch'].shape}")

    def test_joint_normalization(self, env):
        """Joint angles normalized to [-1, 1]."""
        angles, velocities = env.infant.get_joint_states()
        assert np.all(angles >= -1.0) and np.all(angles <= 1.0), \
            f"Angles out of range: [{angles.min()}, {angles.max()}]"
        assert np.all(velocities >= -1.0) and np.all(velocities <= 1.0), \
            f"Velocities out of range: [{velocities.min()}, {velocities.max()}]"


class TestParentCPG:
    """Phase 1 HARD GATE: Parent must walk 500 steps."""

    def test_parent_walks_500_steps(self, env, parent):
        """
        ⛔ HARD GATE: Parent walks 500 steps without falling.
        COM height must stay above 0.3m throughout.
        """
        result = env.run_parent_demo(500, parent_agent=parent)

        print(f"\n  Steps survived: {result['steps_survived']}")
        print(f"  Min height: {result['min_height']:.3f}")
        print(f"  Fell: {result['fell']}")

        assert not result["fell"], \
            f"Parent fell at step {result['steps_survived']} (height {result['min_height']:.3f})"
        assert result["steps_survived"] >= 500, \
            f"Parent only survived {result['steps_survived']} steps"
        assert result["min_height"] > 0.3, \
            f"Parent COM dropped to {result['min_height']:.3f}"

    def test_parent_action_range(self, env, parent):
        """Parent actions are in [-1, 1] range."""
        for t in range(100):
            angles, vels = env.parent.get_joint_states()
            action = parent.get_action(t, angles, vels)
            assert np.all(action >= -1.0) and np.all(action <= 1.0), \
                f"Action out of range at step {t}: [{action.min()}, {action.max()}]"


class TestReset:
    """Test environment reset."""

    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert "vision" in obs
        assert "proprio" in obs
        assert "touch" in obs

    def test_reset_restores_height(self, env):
        """After reset, humanoids should be at starting height."""
        env.reset()
        parent_h = env.parent.get_height()
        infant_h = env.infant.get_height()
        assert parent_h > 0.5, f"Parent height after reset: {parent_h}"
        assert infant_h > 0.5, f"Infant height after reset: {infant_h}"
