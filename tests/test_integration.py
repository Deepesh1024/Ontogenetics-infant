"""
Integration Smoke Test

Validates that all components connect without crashing.
Does NOT require PyBullet — uses mock observations.
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as cfg
from models.vae import VAEWorldModel
from models.mdn_rnn import MDNRNN
from models.forward_model import ForwardModel
from models.action_model import ActionModel
from memory.episodic_buffer import EpisodicBuffer
from cognition.curiosity import CuriosityEngine
from cognition.global_workspace import MessageBus
from development.stage_manager import StageManager


def make_mock_obs(num_joints=17):
    """Create a mock observation dict."""
    return {
        "vision": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        "vision_flat": np.random.rand(12288).astype(np.float32),
        "proprio": np.random.randn(num_joints * 2).astype(np.float32),
        "touch": np.random.rand(6).astype(np.float32),
        "parent_obs": np.random.randn(num_joints * 2).astype(np.float32),
        "infant_pos": np.array([0, 0, 0.9], dtype=np.float32),
        "infant_height": 0.9,
        "parent_pos": np.array([2, 0, 0.9], dtype=np.float32),
        "parent_height": 0.9,
        "timestep": 0,
    }


class TestComponentIntegration:
    """Test that all components connect."""

    def test_vae_to_mdn_rnn_pipeline(self):
        """VAE output feeds into MDN-RNN."""
        vae = VAEWorldModel()
        rnn = MDNRNN()

        vision = torch.rand(1, cfg.VISION_RAW_DIM)
        proprio = torch.randn(1, cfg.PROPRIO_DIM)
        action = torch.randn(1, cfg.ACTION_DIM)

        z_vis, z_pro, _, _ = vae.encode(vision, proprio)
        mu, logvar, hidden = rnn.forward(z_vis, z_pro, action)

        assert mu.shape == (1, 64)
        assert hidden.shape == (1, 256)

    def test_episodic_buffer_round_trip(self):
        """Store and sample transitions."""
        buffer = EpisodicBuffer(capacity=100)

        for i in range(50):
            transition = {
                "vision_flat": np.random.rand(12288).astype(np.float32),
                "proprio": np.random.randn(34).astype(np.float32),
                "action": np.random.randn(17).astype(np.float32),
                "z_vision": np.random.randn(64).astype(np.float32),
                "z_proprio": np.random.randn(32).astype(np.float32),
                "next_vision_flat": np.random.rand(12288).astype(np.float32),
                "next_proprio": np.random.randn(34).astype(np.float32),
                "next_z_vision": np.random.randn(64).astype(np.float32),
                "parent_action": np.random.randn(17).astype(np.float32),
                "timestamp": i,
            }
            buffer.add(transition, prediction_error=np.random.rand())

        assert len(buffer) == 50
        batch = buffer.sample(16)
        assert len(batch) == 16

    def test_global_workspace_passive(self):
        """GWT message bus in passive mode (Phase 1)."""
        bus = MessageBus()
        bus.register_module("sensory")
        bus.register_module("curiosity")

        bus.post("sensory", {"z": np.zeros(64)}, salience=0.5)
        bus.post("curiosity", {"reward": 0.1}, salience=0.3)

        broadcast = bus.get_broadcast()
        assert "sensory" in broadcast
        assert "curiosity" in broadcast  # Phase 1: all modules broadcast

    def test_global_workspace_competitive(self):
        """GWT competition mode (Phase 5)."""
        bus = MessageBus()
        bus.register_module("sensory")
        bus.register_module("curiosity")
        bus.activate_competition()

        bus.post("sensory", {"z": np.zeros(64)}, salience=0.5)
        bus.post("curiosity", {"reward": 0.1}, salience=0.8)

        winner, data = bus.compete()
        assert winner == "curiosity"  # higher salience

        broadcast = bus.get_broadcast()
        assert "curiosity" in broadcast
        assert "sensory" not in broadcast  # only winner broadcasts

    def test_stage_manager_transitions(self):
        """Stage manager advances on correct thresholds."""
        sm = StageManager()
        assert sm.stage == 1

        # Stage 1 → 2: VAE loss < 0.05, buffer ≥ 5000
        advanced = sm.check_transition({
            "vae_recon_loss": 0.03,
            "buffer_size": 6000,
        })
        assert advanced and sm.stage == 2

        # Stage 2 → 3: RNN accuracy > 70%
        advanced = sm.check_transition({
            "rnn_accuracy": 0.75,
        })
        assert advanced and sm.stage == 3

    def test_forward_model_self_signal(self):
        """Forward model produces self-other signal."""
        fm = ForwardModel()
        joint = torch.randn(1, cfg.PROPRIO_DIM)
        action = torch.randn(1, cfg.ACTION_DIM)

        predicted = fm(joint, action)
        assert predicted.shape == (1, cfg.PROPRIO_DIM)

        confidence = ForwardModel.self_other_signal(0.001, 0.1)
        assert 0 < confidence <= 1
        print(f"  Self-confidence (low FM, high VAE): {confidence:.3f}")

        confidence2 = ForwardModel.self_other_signal(0.1, 0.001)
        assert confidence > confidence2  # more confident when FM error is lower

    def test_action_model_modes(self):
        """Action model produces actions in all modes."""
        am = ActionModel()

        z_vis = torch.randn(64)
        z_pro = torch.randn(32)

        # Stage 1: random
        action, info = am.select_action(z_vis, z_pro, stage=1)
        assert action.shape == (17,)
        assert info["mode"] == "random"

        # Stage 3: behavioral cloning
        action, info = am.select_action(z_vis, z_pro, stage=3)
        assert action.shape == (17,)
        assert info["mode"] == "behavioral_cloning"

    def test_full_loop_mock(self):
        """Run 10 steps of the full loop with mock data."""
        vae = VAEWorldModel()
        rnn = MDNRNN()
        buffer = EpisodicBuffer()
        curiosity = CuriosityEngine()
        sm = StageManager()

        rnn_hidden = None

        for t in range(10):
            obs = make_mock_obs()

            # Encode
            vision_t = torch.tensor(obs["vision_flat"]).unsqueeze(0)
            proprio_t = torch.tensor(obs["proprio"]).unsqueeze(0)
            z_vis, z_pro, mu, logvar = vae.encode(vision_t, proprio_t)

            # Predict
            action = torch.randn(1, cfg.ACTION_DIM)
            if rnn_hidden is not None:
                mu_next, logvar_next, rnn_hidden = rnn.forward(
                    z_vis, z_pro, action, rnn_hidden
                )
            else:
                mu_next, logvar_next, rnn_hidden = rnn.forward(
                    z_vis, z_pro, action
                )

            # Store
            pred_error = vae.get_prediction_error(
                torch.tensor(obs["vision_flat"]),
                torch.tensor(obs["proprio"]),
            )
            buffer.add({
                "vision_flat": obs["vision_flat"],
                "proprio": obs["proprio"],
                "action": action.squeeze().numpy(),
                "z_vision": z_vis.squeeze().detach().numpy(),
                "z_proprio": z_pro.squeeze().detach().numpy(),
                "next_vision_flat": obs["vision_flat"],
                "next_proprio": obs["proprio"],
                "next_z_vision": z_vis.squeeze().detach().numpy(),
                "timestamp": t,
            }, prediction_error=pred_error)

            # Curiosity
            curiosity.update(z_vis.squeeze().detach().numpy(), pred_error)

            rnn_hidden = rnn_hidden.detach()

        assert len(buffer) == 10
        assert sm.stage == 1
        print(f"  Full loop: {len(buffer)} transitions, stage {sm.stage}")
