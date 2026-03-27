"""
Test MDN-RNN Temporal World Model

Validates shapes, NLL loss, and that training decreases loss.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as cfg
from models.mdn_rnn import MDNRNN


@pytest.fixture
def rnn():
    return MDNRNN().to(cfg.DEVICE)


class TestMDNRNNShapes:
    """Verify tensor shapes."""

    def test_forward_shapes(self, rnn):
        z_vis = torch.randn(4, cfg.VISION_LATENT_DIM, device=cfg.DEVICE)
        z_pro = torch.randn(4, cfg.PROPRIO_LATENT_DIM, device=cfg.DEVICE)
        action = torch.randn(4, cfg.ACTION_DIM, device=cfg.DEVICE)

        mu, logvar, hidden = rnn.forward(z_vis, z_pro, action)

        assert mu.shape == (4, cfg.RNN_OUTPUT_DIM), f"mu: {mu.shape}"
        assert logvar.shape == (4, cfg.RNN_OUTPUT_DIM), f"logvar: {logvar.shape}"
        assert hidden.shape == (4, cfg.RNN_HIDDEN_DIM), f"hidden: {hidden.shape}"

    def test_predict_shapes(self, rnn):
        z_vis = torch.randn(4, cfg.VISION_LATENT_DIM, device=cfg.DEVICE)
        z_pro = torch.randn(4, cfg.PROPRIO_LATENT_DIM, device=cfg.DEVICE)
        action = torch.randn(4, cfg.ACTION_DIM, device=cfg.DEVICE)

        z_next, hidden = rnn.predict(z_vis, z_pro, action)

        assert z_next.shape == (4, cfg.VISION_LATENT_DIM)
        assert hidden.shape == (4, cfg.RNN_HIDDEN_DIM)

    def test_multi_step_shapes(self, rnn):
        z_vis = torch.randn(2, cfg.VISION_LATENT_DIM, device=cfg.DEVICE)
        z_pro = torch.randn(2, cfg.PROPRIO_LATENT_DIM, device=cfg.DEVICE)
        actions = torch.randn(2, 5, cfg.ACTION_DIM, device=cfg.DEVICE)

        predicted_z, final_h = rnn.multi_step_predict(z_vis, z_pro, actions)

        assert predicted_z.shape == (2, 5, cfg.VISION_LATENT_DIM)
        assert final_h.shape == (2, cfg.RNN_HIDDEN_DIM)

    def test_hidden_state_persistence(self, rnn):
        """GRU hidden state persists across steps."""
        z_vis = torch.randn(1, cfg.VISION_LATENT_DIM, device=cfg.DEVICE)
        z_pro = torch.randn(1, cfg.PROPRIO_LATENT_DIM, device=cfg.DEVICE)
        action = torch.randn(1, cfg.ACTION_DIM, device=cfg.DEVICE)

        _, _, h1 = rnn.forward(z_vis, z_pro, action, hidden=None)
        _, _, h2 = rnn.forward(z_vis, z_pro, action, hidden=h1)

        # Hidden states should be different (state was updated)
        assert not torch.allclose(h1, h2), "Hidden state didn't change"


class TestMDNRNNTraining:
    """Verify training reduces NLL loss."""

    def test_nll_decreases(self, rnn):
        """Train on synthetic sequences. NLL should decrease."""
        optimizer = torch.optim.Adam(rnn.parameters(), lr=cfg.RNN_LR)

        torch.manual_seed(42)
        # Generate synthetic sequence data
        T = 50
        z_vis_seq = torch.randn(T, cfg.VISION_LATENT_DIM, device=cfg.DEVICE) * 0.1
        z_pro_seq = torch.randn(T, cfg.PROPRIO_LATENT_DIM, device=cfg.DEVICE) * 0.1
        action_seq = torch.randn(T, cfg.ACTION_DIM, device=cfg.DEVICE) * 0.1

        # Add temporal structure: each z is a noisy version of previous
        for t in range(1, T):
            z_vis_seq[t] = z_vis_seq[t - 1] * 0.9 + z_vis_seq[t] * 0.1

        initial_loss = None
        final_loss = None

        for step in range(200):
            total_loss = 0
            hidden = None

            for t in range(T - 1):
                z_v = z_vis_seq[t].unsqueeze(0)
                z_p = z_pro_seq[t].unsqueeze(0)
                a = action_seq[t].unsqueeze(0)
                z_next = z_vis_seq[t + 1].unsqueeze(0)

                mu, logvar, hidden = rnn.forward(z_v, z_p, a, hidden)
                loss = MDNRNN.compute_loss(z_next, mu, logvar)
                total_loss += loss

                hidden = hidden.detach()  # truncated BPTT

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()

            if step == 0:
                initial_loss = total_loss.item()
            if step == 199:
                final_loss = total_loss.item()

        print(f"  Initial NLL: {initial_loss:.4f}")
        print(f"  Final NLL:   {final_loss:.4f}")
        assert final_loss < initial_loss, \
            f"NLL did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
