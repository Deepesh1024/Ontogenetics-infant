"""
Test VAE World Model

Validates shapes, loss computation, and that training decreases loss.
Uses synthetic data (pybullet not required for unit testing the model itself).
For real-data validation, run test_environment.py first.
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as cfg
from models.vae import VAEWorldModel


@pytest.fixture
def vae():
    return VAEWorldModel().to(cfg.DEVICE)


class TestVAEShapes:
    """Verify all tensor shapes through the forward pass."""

    def test_encode_shapes(self, vae):
        vision = torch.randn(4, cfg.VISION_RAW_DIM, device=cfg.DEVICE)
        proprio = torch.randn(4, cfg.PROPRIO_DIM, device=cfg.DEVICE)

        z_vis, z_pro, mu, logvar = vae.encode(vision, proprio)

        assert z_vis.shape == (4, cfg.VISION_LATENT_DIM), f"z_vis: {z_vis.shape}"
        assert z_pro.shape == (4, cfg.PROPRIO_LATENT_DIM), f"z_pro: {z_pro.shape}"
        assert mu.shape == (4, cfg.VISION_LATENT_DIM), f"mu: {mu.shape}"
        assert logvar.shape == (4, cfg.VISION_LATENT_DIM), f"logvar: {logvar.shape}"

    def test_decode_shapes(self, vae):
        z = torch.randn(4, cfg.VISION_LATENT_DIM, device=cfg.DEVICE)
        x_recon = vae.decode(z)
        assert x_recon.shape == (4, cfg.VISION_RAW_DIM), f"recon: {x_recon.shape}"

    def test_full_forward_shapes(self, vae):
        vision = torch.randn(4, cfg.VISION_RAW_DIM, device=cfg.DEVICE)
        proprio = torch.randn(4, cfg.PROPRIO_DIM, device=cfg.DEVICE)

        x_recon, z_vis, z_pro, mu, logvar = vae.forward(vision, proprio)

        assert x_recon.shape == (4, cfg.VISION_RAW_DIM)
        assert z_vis.shape == (4, cfg.VISION_LATENT_DIM)
        assert z_pro.shape == (4, cfg.PROPRIO_LATENT_DIM)


class TestVAELoss:
    """Verify loss computation."""

    def test_loss_returns_scalars(self, vae):
        vision = torch.randn(4, cfg.VISION_RAW_DIM, device=cfg.DEVICE)
        proprio = torch.randn(4, cfg.PROPRIO_DIM, device=cfg.DEVICE)

        total, recon, kl = vae.compute_loss(vision, proprio)

        assert total.dim() == 0, "Total loss not scalar"
        assert recon.dim() == 0, "Recon loss not scalar"
        assert kl.dim() == 0, "KL loss not scalar"
        assert total.item() > 0
        assert recon.item() >= 0
        assert kl.item() >= 0

    def test_prediction_error_scalar(self, vae):
        vision = torch.randn(cfg.VISION_RAW_DIM, device=cfg.DEVICE)
        proprio = torch.randn(cfg.PROPRIO_DIM, device=cfg.DEVICE)

        error = vae.get_prediction_error(vision, proprio)
        assert isinstance(error, float)
        assert error >= 0


class TestVAETraining:
    """Verify that training reduces loss."""

    def test_loss_decreases(self, vae):
        """Train for 200 steps on random data. Loss should decrease."""
        optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.VAE_LR)

        # Fixed dataset (random but consistent)
        torch.manual_seed(42)
        data_vision = torch.rand(256, cfg.VISION_RAW_DIM, device=cfg.DEVICE)
        data_proprio = torch.randn(256, cfg.PROPRIO_DIM, device=cfg.DEVICE)

        initial_loss = None
        final_loss = None

        for step in range(200):
            idx = torch.randint(0, 256, (32,))
            v_batch = data_vision[idx]
            p_batch = data_proprio[idx]

            total, recon, kl = vae.compute_loss(v_batch, p_batch)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            if step == 0:
                initial_loss = total.item()
            if step == 199:
                final_loss = total.item()

        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss:   {final_loss:.4f}")
        assert final_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
