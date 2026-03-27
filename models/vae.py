"""
VAE World Model — Subconscious Compression Layer

Variational Autoencoder that compresses high-dimensional sensory observations
into a low-dimensional probabilistic latent representation. This latent space
IS the agent's subconscious — everything it knows about the world, encoded as
a probability distribution.

Visual encoder: 12288 → 512 → 256 → μ(64) + logσ²(64) → z_vis(64)
Proprio encoder: 34 → 128 → 64 → z_proprio(32)
Decoder: z_vis(64) → 256 → 512 → 12288 (Sigmoid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class VisualEncoder(nn.Module):
    """Encodes flattened 64x64x3 visual input to latent distribution parameters."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(cfg.VISION_RAW_DIM, cfg.VAE_HIDDEN_1)       # 12288 → 512
        self.fc2 = nn.Linear(cfg.VAE_HIDDEN_1, cfg.VAE_HIDDEN_2)          # 512 → 256
        self.fc_mu = nn.Linear(cfg.VAE_HIDDEN_2, cfg.VISION_LATENT_DIM)   # 256 → 64
        self.fc_logvar = nn.Linear(cfg.VAE_HIDDEN_2, cfg.VISION_LATENT_DIM)  # 256 → 64

    def forward(self, x):
        """
        Args:
            x: (batch, 12288) flattened RGB image, values in [0, 1]
        Returns:
            mu: (batch, 64)
            logvar: (batch, 64)
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VisualDecoder(nn.Module):
    """Decodes latent z back to visual observation."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(cfg.VISION_LATENT_DIM, cfg.VAE_HIDDEN_2)   # 64 → 256
        self.fc2 = nn.Linear(cfg.VAE_HIDDEN_2, cfg.VAE_HIDDEN_1)         # 256 → 512
        self.fc_out = nn.Linear(cfg.VAE_HIDDEN_1, cfg.VISION_RAW_DIM)    # 512 → 12288

    def forward(self, z):
        """
        Args:
            z: (batch, 64) latent vector
        Returns:
            x_recon: (batch, 12288) reconstructed observation in [0, 1]
        """
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        x_recon = torch.sigmoid(self.fc_out(h))
        return x_recon


class ProprioceptiveEncoder(nn.Module):
    """Encodes proprioceptive input (joint angles + velocities) to latent."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(cfg.PROPRIO_DIM, 128)                         # 34 → 128
        self.fc2 = nn.Linear(128, 64)                                       # 128 → 64
        self.fc_out = nn.Linear(64, cfg.PROPRIO_LATENT_DIM)                # 64 → 32

    def forward(self, x):
        """
        Args:
            x: (batch, 34) proprioceptive vector
        Returns:
            z_proprio: (batch, 32)
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        z_proprio = self.fc_out(h)
        return z_proprio


class VAEWorldModel(nn.Module):
    """
    Complete VAE World Model combining visual and proprioceptive encoders.

    Prediction error signal:
      - Reconstruction loss: MSE(decoded, original)
      - KL divergence: regularization toward N(0,1)
      - Total = reconstruction + β * KL
    """

    def __init__(self, kl_weight=None):
        super().__init__()
        self.visual_encoder = VisualEncoder()
        self.visual_decoder = VisualDecoder()
        self.proprio_encoder = ProprioceptiveEncoder()
        self.kl_weight = kl_weight if kl_weight is not None else cfg.KL_WEIGHT

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + eps * exp(0.5 * logvar)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, vision_flat, proprio):
        """
        Encode observations into latent space.

        Args:
            vision_flat: (batch, 12288) flattened RGB
            proprio: (batch, 34) proprioceptive vector

        Returns:
            z_vision: (batch, 64) sampled visual latent
            z_proprio: (batch, 32) proprioceptive latent
            mu: (batch, 64) visual latent mean
            logvar: (batch, 64) visual latent log-variance
        """
        mu, logvar = self.visual_encoder(vision_flat)
        z_vision = self.reparameterize(mu, logvar)
        z_proprio = self.proprio_encoder(proprio)
        return z_vision, z_proprio, mu, logvar

    def decode(self, z_vision):
        """
        Decode visual latent back to observation.

        Args:
            z_vision: (batch, 64)

        Returns:
            x_recon: (batch, 12288) reconstructed observation
        """
        return self.visual_decoder(z_vision)

    def forward(self, vision_flat, proprio):
        """
        Full forward pass: encode → sample → decode.

        Returns:
            x_recon: reconstructed visual observation
            z_vision: sampled visual latent
            z_proprio: proprioceptive latent
            mu: visual latent mean
            logvar: visual latent log-variance
        """
        z_vision, z_proprio, mu, logvar = self.encode(vision_flat, proprio)
        x_recon = self.decode(z_vision)
        return x_recon, z_vision, z_proprio, mu, logvar

    def compute_loss(self, vision_flat, proprio):
        """
        Compute VAE loss = reconstruction + β * KL divergence.

        Returns:
            total_loss: scalar
            recon_loss: scalar (MSE)
            kl_loss: scalar (KL divergence)
        """
        x_recon, z_vision, z_proprio, mu, logvar = self.forward(vision_flat, proprio)

        # Reconstruction loss (prediction error)
        recon_loss = F.mse_loss(x_recon, vision_flat, reduction="mean")

        # KL divergence: -0.5 * mean(1 + logvar - mu² - exp(logvar))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + self.kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss

    def get_prediction_error(self, vision_flat, proprio):
        """
        Get per-sample reconstruction error (used for curiosity and memory priority).

        Args:
            vision_flat: (batch, 12288) or (12288,)
            proprio: (batch, 34) or (34,)

        Returns:
            error: scalar — mean reconstruction error
        """
        was_1d = vision_flat.dim() == 1
        if was_1d:
            vision_flat = vision_flat.unsqueeze(0)
            proprio = proprio.unsqueeze(0)

        with torch.no_grad():
            x_recon, _, _, _, _ = self.forward(vision_flat, proprio)
            error = F.mse_loss(x_recon, vision_flat, reduction="mean")

        return error.item()
