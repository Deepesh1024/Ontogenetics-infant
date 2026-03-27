"""
MDN-RNN Temporal World Model

GRU-based temporal prediction model (Ha & Schmidhuber, 2018).
Takes current latent state z and action a, predicts the distribution
over the next latent state z_next. The GRU hidden state IS working memory.

Input:  concat(z_vision[64], z_proprio[32], action[17]) = 113-d
GRU:    hidden state 256-d, persists across timesteps
Output: μ_next(64), logσ²_next(64) — gaussian over next z_vision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class MDNRNN(nn.Module):
    """
    Mixture Density Network + RNN for temporal prediction.

    Uses a single Gaussian (not mixture) for simplicity — sufficient for
    the developmental learning task. Predicts distribution over next visual
    latent z_next given current (z_vision, z_proprio, action).
    """

    def __init__(self):
        super().__init__()
        self.input_dim = cfg.RNN_INPUT_DIM       # 64 + 32 + 17 = 113
        self.hidden_dim = cfg.RNN_HIDDEN_DIM      # 256
        self.output_dim = cfg.RNN_OUTPUT_DIM      # 64 (predict z_vision)

        # GRU cell — the temporal prediction engine
        self.gru = nn.GRUCell(self.input_dim, self.hidden_dim)

        # Output heads: predict next z_vision distribution
        self.fc_mu = nn.Linear(self.hidden_dim, self.output_dim)       # μ_next
        self.fc_logvar = nn.Linear(self.hidden_dim, self.output_dim)   # logσ²_next

    def forward(self, z_vision, z_proprio, action, hidden=None):
        """
        Single-step forward pass.

        Args:
            z_vision:  (batch, 64) current visual latent
            z_proprio: (batch, 32) current proprioceptive latent
            action:    (batch, 17) current action
            hidden:    (batch, 256) GRU hidden state, or None for initial

        Returns:
            mu_next:    (batch, 64) predicted mean of next z_vision
            logvar_next:(batch, 64) predicted log-variance
            new_hidden: (batch, 256) updated GRU hidden state
        """
        # Concatenate inputs
        x = torch.cat([z_vision, z_proprio, action], dim=-1)  # (batch, 113)

        # Initialize hidden state if needed
        if hidden is None:
            hidden = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        # GRU step — this is where temporal memory is updated
        new_hidden = self.gru(x, hidden)

        # Predict distribution over next z_vision
        mu_next = self.fc_mu(new_hidden)
        logvar_next = self.fc_logvar(new_hidden)

        # Clamp logvar for numerical stability
        logvar_next = torch.clamp(logvar_next, min=-10.0, max=2.0)

        return mu_next, logvar_next, new_hidden

    def predict(self, z_vision, z_proprio, action, hidden=None):
        """
        Predict next z_vision by sampling from the predicted distribution.

        Returns:
            z_next: (batch, 64) sampled prediction
            new_hidden: (batch, 256) updated hidden state
        """
        mu_next, logvar_next, new_hidden = self.forward(
            z_vision, z_proprio, action, hidden
        )
        std = torch.exp(0.5 * logvar_next)
        eps = torch.randn_like(std)
        z_next = mu_next + eps * std
        return z_next, new_hidden

    @staticmethod
    def compute_loss(z_next_actual, mu_predicted, logvar_predicted):
        """
        Negative log-likelihood loss for temporal prediction.

        NLL = 0.5 * (logvar + (z_actual - mu_pred)² / exp(logvar))

        Args:
            z_next_actual:   (batch, 64) actual next latent
            mu_predicted:    (batch, 64) predicted mean
            logvar_predicted:(batch, 64) predicted log-variance

        Returns:
            nll_loss: scalar — mean NLL across batch and dimensions
        """
        var = torch.exp(logvar_predicted)
        nll = 0.5 * (logvar_predicted + (z_next_actual - mu_predicted).pow(2) / (var + 1e-8))
        return nll.mean()

    def multi_step_predict(self, z_vision, z_proprio, actions, hidden=None):
        """
        Roll out predictions over a sequence of actions.

        Args:
            z_vision:  (batch, 64) initial visual latent
            z_proprio: (batch, 32) initial proprioceptive latent
            actions:   (batch, T, 17) sequence of T actions
            hidden:    (batch, 256) initial hidden state

        Returns:
            predicted_z: (batch, T, 64) predicted z_vision at each step
            final_hidden: (batch, 256) final hidden state
        """
        batch_size = z_vision.size(0)
        T = actions.size(1)

        predicted_z = []
        z_curr = z_vision

        for t in range(T):
            action_t = actions[:, t, :]
            z_next, hidden = self.predict(z_curr, z_proprio, action_t, hidden)
            predicted_z.append(z_next)
            z_curr = z_next

        predicted_z = torch.stack(predicted_z, dim=1)  # (batch, T, 64)
        return predicted_z, hidden

    def compute_accuracy(self, z_next_actual, mu_predicted, threshold=0.1):
        """
        Compute prediction accuracy: fraction of dimensions where
        |predicted - actual| < threshold.

        Returns:
            accuracy: float in [0, 1]
        """
        with torch.no_grad():
            errors = torch.abs(z_next_actual - mu_predicted)
            accurate = (errors < threshold).float()
            return accurate.mean().item()

    def init_hidden(self, batch_size=1, device=None):
        """Create zero-initialized hidden state."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_dim, device=device)
