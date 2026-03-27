"""
Forward Model — Efference Copy & Self-Model Formation

Every time the agent sends a motor command, the forward model predicts
the sensory consequences. If prediction matches reality → self-caused.
If mismatch → external cause. This produces a stable self-model.

Input:  joint_state(34) + motor_command(17) = 51-d
Output: predicted next proprioceptive state (34-d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class ForwardModel(nn.Module):
    """
    Efference copy mechanism. Predicts the sensory consequences of
    motor commands, enabling self vs. other discrimination.
    """

    def __init__(self):
        super().__init__()
        self.input_dim = cfg.FM_INPUT_DIM    # 34 + 17 = 51
        self.hidden_dim = cfg.FM_HIDDEN_DIM  # 128
        self.output_dim = cfg.FM_OUTPUT_DIM  # 34

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
        )

    def forward(self, joint_state, action):
        """
        Predict next proprioceptive state.

        Args:
            joint_state: (batch, 34) current proprioception
            action:      (batch, 17) motor command

        Returns:
            predicted_next: (batch, 34) predicted next proprioceptive state
        """
        x = torch.cat([joint_state, action], dim=-1)  # (batch, 51)
        return self.network(x)

    def predict(self, joint_state, action):
        """Convenience wrapper for inference (no grad)."""
        with torch.no_grad():
            return self.forward(joint_state, action)

    @staticmethod
    def compute_loss(predicted_next, actual_next):
        """
        Forward model prediction error.

        Args:
            predicted_next: (batch, 34)
            actual_next:    (batch, 34)

        Returns:
            loss: scalar MSE
        """
        return F.mse_loss(predicted_next, actual_next, reduction="mean")

    def get_prediction_error(self, joint_state, action, actual_next):
        """
        Compute per-sample prediction error (used for self-other signal).

        Returns:
            error: float
        """
        with torch.no_grad():
            predicted = self.forward(joint_state, action)
            error = F.mse_loss(predicted, actual_next, reduction="mean")
        return error.item()

    @staticmethod
    def self_other_signal(forward_model_error, vae_visual_error):
        """
        Compute self vs. other discrimination signal.

        Self-caused: forward model error LOW, visual error variable
        External:    forward model error HIGH relative to visual error

        Self-model confidence grows as forward model loss decreases.
        Body ownership emerges when FM loss consistently < visual error.

        Args:
            forward_model_error: float, proprioceptive prediction error
            vae_visual_error: float, visual reconstruction error

        Returns:
            self_confidence: float in [0, 1], how confident agent is
                            that current sensation is self-caused
        """
        if forward_model_error < 1e-8:
            return 1.0

        # Ratio: lower FM error relative to visual error → more self-caused
        ratio = forward_model_error / (vae_visual_error + 1e-8)

        # Sigmoid-like mapping: ratio << 1 → confident it's self
        self_confidence = 1.0 / (1.0 + ratio)

        return float(self_confidence)
