"""
Action Model — Behavioral Cloning + Active Inference

Two action selection modes operating on a shared interface:
1. Behavioral Cloning (Stages 1-3): MSE(infant_action, parent_action)
   Weight anneals from 1.0 → 0.0 over 10k steps.
2. Active Inference (Stages 4-5): Select actions minimizing expected
   free energy G = epistemic + pragmatic value.

Transition gate: BC → Active Inference when:
  - VAE reconstruction loss < 0.05 AND
  - MDN-RNN prediction accuracy > 70%
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class ActionModel(nn.Module):
    """
    Combined action selection model.

    Behavioral cloning provides supervised signal from parent demos.
    Active inference generates actions by simulating future outcomes
    and minimizing expected free energy.
    """

    def __init__(self, action_dim=None):
        super().__init__()
        self.action_dim = action_dim or cfg.ACTION_DIM

        # Behavioral cloning network
        # Input: infant's observation latent (z_vis + z_proprio)
        # Output: action matching parent
        bc_input_dim = cfg.VISION_LATENT_DIM + cfg.PROPRIO_LATENT_DIM  # 64 + 32 = 96
        self.bc_network = nn.Sequential(
            nn.Linear(bc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh(),  # actions in [-1, 1]
        )

        # BC weight for annealing
        self.bc_weight = cfg.BC_WEIGHT_INITIAL
        self._bc_step = 0

        # Active inference parameters
        self.num_candidates = cfg.NUM_ACTION_CANDIDATES
        self.planning_horizon = cfg.PLANNING_HORIZON

        # Prior preference: initially = parent state (updated as learning progresses)
        self._prior_preference = None

    def behavioral_cloning_action(self, z_vision, z_proprio):
        """
        Generate action via behavioral cloning network.

        Args:
            z_vision: (batch, 64) or (64,)
            z_proprio: (batch, 32) or (32,)

        Returns:
            action: (batch, 17) or (17,) action in [-1, 1]
        """
        squeeze = z_vision.dim() == 1
        if squeeze:
            z_vision = z_vision.unsqueeze(0)
            z_proprio = z_proprio.unsqueeze(0)

        x = torch.cat([z_vision, z_proprio], dim=-1)
        action = self.bc_network(x)

        if squeeze:
            action = action.squeeze(0)
        return action

    def compute_bc_loss(self, z_vision, z_proprio, parent_action):
        """
        Behavioral cloning loss: MSE between predicted and parent action.

        Args:
            z_vision: (batch, 64)
            z_proprio: (batch, 32)
            parent_action: (batch, 17) target action from parent

        Returns:
            loss: scalar, weighted by current bc_weight
        """
        predicted = self.behavioral_cloning_action(z_vision, z_proprio)
        raw_loss = F.mse_loss(predicted, parent_action, reduction="mean")
        return raw_loss * self.bc_weight

    def anneal_bc_weight(self):
        """Decay BC weight linearly over anneal_steps. Call every training step."""
        self._bc_step += 1
        progress = min(1.0, self._bc_step / cfg.BC_ANNEAL_STEPS)
        self.bc_weight = cfg.BC_WEIGHT_INITIAL * (1.0 - progress)

    def active_inference_action(self, z_vision, z_proprio, rnn_hidden, world_model_rnn,
                                 prior_preference=None):
        """
        Select action via active inference: simulate K candidate action sequences
        using the MDN-RNN, evaluate expected free energy, pick minimum G.

        G(policy) = epistemic_value + pragmatic_value
          - Epistemic: expected reduction in uncertainty (info gain)
          - Pragmatic: KL(predicted outcome ‖ preferred outcome)

        Args:
            z_vision: (64,) current visual latent
            z_proprio: (32,) current proprioceptive latent
            rnn_hidden: (256,) current GRU hidden state
            world_model_rnn: MDNRNN instance for simulation
            prior_preference: (64,) preferred z_vision state (default: parent state)

        Returns:
            best_action: (17,) action with minimum expected free energy
            best_G: float, the expected free energy
        """
        if prior_preference is None:
            prior_preference = self._prior_preference
        if prior_preference is None:
            # Fallback: prefer current state (minimize change)
            prior_preference = z_vision.detach()

        device = z_vision.device
        K = self.num_candidates
        H = self.planning_horizon

        best_G = float("inf")
        best_action = torch.zeros(self.action_dim, device=device)

        with torch.no_grad():
            for _ in range(K):
                # Generate random action sequence
                candidate_actions = torch.rand(H, self.action_dim, device=device) * 2 - 1

                # Simulate trajectory using MDN-RNN
                z_curr = z_vision.unsqueeze(0)  # (1, 64)
                zp = z_proprio.unsqueeze(0)      # (1, 32)
                h = rnn_hidden.unsqueeze(0) if rnn_hidden.dim() == 1 else rnn_hidden

                total_pragmatic = 0.0
                total_epistemic = 0.0

                for t in range(H):
                    action_t = candidate_actions[t].unsqueeze(0)  # (1, 17)
                    mu_next, logvar_next, h = world_model_rnn.forward(
                        z_curr, zp, action_t, h
                    )

                    # Epistemic value: predicted reduction in uncertainty
                    # Approximated as negative variance (more certain = better)
                    variance = torch.exp(logvar_next)
                    epistemic = variance.mean().item()
                    total_epistemic += epistemic

                    # Pragmatic value: distance from preferred state
                    pref = prior_preference.unsqueeze(0) if prior_preference.dim() == 1 else prior_preference
                    pragmatic = F.mse_loss(mu_next, pref, reduction="mean").item()
                    total_pragmatic += pragmatic

                    # Next state for continued simulation
                    z_curr = mu_next  # use mean prediction

                # Expected free energy: lower is better
                G = total_epistemic + total_pragmatic

                if G < best_G:
                    best_G = G
                    best_action = candidate_actions[0]  # execute first action only

        return best_action, best_G

    def select_action(self, z_vision, z_proprio, stage, rnn_hidden=None,
                      world_model_rnn=None, prior_preference=None,
                      vae_loss=float("inf"), rnn_accuracy=0.0):
        """
        Main action selection interface. Chooses between BC and active inference
        based on developmental stage and world model quality.

        Args:
            z_vision: (64,) visual latent
            z_proprio: (32,) proprioceptive latent
            stage: int, current developmental stage (1-5)
            rnn_hidden: (256,) GRU hidden for active inference
            world_model_rnn: MDNRNN for active inference rollouts
            prior_preference: (64,) preferred future state
            vae_loss: current VAE reconstruction loss
            rnn_accuracy: current MDN-RNN accuracy

        Returns:
            action: (17,) numpy array in [-1, 1]
            info: dict with action selection metadata
        """
        info = {"mode": "random", "bc_weight": self.bc_weight}

        if stage == 1:
            # Stage 1: random exploration
            action = np.random.uniform(-0.3, 0.3, self.action_dim).astype(np.float32)
            info["mode"] = "random"
            return action, info

        if stage <= 3 or self.bc_weight > 0.01:
            # Stages 2-3: behavioral cloning (or BC still active during annealing)
            with torch.no_grad():
                action = self.behavioral_cloning_action(z_vision, z_proprio)
            action = action.cpu().numpy()
            info["mode"] = "behavioral_cloning"

            # Add small exploration noise
            noise = np.random.normal(0, 0.05, self.action_dim).astype(np.float32)
            action = np.clip(action + noise, -1.0, 1.0)
            return action, info

        # Stages 4-5: active inference (if world model is ready)
        world_model_ready = (
            vae_loss < cfg.STAGE1_VAE_LOSS_THRESHOLD and
            rnn_accuracy > cfg.STAGE2_RNN_ACCURACY_THRESHOLD
        )

        if world_model_ready and world_model_rnn is not None and rnn_hidden is not None:
            action, G = self.active_inference_action(
                z_vision, z_proprio, rnn_hidden, world_model_rnn,
                prior_preference=prior_preference,
            )
            action = action.cpu().numpy()
            info["mode"] = "active_inference"
            info["expected_free_energy"] = G

            # Small exploration noise
            noise = np.random.normal(0, 0.02, self.action_dim).astype(np.float32)
            action = np.clip(action + noise, -1.0, 1.0)
            return action, info
        else:
            # World model not ready — revert to BC (failure mode fix)
            with torch.no_grad():
                action = self.behavioral_cloning_action(z_vision, z_proprio)
            action = action.cpu().numpy()
            info["mode"] = "bc_fallback"
            info["reason"] = f"world_model_not_ready (vae={vae_loss:.4f}, rnn_acc={rnn_accuracy:.2f})"
            return action, info

    def set_prior_preference(self, preference):
        """Set the preferred future state for active inference."""
        if isinstance(preference, np.ndarray):
            preference = torch.tensor(preference, dtype=torch.float32)
        self._prior_preference = preference

    def get_bc_weight(self):
        return self.bc_weight
