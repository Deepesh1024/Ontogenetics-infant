"""
Infant Agent — Top-Level Orchestrator

Ties together all neural modules: VAE, MDN-RNN, forward model, action model,
episodic buffer, curiosity engine, and global workspace.

Interface: observe() → act() → learn()
"""

import numpy as np
import torch
import torch.optim as optim

import config as cfg
from models.vae import VAEWorldModel
from models.mdn_rnn import MDNRNN
from models.forward_model import ForwardModel
from models.action_model import ActionModel
from memory.episodic_buffer import EpisodicBuffer
from cognition.curiosity import CuriosityEngine
from cognition.global_workspace import MessageBus


class InfantAgent:
    """
    The developing infant intelligence.
    Orchestrates all subsystems through the observe → act → learn loop.
    """

    def __init__(self, num_joints, device=None):
        self.device = device or cfg.DEVICE
        self.num_joints = num_joints

        # ─── Neural modules ────────────────────────────────────────
        self.vae = VAEWorldModel().to(self.device)
        self.mdn_rnn = MDNRNN().to(self.device)
        self.forward_model = ForwardModel().to(self.device)
        self.action_model = ActionModel(action_dim=num_joints).to(self.device)

        # ─── Optimizers ────────────────────────────────────────────
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=cfg.VAE_LR)
        self.rnn_optimizer = optim.Adam(self.mdn_rnn.parameters(), lr=cfg.RNN_LR)
        self.fm_optimizer = optim.Adam(self.forward_model.parameters(), lr=cfg.FM_LR)
        self.bc_optimizer = optim.Adam(self.action_model.parameters(), lr=cfg.VAE_LR)

        # ─── Memory ───────────────────────────────────────────────
        self.episodic_buffer = EpisodicBuffer()

        # ─── Cognition ─────────────────────────────────────────────
        self.curiosity = CuriosityEngine()
        self.workspace = MessageBus()

        # Register modules with global workspace
        for name in ["sensory", "memory", "curiosity", "action_planning"]:
            self.workspace.register_module(name)

        # ─── State tracking ────────────────────────────────────────
        self._rnn_hidden = None
        self._prev_z_vision = None
        self._prev_z_proprio = None
        self._prev_proprio = None
        self._prev_action = None
        self._prev_vision_flat = None
        self._current_z_vision = None
        self._current_z_proprio = None

        # ─── Running metrics ───────────────────────────────────────
        self._vae_recon_loss = float("inf")
        self._vae_kl_loss = 0.0
        self._rnn_nll_loss = float("inf")
        self._rnn_accuracy = 0.0
        self._fm_loss = float("inf")
        self._bc_loss = 0.0
        self._prediction_error = 0.0
        self._self_confidence = 0.0
        self._z_history = []

    def observe(self, obs):
        """
        Process a new observation through the sensory system.

        Args:
            obs: dict with keys: vision_flat, proprio, touch, parent_obs

        Returns:
            prediction_error: float, the VAE reconstruction error
        """
        # Convert to tensors
        vision_flat = torch.tensor(obs["vision_flat"], dtype=torch.float32, device=self.device)
        proprio = torch.tensor(obs["proprio"], dtype=torch.float32, device=self.device)

        # Encode through VAE (compression to subconscious latent space)
        with torch.no_grad():
            z_vis, z_pro, mu, logvar = self.vae.encode(
                vision_flat.unsqueeze(0), proprio.unsqueeze(0)
            )
            z_vis = z_vis.squeeze(0)
            z_pro = z_pro.squeeze(0)

        # Compute prediction error (VAE reconstruction)
        pred_error = self.vae.get_prediction_error(vision_flat, proprio)
        self._prediction_error = pred_error

        # Store previous state for transition storage
        self._prev_z_vision = self._current_z_vision
        self._prev_z_proprio = self._current_z_proprio
        self._prev_proprio = self._prev_proprio
        self._prev_vision_flat = obs["vision_flat"].copy() if isinstance(obs["vision_flat"], np.ndarray) else obs["vision_flat"]

        # Update current state
        self._current_z_vision = z_vis.detach()
        self._current_z_proprio = z_pro.detach()

        # Track latent vectors for visualization
        self._z_history.append(z_vis.cpu().numpy())
        if len(self._z_history) > 5000:
            self._z_history = self._z_history[-3000:]

        # Post to global workspace
        self.workspace.post("sensory", {
            "z_vision": z_vis.cpu().numpy(),
            "z_proprio": z_pro.cpu().numpy(),
            "prediction_error": pred_error,
        }, salience=pred_error)

        # Update curiosity engine
        self.curiosity.update(z_vis.cpu().numpy(), pred_error)
        self.workspace.post("curiosity", {
            "intrinsic_reward": self.curiosity.get_intrinsic_reward(z_vis.cpu().numpy()),
        }, salience=self.curiosity.get_salience())

        # Run competition (no-op in Phase 1)
        self.workspace.compete()

        return pred_error

    def act(self, stage, parent_demo=None):
        """
        Select and return an action based on developmental stage.

        Args:
            stage: int (1-5), current developmental stage
            parent_demo: dict with parent's state (for BC prior preference)

        Returns:
            action: numpy array (num_joints,), torques in [-1, 1]
            info: dict with action metadata
        """
        if self._current_z_vision is None:
            # No observation yet — random action
            return np.random.uniform(-0.3, 0.3, self.num_joints).astype(np.float32), {"mode": "no_obs"}

        # Set prior preference from parent demo (for active inference)
        if parent_demo is not None and "joint_angles" in parent_demo:
            parent_z = torch.tensor(
                parent_demo["joint_angles"][:cfg.VISION_LATENT_DIM],
                dtype=torch.float32, device=self.device
            )
            self.action_model.set_prior_preference(parent_z)

        action, info = self.action_model.select_action(
            z_vision=self._current_z_vision,
            z_proprio=self._current_z_proprio,
            stage=stage,
            rnn_hidden=self._rnn_hidden.squeeze(0) if self._rnn_hidden is not None else None,
            world_model_rnn=self.mdn_rnn,
            vae_loss=self._vae_recon_loss,
            rnn_accuracy=self._rnn_accuracy,
        )

        # Post action planning to workspace
        self.workspace.post("action_planning", {
            "action": action,
            "mode": info["mode"],
        }, salience=np.abs(action).mean())

        self._prev_action = action.copy()
        self._prev_proprio = self._current_z_proprio

        return action, info

    def store_transition(self, obs, action, next_obs, parent_action=None, timestep=0):
        """
        Store a (s, a, s') transition in episodic buffer.
        """
        transition = {
            "vision_flat": obs["vision_flat"] if isinstance(obs["vision_flat"], np.ndarray) else obs["vision_flat"],
            "proprio": obs["proprio"] if isinstance(obs["proprio"], np.ndarray) else obs["proprio"],
            "action": action,
            "next_vision_flat": next_obs["vision_flat"] if isinstance(next_obs["vision_flat"], np.ndarray) else next_obs["vision_flat"],
            "next_proprio": next_obs["proprio"] if isinstance(next_obs["proprio"], np.ndarray) else next_obs["proprio"],
            "timestamp": timestep,
        }

        # Add latent vectors if available
        if self._prev_z_vision is not None:
            transition["z_vision"] = self._prev_z_vision.cpu().numpy()
        if self._prev_z_proprio is not None:
            transition["z_proprio"] = self._prev_z_proprio.cpu().numpy()
        if self._current_z_vision is not None:
            transition["next_z_vision"] = self._current_z_vision.cpu().numpy()
        if parent_action is not None:
            transition["parent_action"] = parent_action

        self.episodic_buffer.add(transition, prediction_error=self._prediction_error)

    def learn(self, step, stage):
        """
        Learning update — called every step, but actual weight updates
        happen only every UPDATE_EVERY_N_STEPS (slow neocortical cycle).

        Args:
            step: global step counter
            stage: current developmental stage
        """
        if step % cfg.UPDATE_EVERY_N_STEPS != 0:
            return
        if not self.episodic_buffer.is_ready:
            return

        # Sample from episodic buffer (prioritized replay)
        batch = self.episodic_buffer.sample_batch_tensors(
            device=self.device
        )
        if batch is None:
            return

        # ─── VAE Update (always active) ───────────────────────────
        if batch["vision_flat"] is not None and batch["proprio"] is not None:
            total_loss, recon_loss, kl_loss = self.vae.compute_loss(
                batch["vision_flat"], batch["proprio"]
            )
            self.vae_optimizer.zero_grad()
            total_loss.backward()
            self.vae_optimizer.step()

            self._vae_recon_loss = recon_loss.item()
            self._vae_kl_loss = kl_loss.item()

        # ─── MDN-RNN Update (Stage 2+) ───────────────────────────
        if stage >= 2 and batch["z_vision"] is not None and batch["next_z_vision"] is not None:
            z_vis = batch["z_vision"]
            z_pro = batch["z_proprio"] if batch["z_proprio"] is not None else torch.zeros(z_vis.size(0), cfg.PROPRIO_LATENT_DIM, device=self.device)
            act = batch["action"]

            mu_next, logvar_next, _ = self.mdn_rnn(z_vis, z_pro, act)
            rnn_loss = MDNRNN.compute_loss(batch["next_z_vision"], mu_next, logvar_next)

            self.rnn_optimizer.zero_grad()
            rnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mdn_rnn.parameters(), cfg.GRAD_CLIP_NORM)
            self.rnn_optimizer.step()

            self._rnn_nll_loss = rnn_loss.item()
            self._rnn_accuracy = self.mdn_rnn.compute_accuracy(
                batch["next_z_vision"], mu_next.detach()
            )

        # ─── Forward Model Update (Stage 3+) ─────────────────────
        if stage >= 3 and batch["proprio"] is not None and batch["next_proprio"] is not None:
            pred_next = self.forward_model(batch["proprio"], batch["action"])
            fm_loss = ForwardModel.compute_loss(pred_next, batch["next_proprio"])

            self.fm_optimizer.zero_grad()
            fm_loss.backward()
            self.fm_optimizer.step()

            self._fm_loss = fm_loss.item()

            # Self-other signal
            self._self_confidence = ForwardModel.self_other_signal(
                self._fm_loss, self._vae_recon_loss
            )

        # ─── Behavioral Cloning Update (Stage 3-4) ───────────────
        if stage in (3, 4) and batch["parent_action"] is not None and batch["z_vision"] is not None:
            z_vis = batch["z_vision"]
            z_pro = batch["z_proprio"] if batch["z_proprio"] is not None else torch.zeros(z_vis.size(0), cfg.PROPRIO_LATENT_DIM, device=self.device)

            bc_loss = self.action_model.compute_bc_loss(z_vis, z_pro, batch["parent_action"])
            self.bc_optimizer.zero_grad()
            bc_loss.backward()
            self.bc_optimizer.step()

            self._bc_loss = bc_loss.item()

            # Anneal BC weight
            self.action_model.anneal_bc_weight()

        # ─── Update RNN hidden state ─────────────────────────────
        if self._current_z_vision is not None and self._prev_action is not None:
            with torch.no_grad():
                z_v = self._current_z_vision.unsqueeze(0)
                z_p = self._current_z_proprio.unsqueeze(0)
                a = torch.tensor(self._prev_action, dtype=torch.float32, device=self.device).unsqueeze(0)
                _, _, self._rnn_hidden = self.mdn_rnn(z_v, z_p, a, self._rnn_hidden)

    def get_metrics(self):
        """Return all current metrics for monitoring and stage transitions."""
        curiosity_stats = self.curiosity.get_stats()
        buffer_stats = self.episodic_buffer.get_stats()

        return {
            "vae_recon_loss": self._vae_recon_loss,
            "vae_kl_loss": self._vae_kl_loss,
            "rnn_nll_loss": self._rnn_nll_loss,
            "rnn_accuracy": self._rnn_accuracy,
            "forward_model_loss": self._fm_loss,
            "bc_loss": self._bc_loss,
            "bc_weight": self.action_model.get_bc_weight(),
            "prediction_error": self._prediction_error,
            "self_confidence": self._self_confidence,
            "buffer_size": len(self.episodic_buffer),
            "active_curiosity_clusters": curiosity_stats["active_clusters"],
            "max_learning_progress": curiosity_stats["max_progress"],
            "mean_learning_progress": curiosity_stats["mean_progress"],
            "curiosity_weight": curiosity_stats["curiosity_weight"],
            "buffer_mean_priority": buffer_stats["mean_priority"],
        }

    def get_models(self):
        """Return dict of all models for checkpointing."""
        return {
            "vae": self.vae,
            "mdn_rnn": self.mdn_rnn,
            "forward_model": self.forward_model,
            "action_model": self.action_model,
        }

    def get_optimizers(self):
        """Return dict of all optimizers for checkpointing."""
        return {
            "vae": self.vae_optimizer,
            "mdn_rnn": self.rnn_optimizer,
            "forward_model": self.fm_optimizer,
            "action_model": self.bc_optimizer,
        }

    def get_z_history(self):
        """Return latent vector history for t-SNE visualization."""
        return self._z_history

    def reset_episode(self):
        """Reset per-episode state (GRU hidden)."""
        self._rnn_hidden = None
        self._prev_z_vision = None
        self._prev_z_proprio = None
        self._prev_action = None
