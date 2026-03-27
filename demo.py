"""
OIS Minimal Demo — Runs the full developmental loop with mock physics.
Shows all components working together: VAE, MDN-RNN, curiosity, stage
transitions, global workspace, and behavioral cloning.

No PyBullet required — uses a simulated physics environment.
"""

import numpy as np
import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from models.vae import VAEWorldModel
from models.mdn_rnn import MDNRNN
from models.forward_model import ForwardModel
from models.action_model import ActionModel
from memory.episodic_buffer import EpisodicBuffer
from cognition.curiosity import CuriosityEngine
from cognition.global_workspace import MessageBus
from development.stage_manager import StageManager


# ─── Mock Physics Environment ────────────────────────────────────────────────
class MockEnvironment:
    """
    Simulated physics environment producing structured observations
    without PyBullet. Uses sinusoidal dynamics for realistic-looking data.
    """
    def __init__(self, num_joints=17):
        self.num_joints = num_joints
        self.t = 0
        self.infant_pos = np.array([0, 0, 0.9], dtype=np.float32)
        self.parent_pos = np.array([2, 0, 0.9], dtype=np.float32)
        self._parent_phase = 0.0

    def reset(self):
        self.t = 0
        self.infant_pos = np.array([0, 0, 0.9], dtype=np.float32)
        return self._get_obs()

    def step(self, action):
        self.t += 1
        self._parent_phase += 0.1

        # Simulate infant movement based on action
        self.infant_pos[0] += action[0] * 0.001
        self.infant_pos[2] = max(0.3, 0.9 + np.sin(self.t * 0.05) * 0.05)

        # Parent walks forward
        self.parent_pos[0] += 0.005

        return self._get_obs()

    def _get_obs(self):
        # Vision: structured pattern with temporal variation
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cx = int(32 + 10 * np.sin(self.t * 0.1))
        cy = int(32 + 10 * np.cos(self.t * 0.1))
        img[max(0,cy-5):min(64,cy+5), max(0,cx-5):min(64,cx+5)] = [200, 100, 50]
        # Add ground
        img[48:, :, 1] = 100

        # Proprioception: oscillating joint angles
        angles = np.array([np.sin(self.t * 0.05 + i * 0.3) * 0.5 for i in range(self.num_joints)], dtype=np.float32)
        vels = np.array([np.cos(self.t * 0.05 + i * 0.3) * 0.3 for i in range(self.num_joints)], dtype=np.float32)
        proprio = np.concatenate([angles, vels])

        # Touch
        touch = np.random.rand(6).astype(np.float32) * 0.1

        # Parent CPG action
        parent_action = np.array([
            np.sin(self._parent_phase + i * np.pi / self.num_joints) * 0.6
            for i in range(self.num_joints)
        ], dtype=np.float32)

        parent_angles = np.array([
            np.sin(self._parent_phase + i * 0.4) * 0.5
            for i in range(self.num_joints)
        ], dtype=np.float32)
        parent_vels = np.array([
            np.cos(self._parent_phase + i * 0.4) * 0.3
            for i in range(self.num_joints)
        ], dtype=np.float32)

        return {
            "vision": img,
            "vision_flat": img.flatten().astype(np.float32) / 255.0,
            "proprio": proprio,
            "touch": touch,
            "parent_obs": np.concatenate([parent_angles, parent_vels]),
            "parent_action": parent_action,
            "infant_pos": self.infant_pos.copy(),
            "infant_height": float(self.infant_pos[2]),
            "parent_pos": self.parent_pos.copy(),
            "parent_height": float(self.parent_pos[2]),
            "timestep": self.t,
        }


# ─── Demo Runner ─────────────────────────────────────────────────────────────
def run_demo(max_steps=300):
    device = cfg.DEVICE
    num_joints = 17

    print()
    print("╔" + "═" * 60 + "╗")
    print("║  ONTOGENETIC INTELLIGENCE SYSTEM — MINIMAL DEMO          ║")
    print("║  Developmental loop: observe → predict → fail → correct  ║")
    print("╚" + "═" * 60 + "╝")
    print()
    print(f"  Device: {device}")
    print(f"  Steps:  {max_steps}")
    print()

    # ── Initialize ────────────────────────────────────────────
    print("  ⚙ Initializing components...")
    env = MockEnvironment(num_joints=num_joints)

    vae = VAEWorldModel().to(device)
    mdn_rnn = MDNRNN().to(device)
    forward_model = ForwardModel().to(device)
    action_model = ActionModel(action_dim=num_joints).to(device)
    buffer = EpisodicBuffer()
    curiosity = CuriosityEngine()
    workspace = MessageBus()
    stage_mgr = StageManager()

    # Register GWT modules
    for name in ["sensory", "memory", "curiosity", "action_planning"]:
        workspace.register_module(name)

    # Optimizers
    vae_opt = torch.optim.Adam(vae.parameters(), lr=cfg.VAE_LR)
    rnn_opt = torch.optim.Adam(mdn_rnn.parameters(), lr=cfg.RNN_LR)
    fm_opt = torch.optim.Adam(forward_model.parameters(), lr=cfg.FM_LR)
    bc_opt = torch.optim.Adam(action_model.parameters(), lr=cfg.VAE_LR)

    print("  ✓ All components initialized")
    print()

    # ── Column headers ───────────────────────────────────────
    print("  ┌─────────┬───────┬──────────┬──────────┬──────────┬──────────┬────────┬─────────┐")
    print("  │  Step   │ Stage │ VAE Loss │ RNN NLL  │  FM Loss │ BC Loss  │ Buffer │ Curious │")
    print("  ├─────────┼───────┼──────────┼──────────┼──────────┼──────────┼────────┼─────────┤")

    # ── Run ───────────────────────────────────────────────────
    obs = env.reset()
    rnn_hidden = None
    prev_z_vis = None
    prev_z_pro = None
    prev_obs = None

    vae_loss_val = float("inf")
    rnn_nll_val = float("inf")
    fm_loss_val = float("inf")
    bc_loss_val = 0.0
    rnn_accuracy = 0.0

    start = time.time()

    for step in range(1, max_steps + 1):
        # Convert obs to tensors
        vision_t = torch.tensor(obs["vision_flat"], dtype=torch.float32, device=device).unsqueeze(0)
        proprio_t = torch.tensor(obs["proprio"], dtype=torch.float32, device=device).unsqueeze(0)

        # ── ENCODE (VAE compression to subconscious) ─────────
        with torch.no_grad():
            z_vis, z_pro, mu, logvar = vae.encode(vision_t, proprio_t)

        # Prediction error
        pred_error = vae.get_prediction_error(
            vision_t.squeeze(0), proprio_t.squeeze(0)
        )

        # ── POST to Global Workspace ─────────────────────────
        workspace.post("sensory", {
            "z": z_vis.cpu().numpy(),
            "error": pred_error,
        }, salience=pred_error)

        curiosity.update(z_vis.squeeze(0).cpu().numpy(), pred_error)
        workspace.post("curiosity", {
            "reward": curiosity.get_intrinsic_reward(z_vis.squeeze(0).cpu().numpy()),
        }, salience=curiosity.get_salience())

        workspace.compete()

        # ── ACT ──────────────────────────────────────────────
        action, info = action_model.select_action(
            z_vis.squeeze(0), z_pro.squeeze(0),
            stage=stage_mgr.stage,
            rnn_hidden=rnn_hidden.squeeze(0) if rnn_hidden is not None else None,
            world_model_rnn=mdn_rnn,
            vae_loss=vae_loss_val,
            rnn_accuracy=rnn_accuracy,
        )

        workspace.post("action_planning", {
            "action": action,
        }, salience=np.abs(action).mean())

        # ── STEP environment ──────────────────────────────────
        next_obs = env.step(action)

        # ── STORE transition ─────────────────────────────────
        transition = {
            "vision_flat": obs["vision_flat"],
            "proprio": obs["proprio"],
            "action": action,
            "z_vision": z_vis.squeeze(0).detach().cpu().numpy(),
            "z_proprio": z_pro.squeeze(0).detach().cpu().numpy(),
            "next_vision_flat": next_obs["vision_flat"],
            "next_proprio": next_obs["proprio"],
            "parent_action": obs["parent_action"],
            "timestamp": step,
        }

        # Compute next z for RNN training
        with torch.no_grad():
            nv = torch.tensor(next_obs["vision_flat"], dtype=torch.float32, device=device).unsqueeze(0)
            np_ = torch.tensor(next_obs["proprio"], dtype=torch.float32, device=device).unsqueeze(0)
            next_z_vis, next_z_pro, _, _ = vae.encode(nv, np_)
        transition["next_z_vision"] = next_z_vis.squeeze(0).cpu().numpy()
        transition["next_z_proprio"] = next_z_pro.squeeze(0).cpu().numpy()

        buffer.add(transition, prediction_error=pred_error)

        # ── LEARN (every 16 steps — slow neocortical cycle) ──
        if step % cfg.UPDATE_EVERY_N_STEPS == 0 and buffer.is_ready:
            batch = buffer.sample_batch_tensors(device=device)

            if batch and batch["vision_flat"] is not None:
                # VAE update
                total_loss, recon_loss, kl_loss = vae.compute_loss(batch["vision_flat"], batch["proprio"])
                vae_opt.zero_grad()
                total_loss.backward()
                vae_opt.step()
                vae_loss_val = recon_loss.item()

                # MDN-RNN update (Stage 2+)
                if stage_mgr.stage >= 2 and batch["z_vision"] is not None:
                    mu_next, logvar_next, _ = mdn_rnn(batch["z_vision"], batch["z_proprio"], batch["action"])
                    rnn_loss = MDNRNN.compute_loss(batch["next_z_vision"], mu_next, logvar_next)
                    rnn_opt.zero_grad()
                    rnn_loss.backward()
                    torch.nn.utils.clip_grad_norm_(mdn_rnn.parameters(), cfg.GRAD_CLIP_NORM)
                    rnn_opt.step()
                    rnn_nll_val = rnn_loss.item()
                    rnn_accuracy = mdn_rnn.compute_accuracy(batch["next_z_vision"], mu_next.detach())

                # Forward model (Stage 3+)
                if stage_mgr.stage >= 3 and batch["next_proprio"] is not None:
                    pred_next = forward_model(batch["proprio"], batch["action"])
                    fm_loss = ForwardModel.compute_loss(pred_next, batch["next_proprio"])
                    fm_opt.zero_grad()
                    fm_loss.backward()
                    fm_opt.step()
                    fm_loss_val = fm_loss.item()

                # Behavioral cloning (Stage 3-4)
                if stage_mgr.stage in (3, 4) and batch["parent_action"] is not None:
                    bc_loss = action_model.compute_bc_loss(
                        batch["z_vision"], batch["z_proprio"], batch["parent_action"]
                    )
                    bc_opt.zero_grad()
                    bc_loss.backward()
                    bc_opt.step()
                    bc_loss_val = bc_loss.item()
                    action_model.anneal_bc_weight()

        # ── UPDATE RNN hidden state ───────────────────────────
        with torch.no_grad():
            action_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, rnn_hidden = mdn_rnn(z_vis, z_pro, action_t, rnn_hidden)

        # ── STAGE TRANSITIONS ────────────────────────────────
        metrics = {
            "vae_recon_loss": vae_loss_val,
            "buffer_size": len(buffer),
            "rnn_accuracy": rnn_accuracy,
            "rnn_nll_loss": rnn_nll_val,
            "locomotion_efficiency": 0.35 if step > 200 else 0.1,  # simulated
            "forward_model_loss": fm_loss_val,
            "active_curiosity_clusters": curiosity.count_active_clusters(),
            "novel_state_count": curiosity.count_active_clusters(),
        }

        advanced = stage_mgr.check_transition(metrics, step)
        if advanced:
            print(f"  │{'':>9}│{'':>7}│{'':>10}│{'':>10}│{'':>10}│{'':>10}│{'':>8}│{'':>9}│")
            print(f"  │  🎯 STAGE {stage_mgr.stage - 1} → {stage_mgr.stage}: {stage_mgr.stage_name:<40s}   │")
            print(f"  │{'':>9}│{'':>7}│{'':>10}│{'':>10}│{'':>10}│{'':>10}│{'':>8}│{'':>9}│")
            if stage_mgr.stage >= 4:
                workspace.activate_competition()
                curiosity.set_curiosity_weight(stage_mgr.get_curiosity_weight())

        # ── PRINT status ─────────────────────────────────────
        if step % 10 == 0:
            curious_active = curiosity.count_active_clusters()
            print(
                f"  │ {step:>6d}  │   {stage_mgr.stage}   "
                f"│ {vae_loss_val:>8.4f} │ {rnn_nll_val:>8.4f} "
                f"│ {fm_loss_val:>8.4f} │ {bc_loss_val:>8.4f} "
                f"│ {len(buffer):>6d} │  {curious_active:>3d}/16 │"
            )

        prev_z_vis = z_vis
        prev_z_pro = z_pro
        prev_obs = obs
        obs = next_obs

    elapsed = time.time() - start

    print("  └─────────┴───────┴──────────┴──────────┴──────────┴──────────┴────────┴─────────┘")
    print()

    # ── Summary ───────────────────────────────────────────────
    gw_mode = "COMPETITIVE (winner-takes-all)" if workspace.is_competitive else "PASSIVE (broadcast-all)"
    cur_stats = curiosity.get_stats()

    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║                    DEVELOPMENTAL REPORT                     ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  Total steps:          {max_steps:>6d}                            ║")
    print(f"  ║  Elapsed:              {elapsed:>6.1f}s                            ║")
    print(f"  ║  Final stage:          {stage_mgr.stage} ({stage_mgr.stage_name:<26s})  ║")
    print(f"  ║  VAE recon loss:       {vae_loss_val:>8.5f}                         ║")
    print(f"  ║  RNN NLL loss:         {rnn_nll_val:>8.4f}                         ║")
    print(f"  ║  Forward model loss:   {fm_loss_val:>8.5f}                         ║")
    print(f"  ║  BC weight:            {action_model.get_bc_weight():>8.4f}                         ║")
    print(f"  ║  Buffer size:          {len(buffer):>6d}                            ║")
    print(f"  ║  Active clusters:      {cur_stats['active_clusters']:>3d}/16                            ║")
    print(f"  ║  Max learning progress:{cur_stats['max_progress']:>8.4f}                         ║")
    print(f"  ║  Global Workspace:     {gw_mode:<30s}     ║")
    print(f"  ║  Self-confidence:      {ForwardModel.self_other_signal(fm_loss_val, vae_loss_val):>8.4f}                         ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║  Stage transitions:                                        ║")
    for ts_step, ts_stage in stage_mgr.history:
        name = StageManager.STAGE_NAMES.get(ts_stage, "?")
        print(f"  ║    Step {ts_step:>6d} → Stage {ts_stage}: {name:<32s}   ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run_demo(max_steps=steps)
