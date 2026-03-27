"""
OIS Training Loop — Main Entry Point

Orchestrates the full developmental learning loop:
  observe → predict → fail → correct → explore

Usage:
  python train.py                          # full training
  python train.py --max-steps 5000         # short run
  python train.py --resume                 # resume from checkpoint
  python train.py --parent-demo            # validate parent CPG only
  python train.py --no-wandb               # disable wandb logging
"""

import argparse
import sys
import numpy as np
import time

import config as cfg
from environment.simulation import OISEnvironment
from agents.parent_agent import ParentAgent
from agents.infant_agent import InfantAgent
from development.stage_manager import StageManager
from monitoring.dashboard import Dashboard
from monitoring.checkpoint import CheckpointManager


def run_parent_demo(render=True, num_steps=500):
    """Validate parent CPG walking — the Phase 1 hard gate."""
    print("\n" + "=" * 60)
    print("  PHASE 1 HARD GATE: Parent CPG Validation")
    print("=" * 60)

    env = OISEnvironment(render=render)
    parent = ParentAgent(
        num_joints=env.parent.num_actuated,
        timestep=cfg.TIMESTEP,
    )
    result = env.run_parent_demo(num_steps, parent_agent=parent)

    print(f"\n  Steps survived:  {result['steps_survived']} / {num_steps}")
    print(f"  Min height:      {result['min_height']:.3f}")
    print(f"  Fell:            {'YES ✗' if result['fell'] else 'NO ✓'}")
    print(f"  GATE:            {'PASSED ✓' if not result['fell'] else 'FAILED ✗'}")
    print("=" * 60 + "\n")

    env.close()
    return not result["fell"]


def train(args):
    """Main training loop."""
    print("\n" + "=" * 60)
    print("  ONTOGENETIC INTELLIGENCE SYSTEM")
    print("  Developmental Training Loop")
    print("=" * 60 + "\n")

    # ─── Initialize components ────────────────────────────────
    print("  Initializing environment...")
    env = OISEnvironment(render=args.render)

    print("  Initializing parent agent...")
    parent = ParentAgent(
        num_joints=env.parent.num_actuated,
        timestep=cfg.TIMESTEP,
    )

    print("  Initializing infant agent...")
    infant = InfantAgent(
        num_joints=env.infant.num_actuated,
        device=cfg.DEVICE,
    )

    print("  Initializing stage manager...")
    stage_mgr = StageManager()

    print("  Initializing dashboard...")
    dashboard = Dashboard(
        use_wandb=not args.no_wandb,
        log_dir=args.log_dir,
    )

    print("  Initializing checkpoint manager...")
    ckpt_mgr = CheckpointManager(checkpoint_dir=args.checkpoint_dir)

    # ─── Resume from checkpoint if requested ──────────────────
    start_step = 0
    if args.resume:
        start_step = ckpt_mgr.load_latest(
            models=infant.get_models(),
            optimizers=infant.get_optimizers(),
            stage_manager=stage_mgr,
            buffer=infant.episodic_buffer,
            device=cfg.DEVICE,
        )

    print(f"\n  Device:          {cfg.DEVICE}")
    print(f"  Starting step:   {start_step}")
    print(f"  Max steps:       {args.max_steps}")
    print(f"  Starting stage:  {stage_mgr.stage} ({stage_mgr.stage_name})")
    print(f"  Episode length:  {cfg.EPISODE_LENGTH}")
    print(f"  Infant joints:   {env.infant.num_actuated}")
    print(f"  Parent joints:   {env.parent.num_actuated}")
    print()

    # ─── Training loop ────────────────────────────────────────
    global_step = start_step
    num_episodes = args.max_steps // cfg.EPISODE_LENGTH + 1
    start_time = time.time()

    try:
        for episode in range(num_episodes):
            obs = env.reset()
            infant.reset_episode()
            episode_start_step = global_step

            for t in range(cfg.EPISODE_LENGTH):
                if global_step >= args.max_steps:
                    break

                # ── Parent action ──────────────────────
                parent_angles, parent_vels = env.parent.get_joint_states()
                parent_action = parent.get_action(
                    t, current_joint_positions=parent_angles,
                    current_joint_velocities=parent_vels,
                )

                # ── Infant observe ────────────────────
                pred_error = infant.observe(obs)

                # ── Infant act ────────────────────────
                parent_demo = env.get_parent_demonstration(parent_action)
                action, action_info = infant.act(
                    stage=stage_mgr.stage,
                    parent_demo=parent_demo,
                )

                # ── Environment step ──────────────────
                next_obs = env.step(
                    infant_action=action,
                    parent_action=parent_action,
                )

                # ── Store transition ──────────────────
                infant.store_transition(
                    obs=obs,
                    action=action,
                    next_obs=next_obs,
                    parent_action=parent_action,
                    timestep=global_step,
                )

                # ── Learn (slow neocortical cycle) ────
                infant.learn(global_step, stage_mgr.stage)

                # ── Stage transition check ────────────
                metrics = infant.get_metrics()
                metrics["locomotion_efficiency"] = env.compute_locomotion_efficiency()
                metrics["novel_state_count"] = metrics.get("active_curiosity_clusters", 0)

                advanced = stage_mgr.check_transition(metrics, global_step)
                if advanced:
                    print(f"\n  🎯 STAGE TRANSITION at step {global_step}:")
                    print(f"     Stage {stage_mgr.stage - 1} → {stage_mgr.stage} ({stage_mgr.stage_name})")
                    print(f"     Active modules: {stage_mgr.get_active_modules()}\n")

                    # Activate GWT competition at Stage 4
                    if stage_mgr.stage >= 4:
                        infant.workspace.activate_competition()

                    # Update curiosity weight based on stage
                    infant.curiosity.set_curiosity_weight(
                        stage_mgr.get_curiosity_weight()
                    )

                # ── Logging ───────────────────────────
                if global_step % cfg.LOG_EVERY_N_STEPS == 0:
                    metrics["stage"] = stage_mgr.stage
                    metrics["action_mode"] = action_info["mode"]
                    metrics["episode"] = episode
                    dashboard.log_metrics(metrics, global_step)
                    dashboard.print_status(global_step, stage_mgr.stage, metrics)

                # ── Checkpoint ────────────────────────
                if ckpt_mgr.should_save(global_step):
                    ckpt_mgr.save(
                        step=global_step,
                        models=infant.get_models(),
                        optimizers=infant.get_optimizers(),
                        stage_manager=stage_mgr,
                        buffer=infant.episodic_buffer,
                        extra={"bc_weight": infant.action_model.get_bc_weight()},
                    )

                    # Generate visualizations periodically
                    if global_step % 5000 == 0:
                        dashboard.plot_loss_curves()
                        z_hist = infant.get_z_history()
                        if len(z_hist) > 100:
                            dashboard.plot_latent_space(z_hist)

                # ── Advance ───────────────────────────
                obs = next_obs
                global_step += 1

                # Check for fallen infant — reset if needed
                if env.infant.is_fallen():
                    break

            # End of episode
            if global_step >= args.max_steps:
                break

    except KeyboardInterrupt:
        print("\n\n  ⚠ Training interrupted by user.")
    finally:
        # Final save
        print(f"\n  Saving final checkpoint at step {global_step}...")
        ckpt_mgr.save(
            step=global_step,
            models=infant.get_models(),
            optimizers=infant.get_optimizers(),
            stage_manager=stage_mgr,
            buffer=infant.episodic_buffer,
        )

        # Final visualizations
        dashboard.plot_loss_curves()
        z_hist = infant.get_z_history()
        if len(z_hist) > 50:
            dashboard.plot_latent_space(z_hist)

        elapsed = (time.time() - start_time) / 60.0
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"  Steps:      {global_step}")
        print(f"  Episodes:   {episode + 1}")
        print(f"  Final stage: {stage_mgr.stage} ({stage_mgr.stage_name})")
        print(f"  Elapsed:    {elapsed:.1f} minutes")
        print(f"  Buffer:     {len(infant.episodic_buffer)} transitions")
        print(f"  Metrics:    {infant.get_metrics()}")
        print(f"{'='*60}\n")

        dashboard.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Ontogenetic Intelligence System — Training")
    parser.add_argument("--max-steps", type=int, default=100_000, help="Maximum training steps")
    parser.add_argument("--render", action="store_true", help="Render PyBullet GUI")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--parent-demo", action="store_true", help="Run parent demo only")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log output directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    args = parser.parse_args()

    if args.parent_demo:
        passed = run_parent_demo(render=args.render)
        sys.exit(0 if passed else 1)

    train(args)


if __name__ == "__main__":
    main()
