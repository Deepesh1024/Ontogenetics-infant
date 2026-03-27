"""
Checkpoint System — Save/Load Training State

Auto-saves every 1000 steps: all model weights, optimizer states,
stage, step counter, and buffer snapshot. Supports resume from crash.
"""

import os
import json
import torch
import numpy as np


class CheckpointManager:
    """
    Manages saving and loading of full training state.
    Every 1000 steps, saves everything needed to resume.
    """

    SAVE_EVERY = 1000

    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._last_save_step = -1

    def should_save(self, step):
        """Check if we should save at this step."""
        return step > 0 and step % self.SAVE_EVERY == 0 and step != self._last_save_step

    def save(self, step, models, optimizers, stage_manager, buffer, extra=None):
        """
        Save full training state.

        Args:
            step: global step counter
            models: dict of name → nn.Module
            optimizers: dict of name → optimizer
            stage_manager: StageManager instance
            buffer: EpisodicBuffer instance
            extra: optional dict of additional state
        """
        checkpoint = {
            "step": step,
            "stage": stage_manager.stage,
            "stage_history": stage_manager.history,
        }

        # Save model weights
        for name, model in models.items():
            if model is not None and hasattr(model, "state_dict"):
                checkpoint[f"model_{name}"] = model.state_dict()

        # Save optimizer states
        for name, opt in optimizers.items():
            if opt is not None:
                checkpoint[f"optimizer_{name}"] = opt.state_dict()

        # Save extra state
        if extra:
            checkpoint["extra"] = extra

        # Save checkpoint file
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{step:08d}.pt")
        torch.save(checkpoint, path)

        # Save buffer separately (can be large)
        buffer_path = os.path.join(self.checkpoint_dir, f"buffer_{step:08d}.npz")
        self._save_buffer(buffer, buffer_path)

        # Save metadata
        meta_path = os.path.join(self.checkpoint_dir, "latest.json")
        with open(meta_path, "w") as f:
            json.dump({
                "step": step,
                "stage": stage_manager.stage,
                "checkpoint_path": path,
                "buffer_path": buffer_path,
            }, f, indent=2)

        self._last_save_step = step
        print(f"  💾 Checkpoint saved at step {step} (Stage {stage_manager.stage})")

        # Clean old checkpoints (keep last 3)
        self._cleanup(keep=3)

    def load_latest(self, models, optimizers, stage_manager, buffer, device=None):
        """
        Load the most recent checkpoint.

        Returns:
            step: the step to resume from, or 0 if no checkpoint found
        """
        meta_path = os.path.join(self.checkpoint_dir, "latest.json")
        if not os.path.exists(meta_path):
            print("  No checkpoint found, starting from scratch.")
            return 0

        with open(meta_path, "r") as f:
            meta = json.load(f)

        checkpoint_path = meta["checkpoint_path"]
        buffer_path = meta["buffer_path"]

        if not os.path.exists(checkpoint_path):
            print(f"  Checkpoint file missing: {checkpoint_path}")
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)

        # Restore model weights
        for name, model in models.items():
            key = f"model_{name}"
            if key in checkpoint and model is not None:
                model.load_state_dict(checkpoint[key])

        # Restore optimizer states
        for name, opt in optimizers.items():
            key = f"optimizer_{name}"
            if key in checkpoint and opt is not None:
                opt.load_state_dict(checkpoint[key])

        # Restore stage
        stage_manager.stage = checkpoint.get("stage", 1)
        if "stage_history" in checkpoint:
            stage_manager._stage_history = checkpoint["stage_history"]

        # Restore buffer
        if os.path.exists(buffer_path):
            self._load_buffer(buffer, buffer_path)

        step = checkpoint["step"]
        print(f"  ✓ Resumed from step {step} (Stage {stage_manager.stage})")
        return step

    def _save_buffer(self, buffer, path):
        """Save episodic buffer to compressed numpy file."""
        if len(buffer) == 0:
            return

        # Extract arrays from buffer entries
        data = {}
        keys = None
        for i, transition in enumerate(buffer._buffer):
            if keys is None:
                keys = [k for k, v in transition.items()
                        if isinstance(v, (np.ndarray, int, float))]
            for k in keys:
                if k not in data:
                    data[k] = []
                v = transition.get(k)
                if isinstance(v, np.ndarray):
                    data[k].append(v)
                elif isinstance(v, (int, float)):
                    data[k].append(np.array([v]))

        # Stack and save
        arrays = {}
        for k, v in data.items():
            try:
                arrays[k] = np.stack(v)
            except ValueError:
                pass  # skip non-stackable

        arrays["priorities"] = np.array(list(buffer._priorities), dtype=np.float32)
        np.savez_compressed(path, **arrays)

    def _load_buffer(self, buffer, path):
        """Load episodic buffer from numpy file."""
        try:
            data = np.load(path, allow_pickle=False)
            keys = [k for k in data.keys() if k != "priorities"]
            priorities = data.get("priorities", None)

            n = len(data[keys[0]]) if keys else 0
            for i in range(n):
                transition = {}
                for k in keys:
                    arr = data[k][i]
                    if arr.shape == (1,):
                        transition[k] = float(arr[0])
                    else:
                        transition[k] = arr
                priority = float(priorities[i]) if priorities is not None else 1.0
                buffer.add(transition, prediction_error=priority)
        except Exception as e:
            print(f"  ⚠ Failed to load buffer: {e}")

    def _cleanup(self, keep=3):
        """Remove old checkpoints, keeping only the last `keep`."""
        files = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ])
        for f in files[:-keep]:
            os.remove(os.path.join(self.checkpoint_dir, f))
            # Also remove corresponding buffer file
            buf_f = f.replace("checkpoint_", "buffer_").replace(".pt", ".npz")
            buf_path = os.path.join(self.checkpoint_dir, buf_f)
            if os.path.exists(buf_path):
                os.remove(buf_path)
