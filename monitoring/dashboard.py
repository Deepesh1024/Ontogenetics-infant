"""
Monitoring Dashboard — wandb + Local Fallback

Logs all metrics every 100 timesteps. Uses wandb if available/configured,
otherwise falls back to CSV + matplotlib.
"""

import os
import csv
import time
import numpy as np

import config as cfg

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Dashboard:
    """
    Monitoring dashboard for the OIS training loop.
    Logs metrics, generates plots, tracks training health.
    """

    def __init__(self, project_name="ontogenetic-intelligence", use_wandb=True, log_dir="logs"):
        self.log_dir = log_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self._history = []
        self._csv_path = os.path.join(log_dir, "metrics.csv")
        self._csv_writer = None
        self._csv_file = None
        self._initialized = False
        self._start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)

        if self.use_wandb:
            try:
                wandb.init(project=project_name, config=self._get_config_dict())
                print(f"  ✓ wandb initialized (project: {project_name})")
            except Exception as e:
                print(f"  ✗ wandb init failed: {e}, falling back to CSV")
                self.use_wandb = False

    def log_metrics(self, metrics, step):
        """
        Log metrics at current step.

        Args:
            metrics: dict of metric_name → value
            step: global timestep
        """
        # Add step and elapsed time
        metrics["step"] = step
        metrics["elapsed_minutes"] = (time.time() - self._start_time) / 60.0

        # Store in memory
        self._history.append(dict(metrics))

        # Log to wandb
        if self.use_wandb:
            try:
                wandb.log(metrics, step=step)
            except Exception:
                pass  # silently continue if wandb fails

        # Log to CSV
        self._log_csv(metrics)

    def _log_csv(self, metrics):
        """Append metrics to CSV file."""
        if self._csv_writer is None:
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=sorted(metrics.keys())
            )
            self._csv_writer.writeheader()

        # Handle missing keys gracefully
        row = {k: metrics.get(k, "") for k in self._csv_writer.fieldnames}
        try:
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        except ValueError:
            # New columns appeared — restart CSV
            self._csv_file.close()
            self._csv_file = open(self._csv_path, "w", newline="")
            all_keys = sorted(set(self._csv_writer.fieldnames) | set(metrics.keys()))
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=all_keys)
            self._csv_writer.writeheader()
            for h in self._history:
                self._csv_writer.writerow({k: h.get(k, "") for k in all_keys})
            self._csv_file.flush()

    def plot_loss_curves(self, save_path=None):
        """Generate loss curves for all tracked losses."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if not self._history:
            return

        steps = [h["step"] for h in self._history]
        loss_keys = [k for k in self._history[0].keys()
                     if "loss" in k.lower() or "nll" in k.lower()]

        fig, ax = plt.subplots(figsize=(12, 6))
        for key in loss_keys:
            values = [h.get(key, np.nan) for h in self._history]
            ax.plot(steps, values, label=key, alpha=0.8)

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("OIS Training Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = save_path or os.path.join(self.log_dir, "loss_curves.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if self.use_wandb:
            try:
                wandb.log({"loss_curves": wandb.Image(save_path)})
            except Exception:
                pass

    def plot_latent_space(self, z_history, labels=None, save_path=None):
        """Generate t-SNE visualization of latent space."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
        except ImportError:
            return

        if len(z_history) < 50:
            return

        z_array = np.array(z_history[-2000:])  # last 2000 for performance

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        z_2d = tsne.fit_transform(z_array)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=range(len(z_2d)),
                            cmap="viridis", alpha=0.5, s=5)
        plt.colorbar(scatter, label="Temporal order")
        ax.set_title("Latent Space t-SNE")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        save_path = save_path or os.path.join(self.log_dir, "latent_tsne.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if self.use_wandb:
            try:
                wandb.log({"latent_tsne": wandb.Image(save_path)})
            except Exception:
                pass

    def print_status(self, step, stage, metrics):
        """Print a concise status line to console."""
        vae_loss = metrics.get("vae_recon_loss", float("nan"))
        rnn_loss = metrics.get("rnn_nll_loss", float("nan"))
        fm_loss = metrics.get("forward_model_loss", float("nan"))
        buf_size = metrics.get("buffer_size", 0)
        elapsed = (time.time() - self._start_time) / 60.0

        print(
            f"  Step {step:>8d} | Stage {stage} "
            f"| VAE {vae_loss:.4f} | RNN {rnn_loss:.4f} "
            f"| FM {fm_loss:.4f} | Buf {buf_size:>6d} "
            f"| {elapsed:.1f}min"
        )

    def _get_config_dict(self):
        """Extract config values for wandb logging."""
        return {
            k: v for k, v in vars(cfg).items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool))
        }

    def close(self):
        """Finalize logging."""
        if self._csv_file:
            self._csv_file.close()
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass
