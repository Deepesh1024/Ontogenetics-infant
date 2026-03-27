"""
Developmental Stage Manager

Automatic stage transitions based on measurable thresholds.
Controls which modules are active at each stage.

Stage 1: Sensory Exploration (random + VAE)
Stage 2: Pattern Recognition (+ MDN-RNN)
Stage 3: Motor Control via Imitation (+ behavioral cloning)
Stage 4: Curiosity-Driven Exploration (+ curiosity engine)
Stage 5: Active Inference and Planning (BC removed, active inference)
"""

import config as cfg


class StageManager:
    """
    Manages developmental stage transitions and module activation.
    You do not advance to the next stage until thresholds are met.
    """

    STAGE_NAMES = {
        1: "Sensory Exploration",
        2: "Pattern Recognition",
        3: "Motor Control (Imitation)",
        4: "Curiosity-Driven Exploration",
        5: "Active Inference & Planning",
    }

    def __init__(self):
        self.stage = 1
        self._stage_history = [(0, 1)]  # (step, stage) transitions
        self._metrics_at_transition = {}

    def check_transition(self, metrics, current_step=0):
        """
        Check if we should advance to the next stage.

        Args:
            metrics: dict with keys:
                - vae_recon_loss: float
                - buffer_size: int
                - rnn_accuracy: float
                - rnn_nll_loss: float
                - locomotion_efficiency: float
                - forward_model_loss: float
                - active_curiosity_clusters: int
                - novel_state_count: int
            current_step: int, for logging

        Returns:
            advanced: bool, True if stage changed
        """
        old_stage = self.stage

        if self.stage == 1:
            vae_ok = metrics.get("vae_recon_loss", float("inf")) < cfg.STAGE1_VAE_LOSS_THRESHOLD
            buffer_ok = metrics.get("buffer_size", 0) >= cfg.STAGE1_MIN_BUFFER_SIZE
            if vae_ok and buffer_ok:
                self.stage = 2

        elif self.stage == 2:
            rnn_ok = metrics.get("rnn_accuracy", 0.0) > cfg.STAGE2_RNN_ACCURACY_THRESHOLD
            if rnn_ok:
                self.stage = 3

        elif self.stage == 3:
            loco_ok = metrics.get("locomotion_efficiency", 0.0) >= cfg.STAGE3_LOCOMOTION_EFFICIENCY
            fm_ok = metrics.get("forward_model_loss", float("inf")) < cfg.STAGE3_FM_LOSS_THRESHOLD
            if loco_ok and fm_ok:
                self.stage = 4

        elif self.stage == 4:
            clusters_ok = metrics.get("active_curiosity_clusters", 0) >= cfg.STAGE4_MIN_NOVEL_CLUSTERS
            novel_ok = metrics.get("novel_state_count", 0) > 0
            if clusters_ok and novel_ok:
                self.stage = 5

        # Stage 5 is terminal — no further transitions

        if self.stage != old_stage:
            self._stage_history.append((current_step, self.stage))
            self._metrics_at_transition[self.stage] = dict(metrics)
            return True

        return False

    def get_active_modules(self):
        """
        Returns list of module names that should be active at current stage.
        """
        modules = ["vae", "episodic_buffer"]  # always active

        if self.stage >= 2:
            modules.append("mdn_rnn")

        if self.stage >= 3:
            modules.extend(["behavioral_cloning", "forward_model"])

        if self.stage >= 4:
            modules.extend(["curiosity", "global_workspace_competition"])

        if self.stage >= 5:
            modules.append("active_inference")
            # BC removed at stage 5
            if "behavioral_cloning" in modules:
                modules.remove("behavioral_cloning")

        return modules

    def is_module_active(self, module_name):
        """Check if a specific module should be active."""
        return module_name in self.get_active_modules()

    def get_curiosity_weight(self):
        """Get the curiosity intrinsic reward weight for current stage."""
        if self.stage < 4:
            return 0.0
        elif self.stage == 4:
            return 0.3  # start with mix of intrinsic + extrinsic
        else:
            return 1.0  # full intrinsic at stage 5

    def get_bc_active(self):
        """Whether behavioral cloning should be applied."""
        return self.stage in (3, 4)  # active in stage 3-4, removed at 5

    def get_action_mode(self):
        """Get the current action selection mode."""
        if self.stage == 1:
            return "random"
        elif self.stage <= 3:
            return "behavioral_cloning"
        elif self.stage == 4:
            return "mixed"  # curiosity + annealing BC
        else:
            return "active_inference"

    @property
    def stage_name(self):
        return self.STAGE_NAMES.get(self.stage, f"Unknown Stage {self.stage}")

    @property
    def history(self):
        return self._stage_history

    def get_stats(self):
        """Summary for monitoring."""
        return {
            "current_stage": self.stage,
            "stage_name": self.stage_name,
            "active_modules": self.get_active_modules(),
            "action_mode": self.get_action_mode(),
            "curiosity_weight": self.get_curiosity_weight(),
            "transitions": self._stage_history,
        }
