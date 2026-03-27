"""
Episodic Buffer — Complementary Learning Systems (Hippocampal Analog)

Fast-learning episodic memory with prioritized replay.
Stores every recent experience with high fidelity.
Priority = prediction error: higher surprise → more replay.

The neocortical world model (VAE, MDN-RNN) is updated SLOWLY via replay
from this buffer every 16 steps — not online at every timestep.
"""

import numpy as np
from collections import deque

import config as cfg


class EpisodicBuffer:
    """
    Circular buffer with prioritized experience replay.
    Capacity: 50,000 transitions.
    """

    def __init__(self, capacity=None):
        self.capacity = capacity or cfg.BUFFER_CAPACITY
        self._buffer = deque(maxlen=self.capacity)
        self._priorities = deque(maxlen=self.capacity)

    def add(self, transition, prediction_error=1.0):
        """
        Store a transition. Always stores regardless of error magnitude.

        Args:
            transition: dict with keys:
                - vision_flat: (12288,) float32
                - proprio: (34,) float32
                - action: (17,) float32
                - z_vision: (64,) float32
                - z_proprio: (32,) float32
                - next_vision_flat: (12288,) float32
                - next_proprio: (34,) float32
                - next_z_vision: (64,) float32
                - parent_action: (17,) float32 (optional)
                - timestamp: int
            prediction_error: float, used as replay priority
        """
        self._buffer.append(transition)
        self._priorities.append(max(prediction_error, 1e-6))

    def sample(self, batch_size=None):
        """
        Sample transitions weighted by priority (prediction error).
        Higher error = more frequent replay = surprise drives learning.

        Args:
            batch_size: number of transitions to sample

        Returns:
            list of transition dicts
        """
        batch_size = batch_size or cfg.REPLAY_BATCH_SIZE
        batch_size = min(batch_size, len(self._buffer))

        if batch_size == 0:
            return []

        # Convert priorities to sampling probabilities
        priorities = np.array(self._priorities, dtype=np.float64)
        probs = priorities / (priorities.sum() + 1e-10)

        # Weighted sampling without replacement
        indices = np.random.choice(
            len(self._buffer),
            size=batch_size,
            replace=False,
            p=probs,
        )

        return [self._buffer[i] for i in indices]

    def sample_batch_tensors(self, batch_size=None, device=None):
        """
        Sample and collate into PyTorch tensors for training.

        Returns:
            dict of batched tensors, or None if buffer too small
        """
        import torch

        batch_size = batch_size or cfg.REPLAY_BATCH_SIZE
        if len(self._buffer) < batch_size:
            return None

        transitions = self.sample(batch_size)

        def stack(key):
            arrays = [t[key] for t in transitions if key in t]
            if not arrays:
                return None
            return torch.tensor(np.stack(arrays), dtype=torch.float32, device=device)

        return {
            "vision_flat": stack("vision_flat"),
            "proprio": stack("proprio"),
            "action": stack("action"),
            "z_vision": stack("z_vision"),
            "z_proprio": stack("z_proprio"),
            "next_vision_flat": stack("next_vision_flat"),
            "next_proprio": stack("next_proprio"),
            "next_z_vision": stack("next_z_vision"),
            "parent_action": stack("parent_action"),
        }

    def update_priority(self, index, new_priority):
        """Update the priority of a specific transition."""
        if 0 <= index < len(self._priorities):
            self._priorities[index] = max(new_priority, 1e-6)

    def get_recent(self, n=10):
        """Get the N most recent transitions."""
        n = min(n, len(self._buffer))
        return [self._buffer[-i - 1] for i in range(n)]

    def get_high_surprise(self, n=10):
        """Get the N highest-priority (most surprising) transitions."""
        if len(self._buffer) == 0:
            return []
        priorities = np.array(self._priorities)
        top_indices = np.argsort(priorities)[-n:][::-1]
        return [self._buffer[i] for i in top_indices]

    def __len__(self):
        return len(self._buffer)

    @property
    def is_ready(self):
        """Buffer has enough transitions for meaningful replay."""
        return len(self._buffer) >= cfg.REPLAY_BATCH_SIZE

    @property
    def mean_priority(self):
        if len(self._priorities) == 0:
            return 0.0
        return float(np.mean(self._priorities))

    def get_stats(self):
        """Summary statistics for monitoring."""
        if len(self._priorities) == 0:
            return {"size": 0, "mean_priority": 0, "max_priority": 0, "min_priority": 0}
        priorities = np.array(self._priorities)
        return {
            "size": len(self._buffer),
            "mean_priority": float(priorities.mean()),
            "max_priority": float(priorities.max()),
            "min_priority": float(priorities.min()),
            "std_priority": float(priorities.std()),
        }
