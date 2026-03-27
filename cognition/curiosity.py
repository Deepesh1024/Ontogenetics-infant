"""
Curiosity Engine — Learning Progress (Oudeyer et al., 2007)

The agent maximizes the RATE at which its prediction error is DECREASING,
not the absolute level of error. This produces three natural zones:
  - Mastered (error ≈ 0, progress ≈ 0): boring → move away
  - Chaotic (error high, not decreasing): overwhelming → move away
  - Zone of proximal development (error high AND decreasing): curious → stay

Online K-means clusters the latent space into 16 regions.
Per-cluster learning progress drives intrinsic reward.
"""

import numpy as np
from collections import defaultdict

import config as cfg


class OnlineKMeans:
    """
    Online K-means clustering of the latent space.
    Clusters update incrementally as new data arrives.
    """

    def __init__(self, n_clusters=None, dim=None):
        self.n_clusters = n_clusters or cfg.NUM_CURIOSITY_CLUSTERS
        self.dim = dim or cfg.VISION_LATENT_DIM
        self.centers = np.random.randn(self.n_clusters, self.dim).astype(np.float32) * 0.1
        self.counts = np.ones(self.n_clusters, dtype=np.float32)  # avoid div by zero
        self._initialized = False

    def assign(self, z):
        """
        Assign a latent vector to its nearest cluster.

        Args:
            z: (dim,) numpy array

        Returns:
            cluster_id: int
        """
        z = np.asarray(z, dtype=np.float32).flatten()
        distances = np.linalg.norm(self.centers - z, axis=1)
        return int(np.argmin(distances))

    def update(self, z, cluster_id=None):
        """
        Online update of cluster center using running mean.

        Args:
            z: (dim,) numpy array
            cluster_id: pre-computed cluster assignment (or None to compute)

        Returns:
            cluster_id: assigned cluster
        """
        z = np.asarray(z, dtype=np.float32).flatten()

        if cluster_id is None:
            cluster_id = self.assign(z)

        # Online mean update: center = center + (z - center) / count
        self.counts[cluster_id] += 1
        lr = 1.0 / self.counts[cluster_id]
        self.centers[cluster_id] += lr * (z - self.centers[cluster_id])

        if not self._initialized and self.counts.min() > 1:
            self._initialized = True

        return cluster_id

    def reset(self):
        """Reset cluster assignments (fix for curiosity collapse)."""
        self.centers = np.random.randn(self.n_clusters, self.dim).astype(np.float32) * 0.1
        self.counts = np.ones(self.n_clusters, dtype=np.float32)
        self._initialized = False


class CuriosityEngine:
    """
    Implements Oudeyer's Learning Progress signal.

    Tracks prediction error history per latent-space cluster.
    Learning progress = rate of error decrease in a sliding window.
    """

    def __init__(self, n_clusters=None, window_size=None):
        self.n_clusters = n_clusters or cfg.NUM_CURIOSITY_CLUSTERS
        self.window_size = window_size or cfg.CURIOSITY_WINDOW_SIZE

        # Online K-means for state clustering
        self.kmeans = OnlineKMeans(n_clusters=self.n_clusters)

        # Per-cluster sliding window of prediction errors
        self._error_history = defaultdict(lambda: [])

        # Current curiosity weight (can be annealed)
        self.curiosity_weight = cfg.CURIOSITY_WEIGHT_INITIAL

        # Track which clusters have been visited
        self._visit_counts = np.zeros(self.n_clusters, dtype=np.int32)

    def update(self, z, prediction_error):
        """
        Update curiosity state with a new (latent, error) observation.

        Args:
            z: (64,) latent vector
            prediction_error: float, current prediction error

        Returns:
            cluster_id: which cluster this observation belongs to
        """
        z = np.asarray(z, dtype=np.float32).flatten()

        # Assign and update cluster
        cluster_id = self.kmeans.update(z)

        # Append error to this cluster's history
        history = self._error_history[cluster_id]
        history.append(float(prediction_error))

        # Keep only last window_size entries
        if len(history) > self.window_size:
            self._error_history[cluster_id] = history[-self.window_size:]

        self._visit_counts[cluster_id] += 1

        return cluster_id

    def get_learning_progress(self, cluster_id):
        """
        Compute learning progress for a specific cluster.

        Learning progress = max(0, mean(older errors) - mean(recent errors))
        Positive when prediction is IMPROVING (errors decreasing).

        Returns:
            progress: float ≥ 0
        """
        history = self._error_history.get(cluster_id, [])

        if len(history) < cfg.CURIOSITY_OLDER_END:
            return 0.0

        recent = history[-cfg.CURIOSITY_RECENT_WINDOW:]
        older = history[-cfg.CURIOSITY_OLDER_END:-cfg.CURIOSITY_OLDER_START]

        if not recent or not older:
            return 0.0

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)

        # Progress is positive when errors are DECREASING
        progress = max(0.0, older_mean - recent_mean)
        return float(progress)

    def get_intrinsic_reward(self, z):
        """
        Compute intrinsic reward for visiting latent state z.

        Reward = learning_progress(cluster(z)) * curiosity_weight

        Args:
            z: (64,) latent vector

        Returns:
            reward: float ≥ 0
        """
        z = np.asarray(z, dtype=np.float32).flatten()
        cluster_id = self.kmeans.assign(z)
        progress = self.get_learning_progress(cluster_id)
        return progress * self.curiosity_weight

    def get_exploration_target(self):
        """
        Find the cluster with the highest learning progress.
        This is where the agent should explore next.

        Returns:
            target_cluster: int, cluster with most active learning
            max_progress: float, the learning progress value
        """
        max_progress = -1.0
        target_cluster = 0

        for cid in range(self.n_clusters):
            progress = self.get_learning_progress(cid)
            if progress > max_progress:
                max_progress = progress
                target_cluster = cid

        return target_cluster, max_progress

    def get_all_progress(self):
        """Get learning progress for all clusters."""
        return {
            cid: self.get_learning_progress(cid)
            for cid in range(self.n_clusters)
        }

    def count_active_clusters(self, min_progress=0.001):
        """Count clusters with positive learning progress."""
        count = 0
        for cid in range(self.n_clusters):
            if self.get_learning_progress(cid) > min_progress:
                count += 1
        return count

    def get_salience(self):
        """
        Compute overall curiosity salience for Global Workspace competition.
        Returns the max learning progress across all clusters.
        """
        _, max_progress = self.get_exploration_target()
        return max_progress

    def reset_clusters(self):
        """
        Reset cluster assignments. Use when curiosity collapses
        (learning progress is zero everywhere).
        """
        self.kmeans.reset()
        self._error_history.clear()
        self._visit_counts = np.zeros(self.n_clusters, dtype=np.int32)

    def set_curiosity_weight(self, weight):
        """Update the curiosity weight (for stage-based annealing)."""
        self.curiosity_weight = weight

    def get_stats(self):
        """Summary statistics for monitoring."""
        all_progress = self.get_all_progress()
        active = self.count_active_clusters()
        return {
            "active_clusters": active,
            "max_progress": max(all_progress.values()) if all_progress else 0.0,
            "mean_progress": np.mean(list(all_progress.values())) if all_progress else 0.0,
            "total_visits": int(self._visit_counts.sum()),
            "curiosity_weight": self.curiosity_weight,
            "visit_distribution": self._visit_counts.tolist(),
        }
