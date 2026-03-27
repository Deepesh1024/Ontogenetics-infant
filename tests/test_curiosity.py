"""
Test Curiosity Engine

Validates:
1. Decreasing errors produce positive learning progress
2. Constant errors produce near-zero progress
3. K-means assigns to 16 clusters
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognition.curiosity import CuriosityEngine, OnlineKMeans


class TestOnlineKMeans:
    """Test the online clustering."""

    def test_cluster_count(self):
        kmeans = OnlineKMeans(n_clusters=16, dim=64)
        assert kmeans.centers.shape == (16, 64)

    def test_assignment(self):
        kmeans = OnlineKMeans(n_clusters=4, dim=8)
        z = np.random.randn(8).astype(np.float32)
        cluster_id = kmeans.assign(z)
        assert 0 <= cluster_id < 4

    def test_update_moves_center(self):
        kmeans = OnlineKMeans(n_clusters=4, dim=8)
        center_before = kmeans.centers[0].copy()
        # Feed many points near a specific location
        for _ in range(100):
            z = np.ones(8, dtype=np.float32) * 5.0 + np.random.randn(8).astype(np.float32) * 0.01
            kmeans.update(z)
        # At least one center should have moved toward (5, 5, ...)
        max_center = kmeans.centers.max(axis=1)
        assert np.any(max_center > 3.0), "No center moved toward the data"

    def test_reset(self):
        kmeans = OnlineKMeans(n_clusters=4, dim=8)
        for _ in range(50):
            kmeans.update(np.random.randn(8).astype(np.float32))
        kmeans.reset()
        assert kmeans.counts.max() == 1.0
        assert not kmeans._initialized


class TestCuriosityEngine:
    """Test learning progress computation."""

    def test_decreasing_errors_positive_progress(self):
        """When prediction errors are decreasing, learning progress should be positive."""
        engine = CuriosityEngine(n_clusters=4, window_size=50)

        z = np.zeros(64, dtype=np.float32)

        # Feed decreasing errors
        for i in range(50):
            error = 1.0 - (i / 50.0) * 0.8  # decreasing from 1.0 to 0.2
            engine.update(z, error)

        cluster_id = engine.kmeans.assign(z)
        progress = engine.get_learning_progress(cluster_id)

        print(f"  Decreasing errors → progress: {progress:.4f}")
        assert progress > 0, f"Expected positive progress, got {progress}"

    def test_constant_errors_zero_progress(self):
        """When errors are constant, learning progress should be near zero."""
        engine = CuriosityEngine(n_clusters=4, window_size=50)

        z = np.zeros(64, dtype=np.float32)

        # Feed constant errors
        for i in range(50):
            engine.update(z, 0.5)  # constant error

        cluster_id = engine.kmeans.assign(z)
        progress = engine.get_learning_progress(cluster_id)

        print(f"  Constant errors → progress: {progress:.4f}")
        assert progress < 0.01, f"Expected near-zero progress, got {progress}"

    def test_increasing_errors_zero_progress(self):
        """When errors are increasing, progress should be zero (clamped)."""
        engine = CuriosityEngine(n_clusters=4, window_size=50)

        z = np.zeros(64, dtype=np.float32)

        for i in range(50):
            error = 0.2 + (i / 50.0) * 0.8  # increasing
            engine.update(z, error)

        cluster_id = engine.kmeans.assign(z)
        progress = engine.get_learning_progress(cluster_id)

        print(f"  Increasing errors → progress: {progress:.4f}")
        assert progress == 0.0, f"Expected zero progress, got {progress}"

    def test_intrinsic_reward(self):
        engine = CuriosityEngine(n_clusters=4, window_size=50)
        z = np.zeros(64, dtype=np.float32)

        for i in range(50):
            engine.update(z, 1.0 - i / 50.0)

        reward = engine.get_intrinsic_reward(z)
        assert reward >= 0, f"Intrinsic reward negative: {reward}"
        print(f"  Intrinsic reward: {reward:.4f}")

    def test_exploration_target(self):
        engine = CuriosityEngine(n_clusters=4, window_size=50)

        # Pre-seed K-means centers to ensure distinct clusters
        for cid in range(4):
            engine.kmeans.centers[cid] = np.ones(64, dtype=np.float32) * cid * 100

        # Feed different progress to different clusters
        for cid in range(4):
            z = np.ones(64, dtype=np.float32) * cid * 100
            for i in range(50):
                # Cluster 2 gets the most improvement
                if cid == 2:
                    error = 1.0 - i / 50.0
                else:
                    error = 0.5
                engine.update(z, error)

        target, progress = engine.get_exploration_target()
        print(f"  Exploration target: cluster {target}, progress {progress:.4f}")
        assert progress > 0

    def test_active_clusters_count(self):
        engine = CuriosityEngine(n_clusters=16, window_size=50)

        # Create activity in 3 clusters
        for cid in range(3):
            z = np.ones(64, dtype=np.float32) * cid * 20
            for i in range(50):
                engine.update(z, 1.0 - i / 50.0 * 0.5)

        active = engine.count_active_clusters()
        print(f"  Active clusters: {active}")
        assert active >= 1  # at least some should be active
