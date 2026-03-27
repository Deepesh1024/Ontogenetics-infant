"""
Parent Agent — Central Pattern Generator (CPG) Walking Policy

A hardcoded expert walking policy. Does NOT learn. Its sole purpose is to
provide stable, repeatable demonstration data for the infant's imitation learning.

The CPG uses coupled sinusoidal oscillators with per-joint PD control and
carefully tuned phase offsets. The parent must walk 500 steps without falling
before any other component is built.
"""

import numpy as np


class ParentAgent:
    """
    CPG-based walking policy producing 17-joint torques.

    Each joint has:
      - An oscillator (amplitude, frequency, phase offset)
      - PD control parameters (Kp, Kd) for tracking the target trajectory
    """

    def __init__(self, num_joints, timestep=1.0 / 60.0):
        """
        Args:
            num_joints: number of actuated joints (should match Humanoid.num_actuated)
            timestep: simulation timestep in seconds
        """
        self.num_joints = num_joints
        self.dt = timestep
        self.frequency = 1.5  # Hz — gait cycle frequency

        # ─── Per-joint oscillator parameters ──────────────────────────
        # Format: (amplitude, phase_offset, Kp, Kd)
        # Phase offsets in radians:
        #   0    = in-phase with right hip reference
        #   π    = anti-phase (left leg)
        #   π/4  = trailing (knee follows hip)

        # Default: small stabilizing oscillation
        self._default_params = (0.0, 0.0, 5.0, 0.5)

        # Joint-specific parameters tuned for stable bipedal gait
        # Index mapping follows Humanoid.ACTUATED_JOINT_NAMES order:
        # 0-2: abdomen (spine), 3-5: right hip, 6: right knee,
        # 7-8: right ankle, 9-11: left hip, 12: left knee,
        # 13-14: left ankle, 15: right shoulder, 16: right elbow

        self.joint_params = {}

        # ── Spine stabilization (indices 0, 1, 2) ──
        # Spine should stay upright with strong PD, minimal oscillation
        self.joint_params[0] = (0.05, 0.0, 20.0, 2.0)    # abdomen_z: yaw stability
        self.joint_params[1] = (0.05, 0.0, 20.0, 2.0)    # abdomen_y: pitch stability
        self.joint_params[2] = (0.02, 0.0, 20.0, 2.0)    # abdomen_x: roll stability

        # ── Right hip (indices 3, 4, 5) ──
        self.joint_params[3] = (0.1, 0.0, 8.0, 1.0)      # right_hip_x: abduction
        self.joint_params[4] = (0.05, 0.0, 8.0, 1.0)     # right_hip_z: rotation
        self.joint_params[5] = (0.6, 0.0, 12.0, 1.5)     # right_hip_y: flexion (main driver)

        # ── Right knee (index 6) ──
        self.joint_params[6] = (0.5, -np.pi / 4, 10.0, 1.2)  # trailing phase, strong PD

        # ── Right ankle (indices 7, 8) ──
        self.joint_params[7] = (0.1, 0.0, 6.0, 0.8)      # right_ankle_x: roll
        self.joint_params[8] = (0.25, np.pi / 6, 8.0, 1.0)  # right_ankle_y: push-off

        # ── Left hip (indices 9, 10, 11) — anti-phase to right ──
        self.joint_params[9] = (0.1, np.pi, 8.0, 1.0)    # left_hip_x
        self.joint_params[10] = (0.05, np.pi, 8.0, 1.0)   # left_hip_z
        self.joint_params[11] = (0.6, np.pi, 12.0, 1.5)   # left_hip_y (main driver, anti-phase)

        # ── Left knee (index 12) ──
        self.joint_params[12] = (0.5, np.pi - np.pi / 4, 10.0, 1.2)  # anti-phase + trailing

        # ── Left ankle (indices 13, 14) ──
        self.joint_params[13] = (0.1, np.pi, 6.0, 0.8)   # left_ankle_x
        self.joint_params[14] = (0.25, np.pi + np.pi / 6, 8.0, 1.0)  # left_ankle_y

        # ── Arms — counter-phase to legs for balance ──
        self.joint_params[15] = (0.25, np.pi, 5.0, 0.5)   # right_shoulder_y
        self.joint_params[16] = (0.15, np.pi / 2, 4.0, 0.4)  # right_elbow

        # Target joint angles (rest position = slightly bent knees for stability)
        self.rest_angles = np.zeros(self.num_joints, dtype=np.float32)
        if self.num_joints > 6:
            self.rest_angles[6] = -0.1    # right knee slightly bent
        if self.num_joints > 12:
            self.rest_angles[12] = -0.1   # left knee slightly bent

        # Previous joint positions for velocity estimation (PD derivative term)
        self._prev_positions = None

    def get_action(self, timestep, current_joint_positions=None, current_joint_velocities=None):
        """
        Compute CPG torques for a given timestep.

        Args:
            timestep: integer timestep (multiplied by dt for phase)
            current_joint_positions: current joint angles (if available, for PD control)
            current_joint_velocities: current joint velocities (if available)

        Returns:
            numpy array of shape (num_joints,) with torques in [-1, 1]
        """
        t = timestep * self.dt
        omega = 2.0 * np.pi * self.frequency  # angular frequency

        torques = np.zeros(self.num_joints, dtype=np.float32)

        for j in range(self.num_joints):
            amp, phase, kp, kd = self.joint_params.get(j, self._default_params)

            # Target position from oscillator
            target = self.rest_angles[j] + amp * np.sin(omega * t + phase)

            if current_joint_positions is not None and current_joint_velocities is not None:
                # PD control: torque = Kp * (target - current) - Kd * velocity
                pos_error = target - current_joint_positions[j]
                vel = current_joint_velocities[j]
                torque = kp * pos_error - kd * vel
            else:
                # Open-loop: just output the oscillator signal directly
                torque = kp * amp * np.sin(omega * t + phase)

            torques[j] = torque

        # Normalize to [-1, 1] range
        max_torque = np.max(np.abs(torques)) + 1e-8
        if max_torque > 1.0:
            torques = torques / max_torque

        torques = np.clip(torques, -1.0, 1.0)

        return torques

    def get_state(self, humanoid):
        """
        Get the parent's full state for infant observation.

        Args:
            humanoid: Humanoid wrapper for the parent body

        Returns:
            dict with joint_angles (N,), joint_velocities (N,), and last_action (N,)
        """
        angles, velocities = humanoid.get_joint_states()
        return {
            "joint_angles": angles,
            "joint_velocities": velocities,
            "proprioception": np.concatenate([angles, velocities]),
        }


def validate_cpg(render=False, num_steps=500):
    """
    Standalone validation: parent must walk num_steps without falling.
    This is the HARD GATE for Phase 1.
    """
    from environment.simulation import OISEnvironment

    env = OISEnvironment(render=render)
    parent = ParentAgent(
        num_joints=env.parent.num_actuated,
        timestep=cfg.TIMESTEP,
    )

    env.reset()
    heights = []
    positions = []
    fell = False

    for t in range(num_steps):
        # Get current joint state for PD control
        angles, velocities = env.parent.get_joint_states()
        action = parent.get_action(t, angles, velocities)
        env.parent.apply_torques(action)

        # Also need to apply zero torques to infant
        env.infant.apply_torques(np.zeros(env.infant.num_actuated))

        import pybullet as p
        p.stepSimulation(physicsClientId=env.client)

        height = env.parent.get_height()
        pos = env.parent.get_position()
        heights.append(height)
        positions.append(pos.copy())

        if env.parent.is_fallen():
            fell = True
            print(f"  ✗ Parent FELL at step {t}, height = {height:.3f}")
            break

    min_h = min(heights)
    displacement = np.linalg.norm(positions[-1][:2] - positions[0][:2])

    print(f"\n{'='*50}")
    print(f"  CPG Validation Report")
    print(f"{'='*50}")
    print(f"  Steps survived:   {len(heights)} / {num_steps}")
    print(f"  Min COM height:   {min_h:.3f} m")
    print(f"  Final height:     {heights[-1]:.3f} m")
    print(f"  Displacement:     {displacement:.3f} m")
    print(f"  Fell:             {'YES ✗' if fell else 'NO ✓'}")
    print(f"  GATE PASSED:      {'YES ✓' if not fell and len(heights) >= num_steps else 'NO ✗'}")
    print(f"{'='*50}\n")

    env.close()
    return not fell and len(heights) >= num_steps


if __name__ == "__main__":
    import config as cfg
    import sys

    render_flag = "--render" in sys.argv
    passed = validate_cpg(render=render_flag, num_steps=500)
    sys.exit(0 if passed else 1)
