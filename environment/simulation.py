"""
OIS Simulation Environment
PyBullet world with ground plane, camera, parent + infant humanoids.
This is the Markov blanket made concrete.
"""

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    p = None
    pybullet_data = None

from environment.humanoid import Humanoid
import config as cfg


class OISEnvironment:
    """
    The simulation world containing two humanoid agents (parent + infant)
    on a flat ground plane with gravity.
    """

    def __init__(self, render=False):
        """
        Args:
            render: If True, use GUI mode for visual debugging.
        """
        self.render_mode = render

        # Connect to PyBullet
        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        # Configure physics
        p.setGravity(0, 0, cfg.GRAVITY, physicsClientId=self.client)
        p.setTimeStep(cfg.TIMESTEP, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            numSolverIterations=10,
            physicsClientId=self.client,
        )

        # Load ground plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_id = p.loadURDF(
            "plane.urdf", physicsClientId=self.client
        )
        p.changeDynamics(
            self.ground_id, -1,
            lateralFriction=cfg.GROUND_FRICTION,
            physicsClientId=self.client,
        )

        # Load parent humanoid (offset to the side so infant can see it)
        parent_start_pos = [2.0, 0, 0.9]
        parent_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.parent_body_id = p.loadURDF(
            "humanoid/humanoid.urdf",
            parent_start_pos, parent_start_orn,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.client,
        )

        # Load infant humanoid
        infant_start_pos = [0, 0, 0.9]
        infant_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.infant_body_id = p.loadURDF(
            "humanoid/humanoid.urdf",
            infant_start_pos, infant_start_orn,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.client,
        )

        # Create Humanoid wrappers
        self.parent = Humanoid(self.client, self.parent_body_id)
        self.infant = Humanoid(self.client, self.infant_body_id)

        # Disable default motors for torque control
        self.parent.disable_default_motors()
        self.infant.disable_default_motors()

        # Camera parameters for 64x64 rendering
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.5],
            distance=3.0,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.client,
        )
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=10.0,
            physicsClientId=self.client,
        )

        self.timestep = 0

    def step(self, infant_action, parent_action=None):
        """
        Step simulation forward by one timestep.

        Args:
            infant_action: 17-d torque vector for infant (range [-1, 1])
            parent_action: 17-d torque vector for parent (if None, parent is passive)

        Returns:
            observation dict from get_observation()
        """
        # Apply parent action
        if parent_action is not None:
            self.parent.apply_torques(parent_action)

        # Apply infant action
        self.infant.apply_torques(infant_action)

        # Step physics
        p.stepSimulation(physicsClientId=self.client)
        self.timestep += 1

        return self.get_observation()

    def get_observation(self):
        """
        Returns the infant's full observation dictionary.

        Keys:
            vision: (64, 64, 3) uint8 RGB image
            vision_flat: (12288,) float32 normalized [0,1]
            proprio: (2*N,) float32 proprioceptive vector
            touch: (6,) float32 contact forces
            parent_obs: (2*N + N,) float32 parent joint state + parent action placeholder
        """
        # Vision: render 64x64 camera
        vision = self.render_camera()

        # Proprioception: infant joint angles + velocities
        proprio = self.infant.get_proprioception()

        # Touch: contact forces on infant
        touch = self.infant.get_contact_forces()

        # Parent observation: parent's joint state
        parent_proprio = self.parent.get_proprioception()

        return {
            "vision": vision,
            "vision_flat": vision.flatten().astype(np.float32) / 255.0,
            "proprio": proprio,
            "touch": touch,
            "parent_obs": parent_proprio,
            "infant_pos": self.infant.get_position(),
            "infant_height": self.infant.get_height(),
            "parent_pos": self.parent.get_position(),
            "parent_height": self.parent.get_height(),
            "timestep": self.timestep,
        }

    def render_camera(self):
        """Render 64x64x3 RGB image from the fixed camera."""
        _, _, rgb, _, _ = p.getCameraImage(
            width=cfg.CAMERA_WIDTH,
            height=cfg.CAMERA_HEIGHT,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        # rgb is (H, W, 4) RGBA — take only RGB
        rgb = np.array(rgb, dtype=np.uint8).reshape(
            cfg.CAMERA_HEIGHT, cfg.CAMERA_WIDTH, 4
        )[:, :, :3]
        return rgb

    def reset(self):
        """Reset both humanoids to initial pose and clear timestep."""
        # Reset parent
        p.resetBasePositionAndOrientation(
            self.parent_body_id,
            [2.0, 0, 0.9],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(
            self.parent_body_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self.client,
        )
        self.parent.reset_pose()
        self.parent.disable_default_motors()

        # Reset infant
        p.resetBasePositionAndOrientation(
            self.infant_body_id,
            [0, 0, 0.9],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(
            self.infant_body_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self.client,
        )
        self.infant.reset_pose()
        self.infant.disable_default_motors()

        self.timestep = 0
        return self.get_observation()

    def get_parent_demonstration(self, parent_action):
        """
        Get the parent's current state as a demonstration for the infant.

        Args:
            parent_action: the action taken by the parent this step

        Returns:
            dict with parent joint state and action
        """
        parent_angles, parent_vels = self.parent.get_joint_states()
        return {
            "joint_angles": parent_angles,
            "joint_velocities": parent_vels,
            "action": np.asarray(parent_action, dtype=np.float32),
        }

    def compute_locomotion_efficiency(self):
        """
        Compute infant's displacement per timestep relative to parent.
        Returns ratio: infant_speed / parent_speed.
        """
        infant_vel, _ = self.infant.get_velocity()
        parent_vel, _ = self.parent.get_velocity()

        infant_speed = np.linalg.norm(infant_vel[:2])  # horizontal speed
        parent_speed = np.linalg.norm(parent_vel[:2])

        if parent_speed < 1e-6:
            return 0.0
        return float(infant_speed / parent_speed)

    def run_parent_demo(self, num_steps, parent_agent=None):
        """
        Run parent agent for num_steps to validate CPG walking.

        Args:
            num_steps: number of simulation steps
            parent_agent: ParentAgent instance (if None, just steps with zero torque)

        Returns:
            dict with validation metrics
        """
        self.reset()
        heights = []
        positions = []

        for t in range(num_steps):
            if parent_agent is not None:
                action = parent_agent.get_action(t)
                self.parent.apply_torques(action)

            # Zero torque for infant (not relevant here)
            self.infant.apply_torques(np.zeros(self.infant.num_actuated))

            p.stepSimulation(physicsClientId=self.client)

            height = self.parent.get_height()
            pos = self.parent.get_position()
            heights.append(height)
            positions.append(pos.copy())

            if self.parent.is_fallen():
                break

        return {
            "steps_survived": len(heights),
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
            "final_position": positions[-1] if positions else np.zeros(3),
            "fell": self.parent.is_fallen() if heights else True,
            "heights": heights,
        }

    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect(physicsClientId=self.client)
