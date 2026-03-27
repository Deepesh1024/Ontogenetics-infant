"""
Humanoid wrapper for PyBullet's built-in humanoid URDF.

The humanoid.urdf uses:
  - 6 SPHERICAL joints (chest, neck, R/L shoulder, R/L hip, R/L ankle) → 3 controllable DOFs each
  - 4 REVOLUTE joints (R/L elbow, R/L knee) → 1 DOF each
  
We convert spherical joints to 3-DOF angular control via setJointMotorControlMultiDof.
Total controllable DOFs = 6×3 + 4×1 = 22 angular outputs.

For simplicity and stability, we use a unified interface:
  - All actions are in [-1, 1]
  - Internal mapping handles spherical vs revolute joint types
"""

import numpy as np

try:
    import pybullet as p
except ImportError:
    p = None


# Joint type constants from pybullet
JOINT_REVOLUTE = 0
JOINT_SPHERICAL = 2
JOINT_FIXED = 4


class Humanoid:
    """Wrapper around a PyBullet humanoid body providing joint & sensor access."""

    # Contact link names for touch sensing
    CONTACT_LINKS = [
        "right_foot", "left_foot",
        "right_lower_arm", "left_lower_arm",
        "head", "torso",
    ]

    def __init__(self, physics_client, body_id):
        """
        Args:
            physics_client: PyBullet physics client ID
            body_id: PyBullet body unique ID
        """
        self.client = physics_client
        self.body_id = body_id

        # Discover all joints from the loaded URDF
        self.num_total_joints = p.getNumJoints(body_id, physicsClientId=self.client)
        self._joint_name_to_idx = {}
        self._link_name_to_idx = {}
        self._joint_types = {}

        for i in range(self.num_total_joints):
            info = p.getJointInfo(body_id, i, physicsClientId=self.client)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            joint_type = info[2]
            self._joint_name_to_idx[joint_name] = i
            self._link_name_to_idx[link_name] = i
            self._joint_types[i] = joint_type

        # Separate joints by type (skip FIXED joints)
        self._spherical_indices = []  # joint indices for spherical joints
        self._revolute_indices = []   # joint indices for revolute joints
        self._spherical_names = []
        self._revolute_names = []

        for i in range(self.num_total_joints):
            info = p.getJointInfo(body_id, i, physicsClientId=self.client)
            name = info[1].decode("utf-8")
            jtype = info[2]
            if jtype == JOINT_SPHERICAL:
                self._spherical_indices.append(i)
                self._spherical_names.append(name)
            elif jtype == JOINT_REVOLUTE:
                self._revolute_indices.append(i)
                self._revolute_names.append(name)

        # Total DOFs: spherical = 3 angular DOFs each, revolute = 1 DOF each
        self._num_spherical_dofs = len(self._spherical_indices) * 3
        self._num_revolute_dofs = len(self._revolute_indices)
        self.num_actuated = self._num_spherical_dofs + self._num_revolute_dofs

        # All controllable joint indices (spherical first, then revolute)
        self._all_controllable = self._spherical_indices + self._revolute_indices

        # Joint limits for revolute joints (for normalization)
        self._rev_lower = np.zeros(len(self._revolute_indices))
        self._rev_upper = np.zeros(len(self._revolute_indices))
        self._rev_max_vel = np.zeros(len(self._revolute_indices))
        self._rev_max_force = np.zeros(len(self._revolute_indices))
        for i, idx in enumerate(self._revolute_indices):
            info = p.getJointInfo(body_id, idx, physicsClientId=self.client)
            self._rev_lower[i] = info[8] if info[8] < info[9] else -np.pi
            self._rev_upper[i] = info[9] if info[8] < info[9] else np.pi
            self._rev_max_vel[i] = info[11] if info[11] > 0 else 10.0
            self._rev_max_force[i] = info[10] if info[10] > 0 else 100.0
        self._rev_range = self._rev_upper - self._rev_lower
        self._rev_range[self._rev_range < 1e-6] = 1.0

        # Spherical joint limits (use π for normalization)
        self._sph_max_angle = np.pi
        self._sph_max_vel = 10.0
        self._sph_max_force = 200.0

        # Contact link indices
        self.contact_link_indices = []
        for name in self.CONTACT_LINKS:
            if name in self._link_name_to_idx:
                self.contact_link_indices.append(self._link_name_to_idx[name])
            else:
                self.contact_link_indices.append(-1)

    def get_joint_states(self):
        """
        Returns (angles, velocities) each as numpy arrays of shape (num_actuated,).
        All normalized to [-1, 1].
        
        Layout: [spherical_0_xyz, spherical_1_xyz, ..., revolute_0, revolute_1, ...]
        """
        angles = np.zeros(self.num_actuated, dtype=np.float32)
        velocities = np.zeros(self.num_actuated, dtype=np.float32)
        idx = 0

        # Spherical joints: state is quaternion [x,y,z,w], velocity is [wx,wy,wz]
        for jidx in self._spherical_indices:
            state = p.getJointStateMultiDof(self.body_id, jidx, physicsClientId=self.client)
            # state[0] = position (quaternion 4-d), state[1] = velocity (3-d)
            quat = state[0]  # (x, y, z, w)
            vel = state[1]   # (wx, wy, wz)

            # Convert quaternion to euler angles for normalization
            euler = p.getEulerFromQuaternion(quat)
            for k in range(3):
                angles[idx + k] = np.clip(euler[k] / self._sph_max_angle, -1.0, 1.0)
                velocities[idx + k] = np.clip(vel[k] / self._sph_max_vel, -1.0, 1.0)
            idx += 3

        # Revolute joints: standard 1-DOF
        for i, jidx in enumerate(self._revolute_indices):
            state = p.getJointState(self.body_id, jidx, physicsClientId=self.client)
            angle = state[0]
            vel = state[1]
            angles[idx] = np.clip(
                2.0 * (angle - self._rev_lower[i]) / self._rev_range[i] - 1.0, -1.0, 1.0
            )
            velocities[idx] = np.clip(vel / (self._rev_max_vel[i] + 1e-8), -1.0, 1.0)
            idx += 1

        return angles, velocities

    def get_proprioception(self):
        """Returns 2*num_actuated dimensional proprioceptive vector (angles + velocities)."""
        angles, velocities = self.get_joint_states()
        return np.concatenate([angles, velocities])

    def apply_torques(self, torques):
        """
        Apply joint torques. Input: array of shape (num_actuated,) in range [-1, 1].
        Layout matches get_joint_states: [sph_0_xyz, sph_1_xyz, ..., rev_0, rev_1, ...]
        """
        torques = np.clip(np.asarray(torques, dtype=np.float32), -1.0, 1.0)
        idx = 0

        # Spherical joints: use setJointMotorControlMultiDof with TORQUE_CONTROL
        for jidx in self._spherical_indices:
            t = torques[idx:idx+3] * self._sph_max_force  # scale to physical torque
            p.setJointMotorControlMultiDof(
                self.body_id, jidx,
                controlMode=p.TORQUE_CONTROL,
                force=[float(t[0]), float(t[1]), float(t[2])],
                physicsClientId=self.client,
            )
            idx += 3

        # Revolute joints: standard torque control
        for i, jidx in enumerate(self._revolute_indices):
            force = torques[idx] * self._rev_max_force[i]
            p.setJointMotorControl2(
                self.body_id, jidx,
                controlMode=p.TORQUE_CONTROL,
                force=float(force),
                physicsClientId=self.client,
            )
            idx += 1

    def get_contact_forces(self):
        """Returns 6-d contact force vector (one per CONTACT_LINKS entry)."""
        forces = np.zeros(len(self.CONTACT_LINKS), dtype=np.float32)
        contact_points = p.getContactPoints(
            bodyA=self.body_id, physicsClientId=self.client
        )
        for cp in contact_points:
            link_idx = cp[3]
            normal_force = cp[9]
            for i, cidx in enumerate(self.contact_link_indices):
                if link_idx == cidx:
                    forces[i] += abs(normal_force)
                    break
        return forces

    def get_position(self):
        """Returns COM position as (x, y, z) numpy array."""
        pos, _ = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client
        )
        return np.array(pos, dtype=np.float32)

    def get_velocity(self):
        """Returns (linear_vel, angular_vel) each as 3-d numpy arrays."""
        lin, ang = p.getBaseVelocity(self.body_id, physicsClientId=self.client)
        return np.array(lin, dtype=np.float32), np.array(ang, dtype=np.float32)

    def get_orientation(self):
        """Returns base orientation as quaternion [x,y,z,w]."""
        _, orn = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client
        )
        return np.array(orn, dtype=np.float32)

    def get_height(self):
        """Returns COM height (z coordinate)."""
        return self.get_position()[2]

    def is_fallen(self, min_height=0.3):
        """Check if the humanoid has fallen (COM below threshold)."""
        return self.get_height() < min_height

    def reset_pose(self):
        """Reset all joints to zero position/velocity."""
        for idx in self._spherical_indices:
            p.resetJointStateMultiDof(
                self.body_id, idx,
                targetValue=[0, 0, 0, 1],  # identity quaternion
                targetVelocity=[0, 0, 0],
                physicsClientId=self.client,
            )
        for idx in self._revolute_indices:
            p.resetJointState(
                self.body_id, idx, targetValue=0.0, targetVelocity=0.0,
                physicsClientId=self.client,
            )

    def disable_default_motors(self):
        """Disable default velocity motors so we can use torque control."""
        for idx in self._spherical_indices:
            p.setJointMotorControlMultiDof(
                self.body_id, idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                force=[0, 0, 0],
                physicsClientId=self.client,
            )
        for idx in self._revolute_indices:
            p.setJointMotorControl2(
                self.body_id, idx,
                controlMode=p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self.client,
            )

    def get_all_joint_info(self):
        """Debug utility: print all joint names and indices."""
        type_names = {0: "REVOLUTE", 1: "PRISMATIC", 2: "SPHERICAL", 3: "PLANAR", 4: "FIXED"}
        info_list = []
        for i in range(self.num_total_joints):
            info = p.getJointInfo(self.body_id, i, physicsClientId=self.client)
            info_list.append({
                "index": i,
                "name": info[1].decode("utf-8"),
                "type": type_names.get(info[2], str(info[2])),
                "link_name": info[12].decode("utf-8"),
                "lower_limit": info[8],
                "upper_limit": info[9],
                "max_force": info[10],
                "max_velocity": info[11],
            })
        return info_list
