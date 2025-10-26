import torch
import mujoco
import numpy as np
import yaml
from huggingface_hub import hf_hub_download

class LowerT1JoyStick:
    """
    Utility class for controlling the lower-level gait generation of a T1 robot
    using joystick commands (X velocity, Y velocity, yaw velocity) as input.

    This class provides:
    - State initialization for joint positions, stiffness, and damping
    - Observation vector construction for the control policy
    - Action generation using a low-level control policy model
    - Gait cycle progression tracking

    Attributes:
        nu (int): Number of actuated degrees of freedom (DoF) from the robot's observation space.
        diff (int): Offset between Mujoco actuator index and observed DoFs.
        qpos_shape (int): Expected qpos size for the robot's base pose + actuated DoFs.
        cfg (dict): Configuration dictionary containing control gains, normalization, and initial states.
        default_dof_pos (np.ndarray): Default joint positions.
        dof_stiffness (np.ndarray): Per-joint stiffness values.
        dof_damping (np.ndarray): Per-joint damping values.
        actions (np.ndarray): Latest actions from the policy model.
        dof_targets (np.ndarray): Target joint positions from the policy.
        gait_frequency (float): Current gait frequency.
        gait_process (float): Current gait phase (0 to 1).
        it (int): Iteration counter.
    """

    def __init__(self, env):
        """
        Initializes the joystick utility with the robot's environment and configuration.

        Args:
            env: Mujoco simulation environment containing the robot.
            cfg (dict): Configuration dictionary with:
                - init_state: Default positions, rotations, and joint angles.
                - control: Stiffness, damping, action scaling, and decimation settings.
                - env: Environment-specific parameters like number of actions and observations.
                - normalization: Scaling factors for observation normalization.
        """

        self.env = env
        self.model, self.cfg = self.load()

        self.get_robot_properties(env)
        self.reset(env)

    @staticmethod
    def quat_rotate_inverse(q: np.ndarray, v: np.ndarray):
        """
        Rotates a vector `v` by the inverse of quaternion `q`.

        This operation effectively transforms the vector from the rotated
        frame back into the original frame.

        Args:
            q (np.ndarray): Quaternion in the format [x, y, z, w].
            v (np.ndarray): 3D vector to be rotated.

        Returns:
            np.ndarray: The rotated vector in the original frame.
        """
        
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v) * 2.0)
        return a - b + c

    def load(self):

        cfg_file = hf_hub_download(
                    repo_id="SaiResearch/booster_soccer_models",
                    filename="config/lower_t1.yaml",
                    repo_type="model")
        
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        model_file = hf_hub_download(
                    repo_id="SaiResearch/booster_soccer_models",
                    filename="robot/lower_t1_control.pt",
                    repo_type="model")
        
        model = torch.jit.load(model_file)
        model.eval()

        return model, cfg
    
    def get_robot_properties(self, env):

        _, mj_model = self.get_env_data_model(env)
        self.nu = len(env.robots[0].get_obs()[0])
        self.diff = mj_model.nu - self.nu
        self.qpos_shape = 7 + self.nu
    
    def get_env_data_model(self, env):

        mj_model = env.sim.model._model
        mj_data = env.sim.data._data

        return mj_data, mj_model

    def reset(self, env):
        """
        Resets the controller's internal state and initializes Mujoco's joint positions
        and control parameters based on the configuration.

        Args:
            mj_model: Mujoco model object.
            mj_data: Mujoco data object.
        """

        mj_data, mj_model = self.get_env_data_model(env)
        self.default_dof_pos = np.zeros(self.nu, dtype=np.float32)
        self.dof_stiffness = np.zeros(self.nu, dtype=np.float32)
        self.dof_damping = np.zeros(self.nu, dtype=np.float32)
        for i in range(self.diff, mj_model.nu):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                    self.default_dof_pos[i-self.diff] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[i-self.diff] = self.cfg["init_state"]["default_joint_angles"]["default"]

            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                    self.dof_stiffness[i-self.diff] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[i-self.diff] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} were not defined")
        
        mj_data.qpos[:self.qpos_shape] = np.concatenate(
            [
                np.array(self.cfg["init_state"]["pos"], dtype=np.float32),
                np.array(self.cfg["init_state"]["rot"][3:4] + self.cfg["init_state"]["rot"][0:3], dtype=np.float32),
                self.default_dof_pos,
            ]
        )

        mujoco.mj_forward(mj_model, mj_data)
        self.actions = np.zeros((self.cfg["env"]["num_actions"]), dtype=np.float32)
        self.dof_targets = np.zeros(self.default_dof_pos.shape, dtype=np.float32)
        self.gait_frequency = self.gait_process = 0.0
        self.it = 0

    def get_obs(self, command, obs, info):
        """
        Constructs the observation vector for the control policy.

        Args:
            command (tuple/list): Desired (lin_vel_x, lin_vel_y, ang_vel_yaw).
            obs (np.array): Current observation from the environment
            info (dict): Information dict from the environment

        Returns:
            np.ndarray: Normalized observation vector containing:
                - Projected gravity
                - Base angular velocity
                - Commanded velocities
                - Gait phase signals (cos/sin)
                - Joint position/velocity differences
                - Previous actions
        """

        lin_vel_x, lin_vel_y, ang_vel_yaw = map(float, command)

        if lin_vel_x == 0 and lin_vel_y == 0 and ang_vel_yaw == 0:
            self.gait_frequency = 0
        else:
            self.gait_frequency = np.average(self.cfg["commands"]["gait_frequency"])

        dof_pos = obs[:12]
        dof_vel = obs[12:24]
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        projected_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        
        obs = np.zeros(self.cfg["env"]["num_observations"], dtype=np.float32)
        obs[0:3] = projected_gravity * self.cfg["normalization"]["gravity"]
        obs[3:6] = base_ang_vel * self.cfg["normalization"]["ang_vel"]
        obs[6] = lin_vel_x * self.cfg["normalization"]["lin_vel"]
        obs[7] = lin_vel_y * self.cfg["normalization"]["lin_vel"]
        obs[8] = ang_vel_yaw * self.cfg["normalization"]["ang_vel"]
        obs[9] = np.cos(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        obs[10] = np.sin(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        obs[11:23] = (dof_pos - self.default_dof_pos) * self.cfg["normalization"]["dof_pos"]
        obs[23:35] = dof_vel * self.cfg["normalization"]["dof_vel"]
        obs[35:47] = self.actions

        return obs
    
    def get_actions(self, command, observation, info):
        """
        Generates joint control signals based on the current observation
        and the policy model.

        Args:
            command (tuple/list): Desired (lin_vel_x, lin_vel_y, ang_vel_yaw).  
            obs (np.array): Current observation from the environment
            info (dict): Information dict from the environment

        Returns:
            np.ndarray: Control signals for the actuators.
        """
        
        _, mj_model = self.get_env_data_model(self.env)
        obs = self.get_obs(command, observation, info)

        dof_pos = observation[:12]
        dof_vel = observation[12:24]

        if self.it % self.cfg["control"]["decimation"] == 0:
            dist = self.model(torch.tensor(obs.reshape(1,-1)))
            self.actions[:] = dist.detach().numpy()
            self.actions[:] = np.clip(self.actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        
        self.dof_targets[:] = self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions
        ctrl = np.clip(
            self.dof_stiffness * (self.dof_targets - dof_pos) - self.dof_damping * dof_vel,
            mj_model.actuator_ctrlrange[self.diff:, 0],
            mj_model.actuator_ctrlrange[self.diff:, 1],
        )

        self.it += 1
        self.gait_process = np.fmod(self.gait_process + self.cfg["sim"]["dt"] * self.gait_frequency, 1.0)

        return ctrl, self.actions.copy()
    
    def get_torque(self, observation, actions):
        """
        Generates joint control signals based on the current observation
        and the policy model.

        Args:
            command (tuple/list): Desired (lin_vel_x, lin_vel_y, ang_vel_yaw).  
            obs (np.array): Current observation from the environment
            info (dict): Information dict from the environment

        Returns:
            np.ndarray: Control signals for the actuators.
        """
        
        _, mj_model = self.get_env_data_model(self.env)
        dof_pos = observation[:12]
        dof_vel = observation[12:24]

        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * actions
        
        ctrl = np.clip(
            self.dof_stiffness * (dof_targets - dof_pos) - self.dof_damping * dof_vel,
            mj_model.actuator_ctrlrange[self.diff:, 0],
            mj_model.actuator_ctrlrange[self.diff:, 1],
        )

        return ctrl