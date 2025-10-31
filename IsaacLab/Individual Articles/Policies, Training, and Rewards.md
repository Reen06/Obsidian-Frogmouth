# 🧠 **Complete Guide: Creating Policies, Training Programs, and Reward Systems for Custom Robots**

This is the most comprehensive guide you'll need to implement RL for your custom robot! Let me break down every aspect in detail.

## 🎯 **Part 1: Policy Architecture and Neural Network Design**

### **1.1 Policy Network Architecture**

#### **RSL-RL Policy Configuration:**
```python
# source/your_project/your_project/tasks/agents/rsl_rl_ppo_cfg.py
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class MyRobotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Training Configuration
    num_steps_per_env = 16          # Steps per environment per update
    max_iterations = 1500           # Total training iterations
    save_interval = 50              # Save checkpoint every N iterations
    experiment_name = "my_robot_task"
    
    # Policy Network Architecture
    policy = RslRlPpoActorCriticCfg(
        # Network Architecture
        actor_hidden_dims=[512, 512, 256],    # Actor network layers
        critic_hidden_dims=[512, 512, 256],   # Critic network layers
        activation="elu",                     # Activation function
        
        # Observation Processing
        actor_obs_normalization=True,         # Normalize actor observations
        critic_obs_normalization=True,        # Normalize critic observations
        
        # Action Space Configuration
        init_noise_std=1.0,                   # Initial action noise
        noise_std_type="scalar",              # Noise type: "scalar" or "log"
    )
    
    # Algorithm Hyperparameters
    algorithm = RslRlPpoAlgorithmCfg(
        # Learning Rates and Optimization
        learning_rate=1.0e-3,                 # Learning rate
        schedule="adaptive",                   # LR schedule: "adaptive", "linear", "constant"
        max_grad_norm=1.0,                    # Gradient clipping
        
        # PPO Specific Parameters
        clip_param=0.2,                       # PPO clipping parameter
        value_loss_coef=1.0,                  # Value loss coefficient
        use_clipped_value_loss=True,          # Clip value loss
        entropy_coef=0.01,                    # Entropy bonus coefficient
        
        # Training Configuration
        num_learning_epochs=5,                # Epochs per update
        num_mini_batches=4,                   # Mini-batches per epoch
        
        # Discount and Advantage Estimation
        gamma=0.99,                           # Discount factor
        lam=0.95,                             # GAE lambda
        desired_kl=0.01,                      # Target KL divergence
    )
```

#### **SKRL Policy Configuration:**
```yaml
# source/your_project/your_project/tasks/agents/skrl_ppo_cfg.yaml
seed: 42

# Neural Network Models
models:
  separate: False  # Use shared network for actor-critic
  
  # Policy Network (Actor)
  policy:
    class: GaussianMixin
    clip_actions: False
    clip_log_std: True
    min_log_std: -20.0
    max_log_std: 2.0
    initial_log_std: 0.0
    network:
      - name: net
        input: STATES
        layers: [512, 512, 256, 128]  # Network architecture
        activations: elu               # Activation function
    output: ACTIONS
  
  # Value Network (Critic)
  value:
    class: DeterministicMixin
    clip_actions: False
    network:
      - name: net
        input: STATES
        layers: [512, 512, 256, 128]  # Same architecture as policy
        activations: elu
    output: ONE

# Memory Configuration
memory:
  class: RandomMemory
  memory_size: -1  # Auto-determined

# PPO Agent Configuration
agent:
  class: PPO
  rollouts: 16              # Rollout buffer size
  learning_epochs: 5        # Training epochs per update
  mini_batches: 4           # Mini-batches per epoch
  
  # Learning Parameters
  learning_rate: 1.0e-3
  lr_scheduler: adaptive
  gamma: 0.99
  lam: 0.95
  
  # PPO Parameters
  clip_param: 0.2
  value_loss_coef: 1.0
  entropy_coef: 0.01
  max_grad_norm: 1.0
```

### **1.2 Network Architecture Guidelines**

#### **For Different Robot Types:**

**Simple Robots (2-6 DOF):**
```python
actor_hidden_dims=[128, 128, 64]
critic_hidden_dims=[128, 128, 64]
```

**Medium Complexity (7-12 DOF):**
```python
actor_hidden_dims=[256, 256, 128]
critic_hidden_dims=[256, 256, 128]
```

**Complex Robots (13+ DOF):**
```python
actor_hidden_dims=[512, 512, 256, 128]
critic_hidden_dims=[512, 512, 256, 128]
```

**Humanoid/Quadruped Robots:**
```python
actor_hidden_dims=[512, 512, 256]
critic_hidden_dims=[512, 512, 256]
```

#### **Activation Functions:**
- **`elu`**: Best for robotics (smooth, non-zero gradient)
- **`relu`**: Fast but can cause dead neurons
- **`tanh`**: Good for bounded outputs
- **`swish`**: Modern alternative to ReLU

### **1.3 Advanced Policy Features**

#### **RSL-RL Advanced Features:**
```python
# Symmetry learning for locomotion
symmetry = RslRlSymmetryCfg(
    enabled=True,
    symmetry_axis="x",  # Mirror along x-axis
    symmetry_groups=["left_leg", "right_leg"]
)

# Curiosity exploration (RND)
rnd = RslRlRndCfg(
    enabled=True,
    hidden_dims=[256, 256],
    learning_rate=1e-3
)

# Recurrent networks
policy = RslRlPpoActorCriticCfg(
    use_rnn=True,
    rnn_hidden_size=256,
    rnn_num_layers=1
)
```

#### **SKRL Multi-Agent Features:**
```python
# Multi-agent configuration
agent = SkrlPpoAgentCfg(
    multi_agent=True,
    shared_network=False,  # Separate networks per agent
    communication_channels=4  # Inter-agent communication
)
```

---

## 🎯 **Part 2: Environment Configuration and Observation/Action Spaces**

### **2.1 Environment Configuration**

```python
# source/your_project/your_project/tasks/direct_single-agent/my_robot_env_cfg.py
import math
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from ..assets.my_robot import MY_ROBOT_CFG

@configclass
class MyRobotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the robot environment."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    # Your custom robot
    robot: ArticulationCfg = MY_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    
    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

@configclass
class MyRobotEnvCfg(DirectRLEnvCfg):
    """Environment configuration for your custom robot."""
    
    # Environment Settings
    decimation = 4                    # Control frequency = sim_freq / decimation
    episode_length_s = 20.0          # Episode length in seconds
    
    # Action and Observation Spaces
    action_space = 6                 # Number of DOF for your robot
    observation_space = 12           # Total observation dimensions
    state_space = 0                  # Privileged information (0 = none)
    
    # Simulation Configuration
    sim = SimulationCfg(
        dt=1/120,                    # Simulation timestep
        substeps=1,                  # Physics substeps
        up_axis="z",                 # Gravity direction
        gravity=(0.0, 0.0, -9.81),  # Gravity vector
    )
    
    # Scene Configuration
    scene: MyRobotSceneCfg = MyRobotSceneCfg(
        num_envs=4096,               # Number of parallel environments
        env_spacing=2.0,             # Spacing between environments
        replicate_physics=True,      # Use GPU physics replication
        clone_in_fabric=True,        # Use Isaac Sim Fabric for cloning
    )
    
    # Viewer Configuration
    viewer = DirectRLEnvCfg.ViewerCfg(
        eye=(3.0, 3.0, 3.0),        # Camera position
        lookat=(0.0, 0.0, 0.0),     # Camera target
    )
    
    # Reset Configuration
    max_joint_pos = 3.14            # Maximum joint position [rad]
    max_joint_vel = 10.0            # Maximum joint velocity [rad/s]
    initial_joint_pos_range = [-0.1, 0.1]  # Initial joint position range [rad]
    
    # Reward Configuration
    rew_scale_alive = 1.0           # Reward for staying alive
    rew_scale_terminated = -10.0    # Penalty for termination
    rew_scale_joint_pos = -1.0      # Penalty for joint position deviation
    rew_scale_joint_vel = -0.1      # Penalty for joint velocity
    rew_scale_action = -0.01        # Penalty for large actions
    rew_scale_energy = -0.001      # Penalty for energy consumption
```

### **2.2 Observation Space Design**

#### **Basic Joint State Observations:**
```python
def _get_observations(self) -> dict:
    """Get observations for the policy."""
    
    # Joint positions and velocities
    joint_pos = self.robot.data.joint_pos[:, self._joint_indices]
    joint_vel = self.robot.data.joint_vel[:, self._joint_indices]
    
    # Root pose (position and orientation)
    root_pos = self.robot.data.root_pos_w
    root_quat = self.robot.data.root_quat_w
    
    # Root linear and angular velocities
    root_lin_vel = self.robot.data.root_lin_vel_w
    root_ang_vel = self.robot.data.root_ang_vel_w
    
    # Combine all observations
    obs = torch.cat([
        joint_pos,           # Joint positions
        joint_vel,           # Joint velocities
        root_pos,            # Root position
        root_quat,           # Root orientation (quaternion)
        root_lin_vel,        # Root linear velocity
        root_ang_vel,        # Root angular velocity
    ], dim=-1)
    
    return {"policy": obs}
```

#### **Advanced Observations with Sensors:**
```python
def _get_observations(self) -> dict:
    """Get observations including sensor data."""
    
    # Basic joint state
    joint_pos = self.robot.data.joint_pos[:, self._joint_indices]
    joint_vel = self.robot.data.joint_vel[:, self._joint_indices]
    
    # IMU data (if available)
    imu_data = self.imu.data.accel  # Accelerometer data
    
    # Force/torque sensors (if available)
    ft_data = self.ft_sensor.data.force_torque
    
    # Camera data (if using vision)
    camera_data = self.camera.data.rgb  # RGB image
    
    # Combine observations
    obs = torch.cat([
        joint_pos,
        joint_vel,
        imu_data.flatten(start_dim=1),  # Flatten sensor data
        ft_data.flatten(start_dim=1),
    ], dim=-1)
    
    return {
        "policy": obs,
        "camera": camera_data,  # Separate camera observation
    }
```

### **2.3 Action Space Design**

#### **Joint Torque Control:**
```python
def _apply_action(self) -> None:
    """Apply actions to the robot."""
    
    # Scale actions to torque limits
    scaled_actions = self.actions * self.cfg.action_scale
    
    # Apply joint torques
    self.robot.set_joint_effort_target(
        scaled_actions,
        joint_ids=self._joint_indices
    )
```

#### **Joint Position Control:**
```python
def _apply_action(self) -> None:
    """Apply actions as joint position targets."""
    
    # Scale actions to joint position range
    joint_pos_targets = self.actions * self.cfg.max_joint_pos
    
    # Apply joint position targets
    self.robot.set_joint_position_target(
        joint_pos_targets,
        joint_ids=self._joint_indices
    )
```

---

## 🎯 **Part 3: Reward System Design**

### **3.1 Basic Reward Components**

#### **Survival Reward:**
```python
def _get_rewards(self) -> torch.Tensor:
    """Compute rewards for the robot."""
    
    # Basic survival reward
    alive_reward = torch.ones(self.num_envs, device=self.device) * self.cfg.rew_scale_alive
    
    return alive_reward
```

#### **Multi-Component Reward System:**
```python
def _get_rewards(self) -> torch.Tensor:
    """Compute comprehensive rewards."""
    
    # 1. Survival reward
    alive_reward = torch.ones(self.num_envs, device=self.device) * self.cfg.rew_scale_alive
    
    # 2. Joint position penalty (encourage staying near zero)
    joint_pos_penalty = torch.sum(torch.abs(self.joint_pos[:, self._joint_indices]), dim=1)
    joint_pos_reward = joint_pos_penalty * self.cfg.rew_scale_joint_pos
    
    # 3. Joint velocity penalty (encourage smooth motion)
    joint_vel_penalty = torch.sum(torch.abs(self.joint_vel[:, self._joint_indices]), dim=1)
    joint_vel_reward = joint_vel_penalty * self.cfg.rew_scale_joint_vel
    
    # 4. Action penalty (encourage small actions)
    action_penalty = torch.sum(self.actions**2, dim=1)
    action_reward = action_penalty * self.cfg.rew_scale_action
    
    # 5. Energy penalty (encourage efficiency)
    energy_consumption = torch.sum(torch.abs(self.actions * self.joint_vel[:, self._joint_indices]), dim=1)
    energy_reward = energy_consumption * self.cfg.rew_scale_energy
    
    # Total reward
    total_reward = alive_reward + joint_pos_reward + joint_vel_reward + action_reward + energy_reward
    
    return total_reward
```

### **3.2 Task-Specific Reward Systems**

#### **Locomotion Reward (Walking/Running):**
```python
def _get_rewards(self) -> torch.Tensor:
    """Reward system for locomotion tasks."""
    
    # 1. Forward progress reward
    root_pos = self.robot.data.root_pos_w
    forward_progress = root_pos[:, 0] - self.initial_root_pos[:, 0]
    progress_reward = forward_progress * self.cfg.rew_scale_progress
    
    # 2. Upright orientation reward
    root_quat = self.robot.data.root_quat_w
    up_vector = torch.tensor([0, 0, 1], device=self.device)
    up_proj = torch.sum(root_quat[:, 1:4] * up_vector, dim=1)
    upright_reward = torch.where(up_proj > 0.9, 
                                torch.ones_like(up_proj) * self.cfg.rew_scale_upright,
                                up_proj * self.cfg.rew_scale_upright)
    
    # 3. Velocity tracking reward
    target_velocity = self.cfg.target_velocity
    current_velocity = self.robot.data.root_lin_vel_w[:, 0]
    velocity_error = torch.abs(current_velocity - target_velocity)
    velocity_reward = -velocity_error * self.cfg.rew_scale_velocity
    
    # 4. Smooth motion reward
    joint_accel = (self.joint_vel[:, self._joint_indices] - 
                   self.prev_joint_vel[:, self._joint_indices]) / self.cfg.sim.dt
    smoothness_reward = -torch.sum(joint_accel**2, dim=1) * self.cfg.rew_scale_smoothness
    
    # 5. Action penalty
    action_penalty = torch.sum(self.actions**2, dim=1) * self.cfg.rew_scale_action
    
    total_reward = (progress_reward + upright_reward + velocity_reward + 
                   smoothness_reward + action_penalty)
    
    return total_reward
```

#### **Manipulation Reward (Pick and Place):**
```python
def _get_rewards(self) -> torch.Tensor:
    """Reward system for manipulation tasks."""
    
    # 1. Distance to target reward
    ee_pos = self.robot.data.ee_pos_w
    target_pos = self.target.data.root_pos_w
    distance = torch.norm(ee_pos - target_pos, p=2, dim=1)
    distance_reward = -distance * self.cfg.rew_scale_distance
    
    # 2. Orientation alignment reward
    ee_quat = self.robot.data.ee_quat_w
    target_quat = self.target.data.root_quat_w
    orientation_error = torch.norm(ee_quat - target_quat, p=2, dim=1)
    orientation_reward = -orientation_error * self.cfg.rew_scale_orientation
    
    # 3. Grasp success reward
    grasp_success = self._check_grasp_success()
    grasp_reward = grasp_success.float() * self.cfg.rew_scale_grasp
    
    # 4. Lift success reward
    lift_success = self._check_lift_success()
    lift_reward = lift_success.float() * self.cfg.rew_scale_lift
    
    # 5. Action penalty
    action_penalty = torch.sum(self.actions**2, dim=1) * self.cfg.rew_scale_action
    
    total_reward = (distance_reward + orientation_reward + grasp_reward + 
                   lift_reward + action_penalty)
    
    return total_reward
```

#### **Balance Reward (Cartpole-like):**
```python
def _get_rewards(self) -> torch.Tensor:
    """Reward system for balance tasks."""
    
    # 1. Pole angle reward (encourage staying upright)
    pole_angle = self.joint_pos[:, self._pole_dof_idx]
    pole_angle_reward = -torch.abs(pole_angle) * self.cfg.rew_scale_pole_angle
    
    # 2. Cart position reward (encourage staying centered)
    cart_pos = self.joint_pos[:, self._cart_dof_idx]
    cart_pos_reward = -torch.abs(cart_pos) * self.cfg.rew_scale_cart_pos
    
    # 3. Velocity penalty (encourage smooth motion)
    pole_vel = self.joint_vel[:, self._pole_dof_idx]
    cart_vel = self.joint_vel[:, self._cart_dof_idx]
    velocity_penalty = -(torch.abs(pole_vel) + torch.abs(cart_vel)) * self.cfg.rew_scale_velocity
    
    # 4. Action penalty
    action_penalty = torch.sum(self.actions**2, dim=1) * self.cfg.rew_scale_action
    
    # 5. Survival bonus
    survival_bonus = torch.ones(self.num_envs, device=self.device) * self.cfg.rew_scale_alive
    
    total_reward = (pole_angle_reward + cart_pos_reward + velocity_penalty + 
                   action_penalty + survival_bonus)
    
    return total_reward
```

### **3.3 Reward Scaling Guidelines**

#### **Reward Scale Ranges:**
```python
# Typical reward scales for different components
rew_scale_alive = 1.0           # Survival reward
rew_scale_progress = 10.0       # Task progress
rew_scale_distance = -1.0       # Distance penalties
rew_scale_action = -0.01        # Action penalties
rew_scale_energy = -0.001      # Energy penalties
rew_scale_terminated = -10.0    # Termination penalties
```

#### **Reward Balancing Tips:**
1. **Start Simple**: Begin with basic survival + action penalty
2. **Add Gradually**: Add one reward component at a time
3. **Monitor Magnitudes**: Ensure no single component dominates
4. **Use Logging**: Track individual reward components
5. **Adjust Dynamically**: Scale rewards based on training progress

---

## 🎯 **Part 4: Complete Environment Implementation**

### **4.1 Full Environment Class**

```python
# source/your_project/your_project/tasks/direct_single-agent/my_robot_env.py
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .my_robot_env_cfg import MyRobotEnvCfg

class MyRobotEnv(DirectRLEnv):
    """Environment for your custom robot."""
    
    cfg: MyRobotEnvCfg
    
    def __init__(self, cfg: MyRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Get joint indices
        self._joint_indices, _ = self.robot.find_joints(".*")
        
        # Store robot data
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.root_pos = self.robot.data.root_pos_w
        self.root_quat = self.robot.data.root_quat_w
        
        # Store initial root position for progress tracking
        self.initial_root_pos = self.root_pos.clone()
        
        # Store previous joint velocities for smoothness reward
        self.prev_joint_vel = self.joint_vel.clone()
    
    def _setup_scene(self):
        """Set up the simulation scene."""
        # Create robot
        self.robot = Articulation(self.cfg.scene.robot)
        
        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # Add robot to scene
        self.scene.articulations["robot"] = self.robot
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-physics step processing."""
        # Store previous joint velocities
        self.prev_joint_vel = self.joint_vel.clone()
        
        # Store actions
        self.actions = actions.clone()
    
    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        # Scale actions to torque limits
        scaled_actions = self.actions * self.cfg.action_scale
        
        # Apply joint torques
        self.robot.set_joint_effort_target(
            scaled_actions,
            joint_ids=self._joint_indices
        )
    
    def _get_observations(self) -> dict:
        """Get observations for the policy."""
        # Joint positions and velocities
        joint_pos = self.joint_pos[:, self._joint_indices]
        joint_vel = self.joint_vel[:, self._joint_indices]
        
        # Root pose
        root_pos = self.root_pos
        root_quat = self.root_quat
        
        # Root velocities
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_w
        
        # Combine observations
        obs = torch.cat([
            joint_pos,
            joint_vel,
            root_pos,
            root_quat,
            root_lin_vel,
            root_ang_vel,
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for the robot."""
        # 1. Survival reward
        alive_reward = torch.ones(self.num_envs, device=self.device) * self.cfg.rew_scale_alive
        
        # 2. Joint position penalty
        joint_pos_penalty = torch.sum(torch.abs(self.joint_pos[:, self._joint_indices]), dim=1)
        joint_pos_reward = joint_pos_penalty * self.cfg.rew_scale_joint_pos
        
        # 3. Joint velocity penalty
        joint_vel_penalty = torch.sum(torch.abs(self.joint_vel[:, self._joint_indices]), dim=1)
        joint_vel_reward = joint_vel_penalty * self.cfg.rew_scale_joint_vel
        
        # 4. Action penalty
        action_penalty = torch.sum(self.actions**2, dim=1)
        action_reward = action_penalty * self.cfg.rew_scale_action
        
        # 5. Energy penalty
        energy_consumption = torch.sum(torch.abs(self.actions * self.joint_vel[:, self._joint_indices]), dim=1)
        energy_reward = energy_consumption * self.cfg.rew_scale_energy
        
        # Total reward
        total_reward = alive_reward + joint_pos_reward + joint_vel_reward + action_reward + energy_reward
        
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        # Time-based termination
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Joint limit termination
        joint_pos_exceeded = torch.any(torch.abs(self.joint_pos[:, self._joint_indices]) > self.cfg.max_joint_pos, dim=1)
        
        # Joint velocity termination
        joint_vel_exceeded = torch.any(torch.abs(self.joint_vel[:, self._joint_indices]) > self.cfg.max_joint_vel, dim=1)
        
        # Combined termination
        terminated = joint_pos_exceeded | joint_vel_exceeded
        
        return terminated, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # Reset robot to default state
        super()._reset_idx(env_ids)
        
        # Randomize joint positions
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos += sample_uniform(
            self.cfg.initial_joint_pos_range[0],
            self.cfg.initial_joint_pos_range[1],
            joint_pos.shape,
            joint_pos.device,
        )
        
        # Reset joint velocities
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        
        # Reset root state
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Write to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Update initial root position
        self.initial_root_pos[env_ids] = default_root_state[:, :3]
```

---

## 🎯 **Part 5: Training Configuration and Execution**

### **5.1 Training Script Usage**

```bash
# Basic training
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --num_envs=4096 \
    --max_iterations=1500

# Training with specific hyperparameters
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --num_envs=4096 \
    --max_iterations=1500 \
    agent.algorithm.learning_rate=0.001 \
    agent.policy.actor_hidden_dims=[512,512,256] \
    agent.algorithm.entropy_coef=0.01

# Training with video recording
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --video \
    --video_length=200 \
    --video_interval=2000
```

### **5.2 Hyperparameter Tuning Guidelines**

#### **Learning Rate:**
- **Start**: 1e-3
- **Range**: 1e-4 to 1e-2
- **Adjust**: Lower if unstable, higher if slow learning

#### **Network Size:**
- **Small robots**: [128, 128, 64]
- **Medium robots**: [256, 256, 128]
- **Large robots**: [512, 512, 256]

#### **Entropy Coefficient:**
- **Start**: 0.01
- **Range**: 0.001 to 0.1
- **Adjust**: Higher for more exploration

#### **PPO Clipping:**
- **Start**: 0.2
- **Range**: 0.1 to 0.3
- **Adjust**: Lower for more stable updates

---

## 🎯 **Part 6: Monitoring and Debugging**

### **6.1 Reward Component Logging**

```python
def _get_rewards(self) -> torch.Tensor:
    """Compute rewards with logging."""
    
    # Compute individual reward components
    alive_reward = torch.ones(self.num_envs, device=self.device) * self.cfg.rew_scale_alive
    joint_pos_reward = torch.sum(torch.abs(self.joint_pos[:, self._joint_indices]), dim=1) * self.cfg.rew_scale_joint_pos
    action_reward = torch.sum(self.actions**2, dim=1) * self.cfg.rew_scale_action
    
    total_reward = alive_reward + joint_pos_reward + action_reward
    
    # Log reward components
    if "log" not in self.extras:
        self.extras["log"] = {}
    
    self.extras["log"]["alive_reward"] = alive_reward.mean()
    self.extras["log"]["joint_pos_reward"] = joint_pos_reward.mean()
    self.extras["log"]["action_reward"] = action_reward.mean()
    self.extras["log"]["total_reward"] = total_reward.mean()
    
    return total_reward
```

### **6.2 Training Monitoring**

```bash
# Monitor training with TensorBoard
tensorboard --logdir=logs/rsl_rl/my_robot_task

# Check training progress
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --checkpoint=logs/rsl_rl/my_robot_task/model_1500.pt
```

---

## 🎯 **Summary: Complete Implementation Checklist**

### **✅ Step-by-Step Implementation:**

1. **Create Robot Asset Configuration** (`my_robot.py`)
2. **Define Environment Configuration** (`my_robot_env_cfg.py`)
3. **Implement Environment Class** (`my_robot_env.py`)
4. **Create Policy Configurations** (`agents/rsl_rl_ppo_cfg.py`, `agents/skrl_ppo_cfg.yaml`)
5. **Design Reward System** (in `_get_rewards()` method)
6. **Define Observation Space** (in `_get_observations()` method)
7. **Implement Action Application** (in `_apply_action()` method)
8. **Set Termination Conditions** (in `_get_dones()` method)
9. **Configure Reset Logic** (in `_reset_idx()` method)
10. **Test and Train** (using `train.py`)

### **🎯 Key Success Factors:**

1. **Start Simple**: Begin with basic survival + action penalty
2. **Incremental Complexity**: Add reward components gradually
3. **Monitor Everything**: Log all reward components
4. **Hyperparameter Tuning**: Adjust based on training progress
5. **Robust Testing**: Test with random and zero agents first

This comprehensive guide gives you everything needed to implement RL for your custom robot! The key is to start simple and gradually add complexity as your robot learns the basic behaviors.