# Isaac Lab Complete Reference for AI Models

## Complete Isaac Lab File Structure

### Core Extensions
```
source/
├── isaaclab/           # Core framework
├── isaaclab_assets/    # Pre-configured assets
├── isaaclab_tasks/     # Pre-configured environments
├── isaaclab_rl/        # RL library wrappers
└── isaaclab_mimic/     # Imitation learning
```

### Core Framework Modules
```
source/isaaclab/isaaclab/
├── actuators/          # Actuator models (ImplicitActuator, DCMotor, etc.)
├── assets/             # Asset types (articulations, rigid objects, deformable objects)
├── envs/               # Environment base classes (DirectRLEnv, ManagerBasedRLEnv)
├── managers/           # Manager-based workflow components
│   ├── action_manager.py
│   ├── observation_manager.py
│   ├── reward_manager.py
│   ├── termination_manager.py
│   ├── event_manager.py
│   └── curriculum_manager.py
├── scene/              # Interactive scene management
├── sensors/            # Sensor implementations
│   ├── camera/
│   ├── contact_sensor/
│   ├── imu/
│   └── ray_caster/
├── sim/                # Simulation core and configuration
│   ├── simulation_context.py
│   ├── simulation_cfg.py
│   └── spawners/
├── terrains/           # Terrain generation (height fields, procedural)
├── utils/              # Utility functions
│   ├── math.py
│   ├── string.py
│   ├── configclass.py
│   └── modifiers/
└── assets/             # Asset types
    ├── articulation/
    ├── rigid_object/
    ├── deformable_object/
    └── surface_gripper/
```

### Scripts Organization
```
scripts/
├── benchmarks/         # Performance testing
│   ├── benchmark_cameras.py
│   ├── benchmark_load_robot.py
│   └── benchmark_non_rl.py
├── demos/              # Robot demonstrations
│   ├── arms.py
│   ├── quadrupeds.py
│   ├── hands.py
│   ├── bipeds.py
│   └── sensors/
├── environments/       # Environment testing
│   ├── list_envs.py
│   ├── random_agent.py
│   └── zero_agent.py
├── imitation_learning/ # Imitation learning workflows
│   └── isaaclab_mimic/
├── reinforcement_learning/ # RL training workflows
│   ├── rsl_rl/
│   ├── rl_games/
│   ├── skrl/
│   └── sb3/
├── tools/              # Utility tools
│   ├── convert_urdf.py
│   ├── convert_mjcf.py
│   ├── record_demos.py
│   ├── replay_demos.py
│   └── pretrained_checkpoint.py
└── tutorials/          # Learning tutorials
    ├── 00_sim/
    ├── 01_assets/
    ├── 02_scene/
    ├── 03_envs/
    └── 04_sensors/
```

## Core Workflows

### Environment Creation via isaaclab.bat --new

Command execution flow:
```bash
isaaclab.bat --new
```

Internal process:
1. Batch script detects --new argument
2. Installs template dependencies: `pip install -q -r tools\template\requirements.txt`
3. Launches template generator: `python tools\template\cli.py`

Template generator collects:
- Project type: External (recommended) or Internal
- Project path: Must be outside Isaac Lab directory
- Project name: Valid Python identifier
- Workflow: Direct single-agent, Direct multi-agent, Manager-based single-agent
- RL library: rl_games, rsl_rl, skrl, sb3
- Algorithms: PPO, AMP (single-agent), IPPO, MAPPO (multi-agent)
- **Note**: Only these 4 algorithms are supported across all libraries in the template generator

Generated project structure:
```
project_name/
├── .dockerignore, .flake8, .gitattributes, .gitignore, .pre-commit-config.yaml
├── README.md
├── scripts/
│   ├── [rl_library_name]/
│   ├── list_envs.py
│   ├── zero_agent.py, random_agent.py
├── source/
│   └── [project_name]/
│       ├── config/extension.toml
│       ├── docs/CHANGELOG.rst
│       ├── setup.py, pyproject.toml
│       └── [project_name]/
│           ├── tasks/
│           │   ├── [workflow_name]/
│           │   │   └── [task_name]/
│           │   │       ├── __init__.py
│           │   │       ├── agents/
│           │   │       ├── [task]_env.py, [task]_env_cfg.py
│           │   │       └── mdp/
│           │   └── __init__.py
│           └── ui_extension_example.py
└── .vscode/
```

### Workflow Differences

Direct Workflow:
- Single-agent: Inherits from DirectRLEnv, uses DirectRLEnvCfg
- Multi-agent: Inherits from DirectMARLEnv, uses DirectMARLEnvCfg
- File structure: Creates [task]_env.py and [task]_env_cfg.py
- Features: Supports multi-agent, fundamental/composite spaces
- Entry point: Direct class reference in gym registration

Manager-based Workflow:
- Single-agent only: Uses ManagerBasedRLEnv, ManagerBasedRLEnvCfg
- File structure: Creates [task]_env_cfg.py + mdp/ folder
- Features: Limited to single-agent, Box spaces only
- Entry point: Uses isaaclab.envs:ManagerBasedRLEnv with config reference
- Architecture: Uses manager pattern (ActionManager, ObservationManager, EventManager)

## Manager-Based Workflow (Complete Implementation)

### Manager System Architecture
The manager-based workflow uses a modular system of managers to handle different aspects of the environment:

```python
from isaaclab.managers import (
    ActionManager, ObservationManager, RewardManager,
    TerminationManager, EventManager, CurriculumManager
)

@configclass
class MyTaskEnvCfg(ManagerBasedRLEnvCfg):
    # Action management
    actions = ActionTermCfg(
        func=JointPositionAction,
        params={"joint_names": [".*"], "scale": 0.25}
    )
    
    # Observation management
    observations = ObservationTermCfg(
        func=JointPositionObservation,
        params={"joint_names": [".*"]}
    )
    
    # Reward management
    rewards = RewardTermCfg(
        func=JointPositionReward,
        params={"joint_names": [".*"], "weight": -1.0}
    )
    
    # Termination management
    terminations = TerminationTermCfg(
        func=JointPositionTermination,
        params={"joint_names": [".*"], "threshold": 0.5}
    )
    
    # Event management (randomization)
    events = EventTermCfg(
        func=RandomizeJointProperties,
        params={"joint_names": [".*"], "mode": "reset"}
    )
```

### Manager Types and Usage
- **ActionManager**: Handles action processing and application
- **ObservationManager**: Manages observation computation and normalization
- **RewardManager**: Computes reward terms and episodic tracking
- **TerminationManager**: Handles termination conditions and timeouts
- **EventManager**: Manages randomization and curriculum events
- **CurriculumManager**: Handles curriculum learning and difficulty progression

### Extension System Integration

Extension metadata in extension.toml:
```toml
[package]
name = "project_name"
version = "0.1.0"
description = "Isaac Lab extension template"
authors = ["Your Name <your.email@example.com>"]

[dependencies]
isaaclab = ">=1.0.0"
```

Task registration:
```python
import gymnasium as gym
from .my_task_env import MyTaskEnv
from .my_task_env_cfg import MyTaskEnvCfg

gym.register(
    id="Template-MyTask-Direct-v0",
    entry_point=MyTaskEnv,
    kwargs={"cfg": MyTaskEnvCfg()},
)
```

Installation:
```bash
pip install -e source/project_name
```

## Sensor Integration

### Available Sensor Types
```python
from isaaclab.sensors import (
    CameraCfg, ContactSensorCfg, ImuCfg, 
    RayCasterCfg, FrameTransformerCfg
)

# Camera sensors
camera = CameraCfg(
    prim_path="/World/robot/camera",
    update_period=0.1,
    height=480,
    width=640
)

# Contact sensors
contact_sensor = ContactSensorCfg(
    prim_path="/World/robot/feet_*",
    update_period=0.0,
    history_length=2
)

# IMU sensors
imu = ImuCfg(
    prim_path="/World/robot/base",
    update_period=0.0
)

# Ray casting sensors
ray_caster = RayCasterCfg(
    prim_path="/World/robot/base",
    max_distance=10.0,
    num_rays=16
)
```

### Sensor Data Access
```python
# Access sensor data
camera_data = self.scene.sensors["camera"].data
contact_data = self.scene.sensors["contact_sensor"].data
imu_data = self.scene.sensors["imu"].data
```

### Sensor Configuration in Environment
```python
@configclass
class MyTaskSceneCfg(InteractiveSceneCfg):
    # Add sensors to scene
    camera = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera",
        spawn=CameraCfg(
            height=480,
            width=640,
            data_types=["rgb", "depth", "segmentation"]
        )
    )
    
    imu = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        spawn=ImuCfg()
    )
```

## Robot Integration

### URDF to USD Conversion

Command line:
```bash
python scripts/tools/convert_urdf.py --input /path/to/robot.urdf --output /path/to/robot.usd
```

Programmatic:
```python
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

urdf_cfg = UrdfConverterCfg(
    asset_path="/path/to/robot.urdf",
    usd_path="/path/to/robot.usd",
    joint_drive=True,
    root_link_name="base_link"
)

converter = UrdfConverter(urdf_cfg)
converter.convert()
```

### Robot Configuration

```python
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/robot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=1000.0,
            damping=100.0,
        ),
    },
)
```

## Environment Implementation

### DirectRLEnv Base Class Pattern

```python
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

class MyTaskEnv(DirectRLEnv):
    cfg: MyTaskEnvCfg
    
    def __init__(self, cfg: MyTaskEnvCfg, render_mode: str | None = None, **kwargs):
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
        # Store previous joint velocities
        self.prev_joint_vel = self.joint_vel.clone()
        
        # Store actions
        self.actions = actions.clone()
    
    def _apply_action(self) -> None:
        # Scale actions to torque limits
        scaled_actions = self.actions * self.cfg.action_scale
        
        # Apply joint torques
        self.robot.set_joint_effort_target(
            scaled_actions,
            joint_ids=self._joint_indices
        )
    
    def _get_observations(self) -> dict:
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

### Environment Configuration

```python
import math
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class MyTaskSceneCfg(InteractiveSceneCfg):
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    # Your robot
    robot: ArticulationCfg = MY_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    
    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

@configclass
class MyTaskEnvCfg(DirectRLEnvCfg):
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
    scene: MyTaskSceneCfg = MyTaskSceneCfg(
        num_envs=4096,               # Number of parallel environments
        env_spacing=2.0,             # Spacing between environments
        replicate_physics=True,       # Use GPU physics replication
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

## Reward Systems

### Basic Reward Components

```python
def _get_rewards(self) -> torch.Tensor:
    # Basic survival reward
    alive_reward = torch.ones(self.num_envs, device=self.device) * self.cfg.rew_scale_alive
    
    return alive_reward
```

### Multi-Component Reward System

```python
def _get_rewards(self) -> torch.Tensor:
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

### Locomotion Reward (Walking/Running)

```python
def _get_rewards(self) -> torch.Tensor:
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

### Manipulation Reward (Pick and Place)

```python
def _get_rewards(self) -> torch.Tensor:
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

### Balance Reward (Cartpole-like)

```python
def _get_rewards(self) -> torch.Tensor:
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

### Reward Scaling Guidelines

```python
# Typical reward scales for different components
rew_scale_alive = 1.0           # Survival reward
rew_scale_progress = 10.0       # Task progress
rew_scale_distance = -1.0       # Distance penalties
rew_scale_action = -0.01        # Action penalties
rew_scale_energy = -0.001      # Energy penalties
rew_scale_terminated = -10.0    # Termination penalties
```

## RL Library Integration

### RSL-RL Configuration

```python
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

### SKRL Configuration

**SKRL Algorithm Selection Logic:**
SKRL uses the `--algorithm` parameter to determine the default agent configuration:
- If `--agent` is not specified, SKRL automatically selects the agent config based on `--algorithm`
- For PPO: Uses `skrl_cfg_entry_point` 
- For IPPO/MAPPO: Uses `skrl_{algorithm}_cfg_entry_point`
- The `--ml_framework` parameter determines the backend (torch, jax, jax-numpy)

```yaml
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

### RL Library Comparison

| Feature | RSL-RL | RL-Games | SKRL | SB3 |
|---------|--------|----------|------|-----|
| Performance | ~1X | ~1X | ~1X | ~1.6X (verified) |
| Multi-Agent | ❌ | ❌ | ✅ | ❌ |
| Distributed | ❌ | ✅ | ✅ | ❌ |
| Vectorized | ✅ | ✅ | ✅ | ❌ |
| ML Frameworks | PyTorch | PyTorch | PyTorch, JAX | PyTorch |
| Observation Spaces | Box | Box, Dict | Box, Dict, Composite | Box |
| Algorithms | PPO, AMP | PPO, AMP | PPO, AMP, IPPO, MAPPO | PPO (primary) |
| Robotics Features | ✅ Advanced | ✅ High-perf | ✅ Research | ❌ Basic |
| Ease of Use | Medium | Medium | Medium | Easy |

**Note**: Performance benchmarks based on Isaac-Humanoid-v0 training on RTX PRO 6000 GPU with 4096 environments for 65.5M steps. SB3 is approximately 1.6X slower than other libraries, not 30X slower as sometimes claimed.

**Note**: While SB3 supports many algorithms (SAC, TD3, DQN, etc.), Isaac Lab officially supports PPO as the primary algorithm. Other SB3 algorithms may work but are not officially tested or supported.

### When to Choose Each Library

Choose RSL-RL when:
- Robotics locomotion (quadrupeds, bipeds)
- Need symmetry learning
- Want curiosity exploration
- Single-agent robotics
- Research requiring advanced features

Choose RL-Games when:
- High-performance training
- Distributed training (multi-GPU/node)
- Production applications
- Asymmetric actor-critic
- Population-based training
- Maximum performance needed

Choose SKRL when:
- Multi-agent environments
- JAX backend needed
- Complex observation spaces
- Research flexibility
- Custom architectures
- Academic projects

Choose SB3 when:
- Learning RL concepts
- Quick prototyping
- Maximum stability needed
- Small-scale experiments
- Performance not critical
- Easy debugging

### Network Architecture Guidelines

Simple Robots (2-6 DOF):
```python
actor_hidden_dims=[128, 128, 64]
critic_hidden_dims=[128, 128, 64]
```

Medium Complexity (7-12 DOF):
```python
actor_hidden_dims=[256, 256, 128]
critic_hidden_dims=[256, 256, 128]
```

Complex Robots (13+ DOF):
```python
actor_hidden_dims=[512, 512, 256, 128]
critic_hidden_dims=[512, 512, 256, 128]
```

Humanoid/Quadruped Robots:
```python
actor_hidden_dims=[512, 512, 256]
critic_hidden_dims=[512, 512, 256]
```

Activation Functions:
- elu: Best for robotics (smooth, non-zero gradient)
- relu: Fast but can cause dead neurons
- tanh: Good for bounded outputs
- swish: Modern alternative to ReLU

## Training Commands

**Note**: The `--agent` parameter defaults vary by library:
- RSL-RL: `rsl_rl_cfg_entry_point` (default)
- RL-Games: `rl_games_cfg_entry_point` (default)  
- SKRL: Uses algorithm-based naming (e.g., `skrl_ppo_cfg_entry_point`)
- SB3: `sb3_cfg_entry_point` (default)

Examples below show explicit agent names for clarity.

### RSL-RL Training

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

### RL-Games Training

```bash
# Basic training
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=rl_games_ppo_cfg

# Distributed training
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=rl_games_ppo_cfg \
    --distributed

# With Wandb logging and custom parameters
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=rl_games_ppo_cfg \
    --wandb-project-name=my_project \
    --wandb-entity=my_entity \
    --wandb-name=my_experiment \
    --sigma=0.5
```

### SKRL Training

```bash
# Basic training
isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=skrl_ppo_cfg

# Training with specific algorithm and ML framework
isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=skrl_ppo_cfg \
    --algorithm=PPO \
    --ml_framework=torch

# Multi-agent training with IPPO
isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=skrl_ippo_cfg \
    --algorithm=IPPO \
    --ml_framework=torch
```

### SB3 Training

```bash
# Basic training
isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=sb3_ppo_cfg

# Training with logging intervals
isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=sb3_ppo_cfg \
    --log_interval=100000
```

### Common Training Arguments

**Core Arguments:**
- `--task`: Name of the task/environment
- `--agent`: Name of the RL agent configuration (defaults vary by library)
- `--num_envs`: Number of parallel environments
- `--max_iterations`: Maximum training iterations
- `--seed`: Random seed

**Visualization Arguments:**
- `--video`: Enable video recording
- `--video_length`: Length of recorded videos
- `--video_interval`: Interval between video recordings
- `--headless`: Run without GUI for faster training
- `--enable_cameras`: Enable camera sensors for vision tasks

**Performance Arguments:**
- `--distributed`: Enable distributed training (RL-Games)
- `--disable_fabric`: Disable Isaac Sim Fabric acceleration

**SKRL-Specific Arguments:**
- `--algorithm`: Algorithm type (PPO, IPPO, MAPPO)
- `--ml_framework`: ML framework (torch, jax, jax-numpy)
- `--export_io_descriptors`: Export input/output descriptors for debugging

**RL-Games-Specific Arguments:**
- `--sigma`: Policy initial standard deviation
- `--wandb-project-name`: Wandb project name
- `--wandb-entity`: Wandb entity name
- `--wandb-name`: Wandb experiment name

**Resume Arguments:**
- `--checkpoint`: Path to checkpoint for resuming
- `--resume`: Resume training (RSL-RL)
- `--load_run`: Run timestamp to load (RSL-RL)
- `--load_checkpoint`: Checkpoint iteration to load (RSL-RL)

## Inference Commands

### RSL-RL Playing

```bash
# Use latest checkpoint automatically
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=rsl_rl_ppo_cfg

# Use specific checkpoint
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --checkpoint=logs\rsl_rl\my_robot_task\model_1500.pt

# Use pre-trained checkpoint
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --use_pretrained_checkpoint
```

### RL-Games Playing

```bash
# Use latest checkpoint automatically
isaaclab.bat -p scripts\reinforcement_learning\rl_games\play.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=rl_games_ppo_cfg

# Use specific checkpoint
isaaclab.bat -p scripts\reinforcement_learning\rl_games\play.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=rl_games_ppo_cfg \
    --checkpoint=logs\rl_games\my_config\model.pth
```

### SKRL Playing

```bash
# Use latest checkpoint automatically
isaaclab.bat -p scripts\reinforcement_learning\skrl\play.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=skrl_ppo_cfg

# Use specific checkpoint
isaaclab.bat -p scripts\reinforcement_learning\skrl\play.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=skrl_ppo_cfg \
    --checkpoint=logs\skrl\my_task\checkpoints\agent_1500.pt
```

### SB3 Playing

```bash
# Use latest checkpoint automatically
isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=sb3_ppo_cfg

# Use specific checkpoint
isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py \
    --task=Template-MyRobot-Direct-v0 \
    --agent=sb3_ppo_cfg \
    --checkpoint=logs\sb3\my_task\model.zip
```

### Common Playing Arguments

- `--task`: Name of the task/environment
- `--agent`: Name of the RL agent configuration
- `--checkpoint`: Path to trained model checkpoint
- `--use_pretrained_checkpoint`: Use pre-trained models from Nucleus
- `--real-time`: Run in real-time mode
- `--video`: Enable video recording
- `--disable_fabric`: Disable fabric acceleration
- `--headless`: Run without GUI
- `--enable_cameras`: Enable camera sensors for vision tasks
- `--num_envs`: Number of environments (keep small for visualization)

## File Locations

**Note**: File paths may vary based on configuration and Isaac Lab version. The examples below show typical patterns.

### RSL-RL Models

```
logs/
└── rsl_rl/
    └── [experiment_name]/
        └── [timestamp]_[run_name]/
            ├── model_1000.pt
            ├── model_1050.pt
            ├── model_1100.pt
            ├── ...
            ├── model_1500.pt  ← Latest checkpoint
            ├── params/
            │   ├── env.yaml
            │   ├── agent.yaml
            │   ├── env.pkl
            │   └── agent.pkl
            └── events.out.tfevents.*  ← TensorBoard logs
```

### RL-Games Models

```
logs/
└── rl_games/
    └── [config_name]/
        └── [timestamp]/
            ├── [config_name].pth  ← Main checkpoint
            ├── params/
            │   ├── env.yaml
            │   ├── agent.yaml
            │   ├── env.pkl
            │   └── agent.pkl
            └── events.out.tfevents.*
```

### SKRL Models

```
logs/
└── skrl/
    └── [task_name]/
        └── [timestamp]_[algorithm]_[framework]/
            ├── checkpoints/
            │   ├── agent_1000.pt
            │   ├── agent_1050.pt
            │   └── ...
            ├── params/
            │   ├── env.yaml
            │   ├── agent.yaml
            │   ├── env.pkl
            │   └── agent.pkl
            └── events.out.tfevents.*
```

### SB3 Models

```
logs/
└── sb3/
    └── [task_name]/
        └── [timestamp]/
            ├── model.zip  ← Main checkpoint
            ├── model_1000.zip
            ├── model_2000.zip
            ├── ...
            ├── model_vecnormalize.pkl  ← Normalization params
            └── events.out.tfevents.*
```

### Model File Formats

- RSL-RL: `.pt` files (PyTorch checkpoints)
- RL-Games: `.pth` files (PyTorch checkpoints)  
- SKRL: `.pt` files (PyTorch checkpoints)
- SB3: `.zip` files (compressed model + normalization)

**File Size Guidelines:**
- RSL-RL models: ~50-200 MB (depending on network size)
- RL-Games models: ~50-200 MB
- SKRL models: ~50-200 MB
- SB3 models: ~100-300 MB (includes normalization parameters)

**Model Compatibility:**
- Models are not cross-compatible between RL libraries
- Models trained with different Isaac Lab versions may not be compatible
- Always verify model compatibility before deployment

### Finding Checkpoints Programmatically

```python
import os
import glob

# Find latest RSL-RL checkpoint
rsl_logs = glob.glob("logs/rsl_rl/*/model_*.pt")
latest_rsl = max(rsl_logs, key=os.path.getctime)
print(f"Latest RSL-RL checkpoint: {latest_rsl}")

# Find latest RL-Games checkpoint
rl_games_logs = glob.glob("logs/rl_games/*/*.pth")
latest_rl_games = max(rl_games_logs, key=os.path.getctime)
print(f"Latest RL-Games checkpoint: {latest_rl_games}")
```

## Hydra Configuration System

Hydra is a configuration manager that Isaac Lab uses to handle:
- Command-line arguments (`--task`, `--num_envs`, etc.)
- Config files (like YAMLs defining PPO or PhysX parameters)
- Output directories and experiment naming
- Logging for what arguments and overrides were actually used

### Hydra Output Structure

When running training, Hydra creates:
```
outputs/
  2025-10-08_22-31-59/
      hydra/
          config.yaml      # Full merged configuration
          overrides.yaml   # Only manual overrides
          hydra.yaml       # Hydra internal metadata
      train.log
      wandb/
      checkpoints/
```

### Hydra Files Purpose

| File | Purpose |
|------|---------|
| config.yaml | Full merged configuration Hydra used for this run |
| overrides.yaml | Only the options you overrode manually |
| hydra.yaml | Hydra's internal metadata about logging and paths |
| hydra.log | Text log of Hydra's operations |

### Custom Hydra Output Directory

```bash
--config-dir configs --config-name train hydra.run.dir=./runs/${now:%Y-%m-%d_%H-%M-%S}
```

## Non-Training Scripts and Tools

### Demonstration Scripts
```bash
# Robot demonstrations
isaaclab.bat -p scripts/demos/arms.py          # Single-arm manipulators
isaaclab.bat -p scripts/demos/quadrupeds.py   # Legged robots
isaaclab.bat -p scripts/demos/hands.py        # Hand manipulation
isaaclab.bat -p scripts/demos/bipeds.py       # Bipedal robots

# Sensor demonstrations
isaaclab.bat -p scripts/demos/sensors/camera_demo.py
isaaclab.bat -p scripts/demos/sensors/imu_demo.py
isaaclab.bat -p scripts/demos/sensors/contact_demo.py
```

### Utility Tools
```bash
# Asset conversion
isaaclab.bat -p scripts/tools/convert_urdf.py --input robot.urdf --output robot.usd
isaaclab.bat -p scripts/tools/convert_mjcf.py --input robot.xml --output robot.usd

# Demonstration recording/playback
isaaclab.bat -p scripts/tools/record_demos.py --task Isaac-Reach-Franka-v0
isaaclab.bat -p scripts/tools/replay_demos.py --dataset_file demos.hdf5

# Checkpoint management
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py --train rl_games:Isaac-Cartpole-v0
```

### Environment Testing
```bash
# List available environments
isaaclab.bat -p scripts/environments/list_envs.py

# Test with random actions
isaaclab.bat -p scripts/environments/random_agent.py --task Isaac-Cartpole-Direct-v0

# Test with zero actions
isaaclab.bat -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct-v0
```

### Performance Benchmarks
```bash
# Benchmark environment performance
isaaclab.bat -p scripts/benchmarks/benchmark_non_rl.py --task Isaac-Ant-Direct-v0

# Benchmark camera performance
isaaclab.bat -p scripts/benchmarks/benchmark_cameras.py --task Isaac-Cartpole-RGB-Camera-Direct-v0

# Benchmark robot loading
isaaclab.bat -p scripts/benchmarks/benchmark_load_robot.py --robot_path assets/robots/franka.usd
```

### Learning Tutorials
```bash
# Simulation basics
isaaclab.bat -p scripts/tutorials/00_sim/sim_01_basic.py

# Asset management
isaaclab.bat -p scripts/tutorials/01_assets/assets_01_articulation.py

# Scene setup
isaaclab.bat -p scripts/tutorials/02_scene/scene_01_basic.py

# Environment creation
isaaclab.bat -p scripts/tutorials/03_envs/env_01_basic.py

# Sensor integration
isaaclab.bat -p scripts/tutorials/04_sensors/sensors_01_camera.py
```

## Available Tasks

**Important**: Task availability may vary by Isaac Lab version and installed extensions. Always verify available tasks with:
```bash
isaaclab.bat -p scripts\environments\list_envs.py
```

**Task Discovery Tips:**
- Use `findstr` to filter tasks: `isaaclab.bat -p scripts\environments\list_envs.py | findstr "Direct"`
- Check for specific robot tasks: `isaaclab.bat -p scripts\environments\list_envs.py | findstr "Franka"`
- Verify camera tasks: `isaaclab.bat -p scripts\environments\list_envs.py | findstr "Camera"`

### Direct Workflow Tasks

Classic Control Tasks:
- `Isaac-Cartpole-Direct-v0` - Cartpole balancing
- `Isaac-Cartpole-RGB-Camera-Direct-v0` - Cartpole with RGB camera
- `Isaac-Cartpole-Depth-Camera-Direct-v0` - Cartpole with depth camera
- `Isaac-Cart-Double-Pendulum-Direct-v0` - Double pendulum on cart

Locomotion Tasks:
- `Isaac-Ant-Direct-v0` - Ant locomotion
- `Isaac-Humanoid-Direct-v0` - Humanoid locomotion
- `Isaac-Velocity-Flat-Anymal-C-Direct-v0` - ANYmal-C flat terrain locomotion
- `Isaac-Velocity-Rough-Anymal-C-Direct-v0` - ANYmal-C rough terrain locomotion
- `Isaac-Locomotion-Direct-v0` - General locomotion task

Manipulation Tasks:
- `Isaac-Franka-Cabinet-Direct-v0` - Franka cabinet opening
- `Isaac-Inhand-Manipulation-Direct-v0` - In-hand manipulation
- `Isaac-Lift-Direct-v0` - Object lifting task

Quadcopter Tasks:
- `Isaac-Quadcopter-Direct-v0` - Quadcopter control

Hand Tasks:
- `Isaac-Allegro-Hand-Direct-v0` - Allegro hand manipulation
- `Isaac-Shadow-Hand-Direct-v0` - Shadow hand manipulation
- `Isaac-Shadow-Hand-Vision-Direct-v0` - Shadow hand with vision
- `Isaac-Shadow-Hand-Over-Direct-v0` - Shadow hand overhand manipulation

Humanoid AMP Tasks:
- `Isaac-Humanoid-AMP-Direct-v0` - Humanoid Adversarial Motion Priors

Factory Tasks:
- `Isaac-Factory-PegInsert-Direct-v0` - Peg insertion task
- `Isaac-Factory-GearMesh-Direct-v0` - Gear meshing task
- `Isaac-Factory-NutThread-Direct-v0` - Nut threading task

Forge Tasks:
- `Isaac-Forge-Direct-v0` - Forge manipulation task

Automate Tasks:
- `Isaac-Assembly-Direct-v0` - Assembly task
- `Isaac-Disassembly-Direct-v0` - Disassembly task

### Manager-Based Workflow Tasks

Classic Control Tasks:
- `Isaac-Cartpole-v0` - Cartpole balancing
- `Isaac-Ant-v0` - Ant locomotion
- `Isaac-Humanoid-v0` - Humanoid locomotion

Locomotion Tasks:
- `Isaac-Velocity-Flat-Anymal-C-v0` - ANYmal-C flat terrain
- `Isaac-Velocity-Rough-Anymal-C-v0` - ANYmal-C rough terrain
- `Isaac-Velocity-Flat-Anymal-C-Play-v0` - ANYmal-C flat terrain (play mode)
- `Isaac-Velocity-Rough-Anymal-C-Play-v0` - ANYmal-C rough terrain (play mode)
- `Isaac-Velocity-Flat-Anymal-D-v0` - ANYmal-D flat terrain
- `Isaac-Velocity-Rough-Anymal-D-v0` - ANYmal-D rough terrain
- `Isaac-Velocity-Flat-Anymal-D-Play-v0` - ANYmal-D flat terrain (play mode)
- `Isaac-Velocity-Rough-Anymal-D-Play-v0` - ANYmal-D rough terrain (play mode)
- `Isaac-Velocity-Flat-Unitree-Go1-v0` - Unitree Go1 flat terrain
- `Isaac-Velocity-Rough-Unitree-Go1-v0` - Unitree Go1 rough terrain
- `Isaac-Velocity-Flat-Unitree-Go1-Play-v0` - Unitree Go1 flat terrain (play mode)
- `Isaac-Velocity-Rough-Unitree-Go1-Play-v0` - Unitree Go1 rough terrain (play mode)
- `Isaac-Velocity-Flat-Unitree-Go2-v0` - Unitree Go2 flat terrain
- `Isaac-Velocity-Rough-Unitree-Go2-v0` - Unitree Go2 rough terrain
- `Isaac-Velocity-Flat-Unitree-Go2-Play-v0` - Unitree Go2 flat terrain (play mode)
- `Isaac-Velocity-Rough-Unitree-Go2-Play-v0` - Unitree Go2 rough terrain (play mode)
- `Isaac-Velocity-Flat-Spot-v0` - Spot flat terrain
- `Isaac-Velocity-Rough-Spot-v0` - Spot rough terrain
- `Isaac-Velocity-Flat-Spot-Play-v0` - Spot flat terrain (play mode)
- `Isaac-Velocity-Rough-Spot-Play-v0` - Spot rough terrain (play mode)
- `Isaac-Velocity-Flat-H1-v0` - H1 flat terrain
- `Isaac-Velocity-Rough-H1-v0` - H1 rough terrain
- `Isaac-Velocity-Flat-H1-Play-v0` - H1 flat terrain (play mode)
- `Isaac-Velocity-Rough-H1-Play-v0` - H1 rough terrain (play mode)

Manipulation Tasks:

Reach Tasks:
- `Isaac-Reach-Franka-v0` - Franka reach task
- `Isaac-Reach-Franka-Play-v0` - Franka reach task (play mode)

Lift Tasks:
- `Isaac-Lift-Franka-v0` - Franka lift task
- `Isaac-Lift-Franka-Play-v0` - Franka lift task (play mode)

Stack Tasks:
- `Isaac-Stack-Franka-v0` - Franka stack task
- `Isaac-Stack-Franka-Play-v0` - Franka stack task (play mode)
- `Isaac-Stack-Galbot-v0` - Galbot stack task
- `Isaac-Stack-Galbot-Play-v0` - Galbot stack task (play mode)

Cabinet Tasks:
- `Isaac-Cabinet-Franka-v0` - Franka cabinet task
- `Isaac-Cabinet-Franka-Play-v0` - Franka cabinet task (play mode)

Pick and Place Tasks:
- `Isaac-PickPlace-GR1T2-Abs-v0` - GR1T2 pick and place
- `Isaac-NutPour-GR1T2-Pink-IK-Abs-v0` - GR1T2 nut pouring
- `Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0` - GR1T2 exhaust pipe
- `Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0` - GR1T2 pick and place with waist

In-Hand Manipulation Tasks:
- `Isaac-Inhand-Allegro-Hand-v0` - Allegro hand in-hand manipulation
- `Isaac-Inhand-Allegro-Hand-Play-v0` - Allegro hand in-hand manipulation (play mode)

DexSuite Tasks:
- `Isaac-DexSuite-Kuka-Allegro-v0` - Kuka Allegro manipulation
- `Isaac-DexSuite-Kuka-Allegro-Play-v0` - Kuka Allegro manipulation (play mode)

Deploy Tasks:
- `Isaac-Deploy-Franka-v0` - Franka deploy task
- `Isaac-Deploy-Franka-Play-v0` - Franka deploy task (play mode)

Place Tasks:
- `Isaac-Place-Franka-v0` - Franka place task
- `Isaac-Place-Franka-Play-v0` - Franka place task (play mode)

Navigation Tasks:
- `Isaac-Navigation-Anymal-C-v0` - ANYmal-C navigation
- `Isaac-Navigation-Anymal-C-Play-v0` - ANYmal-C navigation (play mode)

Locomotion + Manipulation Tasks:
- `Isaac-Tracking-Anymal-C-v0` - ANYmal-C tracking task
- `Isaac-Tracking-Anymal-C-Play-v0` - ANYmal-C tracking task (play mode)

### Task Categories Summary

| Category | Direct Tasks | Manager-Based Tasks | Total |
|----------|-------------|-------------------|-------|
| Classic Control | 4 | 3 | 7 |
| Locomotion | 6 | 20+ | 26+ |
| Manipulation | 3 | 20+ | 23+ |
| Hand Tasks | 4 | 2 | 6 |
| Factory/Forge | 4 | 0 | 4 |
| Navigation | 0 | 2 | 2 |
| Multi-Modal | 0 | 2 | 2 |
| TOTAL | **21+** | **50+** | **70+** |

**Note**: Task counts are approximate and may vary by Isaac Lab version and installed extensions. Some tasks may be deprecated or require specific extensions. Always verify available tasks with `isaaclab.bat -p scripts\environments\list_envs.py`.

## Quick Reference

### Command Templates

List all available tasks:
```bash
isaaclab.bat -p scripts\environments\list_envs.py
```

Test environment with random actions:
```bash
isaaclab.bat -p scripts\environments\random_agent.py --task=<TASK_NAME>
```

Test environment with zero actions:
```bash
isaaclab.bat -p scripts\environments\zero_agent.py --task=<TASK_NAME>
```

Train any task:
```bash
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=<TASK_NAME> --agent=rsl_rl_ppo_cfg
```

Play any task:
```bash
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=<TASK_NAME> --agent=rsl_rl_ppo_cfg
```

### Task Naming Conventions

- Isaac Lab built-in tasks: `Isaac-<TaskName>-v0`
- External project tasks: `Template-<TaskName>-<Workflow>-v0`
- Direct workflow tasks: `Template-<TaskName>-Direct-v0`
- Manager-based workflow tasks: `Template-<TaskName>-v0`

### Project Installation Patterns

Install existing project:
```bash
python -m pip install -e /path/to/project/source/project_name
```

Install multiple projects:
```bash
python -m pip install -e /path/to/project1/source/project1
python -m pip install -e /path/to/project2/source/project2
python -m pip install -e /path/to/project3/source/project3
```

### Installation Troubleshooting

**Common Installation Issues:**
```bash
# Permission errors on Windows
python -m pip install -e source/project_name --user

# Missing dependencies
pip install -r source/project_name/requirements.txt

# Conflicting packages
pip uninstall conflicting_package
pip install -e source/project_name

# Verify installation
python -c "import project_name; print('Installation successful')"
```

**Development Installation:**
```bash
# Install in development mode with auto-reload
pip install -e source/project_name

# Install with all dependencies
pip install -e source/project_name[dev]

# Install specific RL library support
pip install -e source/project_name[rsl_rl,rl_games]
```

### Training vs Playing Guidelines

Training:
- Use large `--num_envs` (2048-4096)
- Use `--headless` for speed
- Disable video recording
- Focus on throughput

Playing:
- Use small `--num_envs` (1-16)
- Enable GUI (remove `--headless`)
- Enable video recording if needed
- Focus on visualization

### Hyperparameter Guidelines

Learning Rate:
- Start: 1e-3
- Range: 1e-4 to 1e-2
- Adjust: Lower if unstable, higher if slow learning

Network Size:
- Small robots: [128, 128, 64]
- Medium robots: [256, 256, 128]
- Large robots: [512, 512, 256]

Entropy Coefficient:
- Start: 0.01
- Range: 0.001 to 0.1
- Adjust: Higher for more exploration

PPO Clipping:
- Start: 0.2
- Range: 0.1 to 0.3
- Adjust: Lower for more stable updates

### Troubleshooting Patterns

#### Common Issues and Solutions

**Task not found error:**
```bash
# Check if task is available
isaaclab.bat -p scripts\environments\list_envs.py

# Verify project installation
pip list | findstr project_name
```

**Camera tasks not working:**
```bash
# Add --enable_cameras flag for vision tasks
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --agent=rsl_rl_ppo_cfg --enable_cameras
```

**Performance issues:**
```bash
# Use --headless for training
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Ant-v0 --agent=rsl_rl_ppo_cfg --headless

# Disable fabric if having issues
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Ant-v0 --agent=rsl_rl_ppo_cfg --disable_fabric
```

**Checkpoint loading errors:**
```bash
# Verify checkpoint exists
ls logs/rsl_rl/experiment_name/timestamp/

# Use correct agent name (defaults vary by library)
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Ant-v0 --agent=rsl_rl_cfg_entry_point --checkpoint=path/to/model.pt
```

**Real-time playing issues:**
```bash
# Use --real-time flag for real-time inference
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Ant-v0 --agent=rsl_rl_ppo_cfg --real-time
```

**CUDA/GPU Issues:**
```bash
# Check CUDA compatibility
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Verify Isaac Sim GPU support
isaaclab.bat -p scripts\environments\random_agent.py --task=Isaac-Cartpole-Direct-v0 --headless
```

**Import/Dependency Errors:**
```bash
# Reinstall Isaac Lab extensions
isaaclab.bat -i

# Check for conflicting packages
pip list | findstr "torch gym isaac"

# Clean install
pip uninstall isaaclab isaaclab_rl isaaclab_tasks isaaclab_assets
isaaclab.bat -i
```

**Memory Issues:**
```bash
# Reduce number of environments for low-memory GPUs
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Ant-Direct-v0 --agent=rsl_rl_ppo_cfg --num_envs=1024

# Disable fabric acceleration if having issues
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Ant-Direct-v0 --agent=rsl_rl_ppo_cfg --disable_fabric
```

**Task Not Found Errors:**
```bash
# Verify task exists
isaaclab.bat -p scripts\environments\list_envs.py | findstr "YourTaskName"

# Check project installation
pip list | findstr your_project_name

# Reinstall project
pip uninstall your_project_name
pip install -e source/your_project_name
```

### Monitoring Training

Start TensorBoard:
```bash
tensorboard --logdir logs/
```

Check training logs:
```bash
# Unix/Linux
tail -f logs/rsl_rl/experiment_name/timestamp/training.log

# Windows PowerShell
Get-Content logs/rsl_rl/experiment_name/timestamp/training.log -Wait
```

Check configuration files:
```bash
# Unix/Linux
cat logs/rsl_rl/experiment_name/timestamp/params/agent.yaml
cat logs/rsl_rl/experiment_name/timestamp/params/env.yaml

# Windows PowerShell
Get-Content logs/rsl_rl/experiment_name/timestamp/params/agent.yaml
Get-Content logs/rsl_rl/experiment_name/timestamp/params/env.yaml
```

### Resume Training

**RSL-RL Resume:**
```bash
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task=<TASK_NAME> \
    --agent=rsl_rl_cfg_entry_point \
    --resume \
    --load_run=2025-01-08_22-31-59 \
    --load_checkpoint=1500
```

**RL-Games Resume:**
```bash
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py \
    --task=<TASK_NAME> \
    --agent=rl_games_cfg_entry_point \
    --checkpoint=logs\rl_games\config_name\timestamp\config_name.pth
```

**SKRL Resume:**
```bash
isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py \
    --task=<TASK_NAME> \
    --agent=skrl_ppo_cfg \
    --checkpoint=logs\skrl\task_name\timestamp_PPO_torch\checkpoints\agent_1500.pt
```

**SB3 Resume:**
```bash
isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py \
    --task=<TASK_NAME> \
    --agent=sb3_cfg_entry_point \
    --checkpoint=logs\sb3\task_name\timestamp\model.zip
```

This reference provides complete Isaac Lab implementation knowledge for AI models, covering all workflows from environment creation to deployment.

## Advanced Features

### Imitation Learning (Isaac Lab Mimic)

Isaac Lab supports imitation learning through the Mimic extension:

```bash
# Install mimic extension
isaaclab.bat -i mimic

# Generate demonstration dataset
isaaclab.bat -p scripts\imitation_learning\isaaclab_mimic\generate_dataset.py \
    --task=Isaac-Reach-Franka-v0 \
    --num_demos=100 \
    --demo_path=demos/reach_franka.hdf5

# Train with imitation learning
isaaclab.bat -p scripts\imitation_learning\isaaclab_mimic\train.py \
    --task=Isaac-Reach-Franka-v0 \
    --demo_path=demos/reach_franka.hdf5 \
    --agent=mimic_ppo_cfg
```

### Sim2Sim Transfer

Transfer trained models between different simulation environments:

```bash
# Train source environment
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task=Isaac-Velocity-Flat-Anymal-C-v0 \
    --agent=rsl_rl_ppo_cfg

# Transfer to target environment
isaaclab.bat -p scripts\sim2sim_transfer\rsl_rl_transfer.py \
    --source_task=Isaac-Velocity-Flat-Anymal-C-v0 \
    --target_task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --source_checkpoint=logs/rsl_rl/anymal_flat/model_1500.pt
```

### Population-Based Training (RL-Games)

Train multiple agents with different hyperparameters:

```bash
# Enable population-based training
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rl_games_ppo_cfg \
    --pbt=True \
    --pbt_population_size=8
```

### Custom Hydra Configuration

Advanced Hydra configuration management:

```bash
# Use custom config directory
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --config-dir=my_configs \
    --config-name=custom_train

# Override multiple parameters
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    agent.algorithm.learning_rate=0.001 \
    agent.policy.actor_hidden_dims=[512,512,256] \
    agent.algorithm.entropy_coef=0.01 \
    scene.num_envs=2048

# Custom output directory with timestamp
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    hydra.run.dir=./experiments/\${now:%Y-%m-%d_%H-%M-%S}
```

## Debugging and Profiling

### TensorBoard Integration

```bash
# Start TensorBoard
tensorboard --logdir logs/ --port 6006

# View specific experiment
tensorboard --logdir logs/rsl_rl/my_experiment/2025-01-08_22-31-59/

# Compare multiple experiments
tensorboard --logdir logs/rsl_rl/experiment1:logs/rsl_rl/experiment2
```

### Performance Profiling

```bash
# Profile environment performance
isaaclab.bat -p scripts\benchmarks\benchmark_non_rl.py \
    --task=Isaac-Ant-Direct-v0 \
    --num_envs=4096 \
    --headless

# Profile with specific RL library
isaaclab.bat -p scripts\benchmarks\benchmark_rsl_rl.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --headless
```

### Common Error Messages and Solutions

**"CUDA out of memory":**
```bash
# Reduce environments
--num_envs=1024

# Reduce batch size
agent.algorithm.num_mini_batches=2

# Use CPU simulation
scene.device=cpu
```

**"Task not found":**
```bash
# Check available tasks
isaaclab.bat -p scripts\environments\list_envs.py

# Verify project installation
pip list | findstr project_name

# Check task registration
python -c "import gymnasium as gym; print([env for env in gym.envs.registry.keys() if 'YourTask' in env])"
```

**"ImportError: No module named 'isaaclab'":**
```bash
# Reinstall Isaac Lab
isaaclab.bat -i

# Check Python path
python -c "import sys; print(sys.path)"

# Verify installation
python -c "import isaaclab; print(isaaclab.__version__)"
```

## Model Deployment

### Exporting Trained Models

```bash
# Export RSL-RL model
isaaclab.bat -p scripts\tools\pretrained_checkpoint.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --checkpoint=logs/rsl_rl/ant/model_1500.pt \
    --output=deployed_models/ant_model.pt

# Export RL-Games model
isaaclab.bat -p scripts\tools\pretrained_checkpoint.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rl_games_ppo_cfg \
    --checkpoint=logs/rl_games/ant/model.pth \
    --output=deployed_models/ant_model.pth
```

### Real-Time Inference

```bash
# Run in real-time mode
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --real-time \
    --num_envs=1

# Run with specific checkpoint
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --checkpoint=deployed_models/ant_model.pt \
    --real-time
```

### Integration with Real Robots

```python
# Example: Load trained model for real robot control
import torch
from isaaclab.envs import DirectRLEnv
from your_project.tasks.your_task_env import YourTaskEnv
from your_project.tasks.your_task_env_cfg import YourTaskEnvCfg

# Load environment
cfg = YourTaskEnvCfg()
env = YourTaskEnv(cfg)

# Load trained model
model = torch.load("deployed_models/your_model.pt")
model.eval()

# Real-time control loop
obs = env.reset()
while True:
    with torch.no_grad():
        action = model(obs["policy"])
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
```

## Environment Variables

### Isaac Lab Configuration

```bash
# Set Isaac Lab paths
export ISAACLAB_PATH="/path/to/isaaclab"
export ISAAC_NUCLEUS_DIR="/path/to/nucleus"

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export ISAAC_SIM_GPU_ID=0

# Performance tuning
export ISAAC_SIM_NUM_THREADS=8
export ISAAC_SIM_MEMORY_POOL_SIZE=1024
```

### Python Environment

```bash
# Set Python paths
export PYTHONPATH="${ISAACLAB_PATH}/source:${PYTHONPATH}"

# Isaac Sim configuration
export ISAAC_SIM_PATH="/path/to/isaac-sim"
export ISAAC_SIM_PYTHON_PATH="${ISAAC_SIM_PATH}/python"
```
