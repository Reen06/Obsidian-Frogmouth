
# Complete Guide: From URDF to RL Training in Isaac Lab

## Step 1: Convert URDF to USD Format

**First, convert your Fusion 360 URDF to USD format:**

```bash
# Navigate to your Isaac Lab directory
cd /path/to/IsaacLab

# Convert URDF to USD
python scripts/tools/convert_urdf.py --input /path/to/your/robot.urdf --output /path/to/output/robot.usd
```

**Or programmatically:**
```python
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

# Configure URDF conversion
urdf_cfg = UrdfConverterCfg(
    asset_path="/path/to/your/robot.urdf",
    usd_path="/path/to/output/robot.usd",
    joint_drive=True,  # Enable joint drives
    root_link_name="base_link"  # Specify root link
)

# Convert
converter = UrdfConverter(urdf_cfg)
converter.convert()
```

## Step 2: Create Robot Configuration

**In your generated project, create a robot configuration file:**

```python
# source/your_project/your_project/assets/my_robot.py
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Define your robot configuration
MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/your/robot.usd",  # Path to your converted USD
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,  # Enable contact sensors if needed
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            # Add all your joint names and initial positions
        },
        pos=(0.0, 0.0, 0.0),  # Initial position
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # All joints
            effort_limit_sim=100.0,   # Torque limits
            velocity_limit_sim=10.0,  # Velocity limits
            stiffness=1000.0,         # Position control stiffness
            damping=100.0,            # Position control damping
        ),
    },
)
```

## Step 3: Configure Environment (Direct Workflow)

**Edit your environment configuration:**

```python
# source/your_project/your_project/tasks/direct_single-agent/my_task_env_cfg.py
import math
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ..assets.my_robot import MY_ROBOT_CFG

@configclass
class MyTaskSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    # Your robot
    robot: ArticulationCfg = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

@configclass
class MyTaskEnvCfg(DirectRLEnvCfg):
    """Configuration for the environment."""
    
    # Scene settings
    scene: MyTaskSceneCfg = MyTaskSceneCfg(num_envs=4096, env_spacing=2.0)
    
    # Basic settings
    decimation = 4  # Control frequency: 120Hz / 4 = 30Hz
    episode_length_s = 20.0  # Episode duration
    
    # Observation space (define what the agent can observe)
    observation_space = 8  # Number of observations (e.g., joint positions + velocities)
    
    # Action space (define what the agent can control)
    action_space = 6  # Number of actions (e.g., joint torques)
    
    # Simulation settings
    sim = SimulationCfg(
        dt=1/120,  # Physics timestep
        substeps=1,
        up_axis="z",
        gravity=(0.0, 0.0, -9.81),
    )
    
    # Viewer settings
    viewer = DirectRLEnvCfg.ViewerCfg(
        eye=(3.0, 3.0, 3.0),
        lookat=(0.0, 0.0, 0.0),
    )
```

## Step 4: Implement Environment Logic

**Edit your environment implementation:**

```python
# source/your_project/your_project/tasks/direct_single-agent/my_task_env.py
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .my_task_env_cfg import MyTaskEnvCfg

class MyTaskEnv(DirectRLEnv):
    """Environment for your custom robot task."""
    
    cfg: MyTaskEnvCfg
    
    def __init__(self, cfg: MyTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Get joint indices
        self._joint_indices, _ = self.robot.find_joints(".*")
        
        # Get joint data
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
    def _setup_scene(self):
        """Setup the simulation scene."""
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
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions before physics step."""
        self.actions = actions.clone()
    
    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        # Apply joint torques
        self.robot.set_joint_effort_target(
            self.actions * 100.0,  # Scale actions
            joint_ids=self._joint_indices
        )
    
    def _get_observations(self) -> dict:
        """Get observations."""
        # Combine joint positions and velocities
        obs = torch.cat([
            self.joint_pos[:, self._joint_indices],
            self.joint_vel[:, self._joint_indices],
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # Example reward: penalize large joint velocities
        joint_vel_penalty = -torch.sum(torch.abs(self.joint_vel[:, self._joint_indices]), dim=1)
        
        # Example reward: reward for staying upright (if applicable)
        # upright_reward = torch.exp(-torch.norm(self.robot.data.root_lin_vel_b[:, :2], dim=1))
        
        return joint_vel_penalty
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Example: terminate if robot falls
        # fallen = torch.any(torch.abs(self.robot.data.root_pos_b[:, 2]) < 0.1, dim=1)
        
        # For now, only time-based termination
        terminated = torch.zeros_like(time_out)
        
        return terminated, time_out
    
    def _reset_idx(self, env_ids):
        """Reset specific environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)
        
        # Reset joint positions with some randomization
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos += sample_uniform(
            -0.1, 0.1,  # Random offset range
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
```

## Step 5: Configure RL Algorithm

**Edit your agent configuration:**

```python
# source/your_project/your_project/tasks/agents/rsl_rl_ppo_cfg.py
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for PPO training."""
    
    num_steps_per_env = 16
    max_iterations = 1500
    save_interval = 50
    experiment_name = "my_robot_task"
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

## Step 6: Install and Test Your Project

**Install your project:**
```bash
cd /path/to/your/project
pip install -e source/your_project
```

**Test your environment:**
```bash
# List available environments
python scripts/list_envs.py

# Test with random actions
python scripts/random_agent.py --task=Template-My-Task-Direct-v0 --num_envs=64

# Test with zero actions
python scripts/zero_agent.py --task=Template-My-Task-Direct-v0 --num_envs=64
```

## Step 7: Start Training

**Train with RSL-RL:**
```bash
python scripts/rsl_rl/train.py --task=Template-My-Task-Direct-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096
```

**Train with RL-Games:**
```bash
python scripts/rl_games/train.py --task=Template-My-Task-Direct-v0 --agent=rl_games_ppo_cfg --num_envs=4096
```

**Train with SKRL:**
```bash
python scripts/skrl/train.py --task=Template-My-Task-Direct-v0 --agent=skrl_ppo_cfg --num_envs=4096
```

## Step 8: Monitor Training

**Training logs are stored in:**
```
logs/
├── rsl_rl/
│   └── my_robot_task/
│       └── 2024-01-15_10-30-45/
│           ├── model_1000.pt
│           ├── model_1050.pt
│           └── events.out.tfevents.*
```

**Monitor with TensorBoard:**
```bash
tensorboard --logdir logs/rsl_rl/my_robot_task/
```

## Step 9: Play Trained Model

**Test your trained model:**
```bash
python scripts/rsl_rl/play.py --task=Template-My-Task-Direct-v0 --agent=rsl_rl_ppo_cfg --checkpoint=logs/rsl_rl/my_robot_task/2024-01-15_10-30-45/model_1500.pt
```

## Key Configuration Points

**Observation Space:** Define what your robot can observe (joint positions, velocities, sensor data, etc.)

**Action Space:** Define what your robot can control (joint torques, velocities, positions)

**Reward Function:** Design rewards that encourage desired behavior (task completion, stability, efficiency)

**Termination Conditions:** Define when episodes should end (time limit, failure conditions, success)

**Environment Parameters:** Number of parallel environments, episode length, physics timestep

This workflow gives you complete control over your robot's behavior while leveraging Isaac Lab's powerful simulation and training infrastructure. The key is to start simple and gradually add complexity to your reward functions and observations as you refine your task.