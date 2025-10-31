# 🏗️ **Manager-Based Workflow - Complete Guide**

The Manager-Based Workflow is Isaac Lab's modular approach to environment design, using specialized managers to handle different aspects of the RL environment.

## 🎯 **What is Manager-Based Workflow?**

The Manager-Based Workflow uses a system of managers to handle different aspects of the environment:
- **ActionManager**: Handles action processing and application
- **ObservationManager**: Manages observation computation and normalization
- **RewardManager**: Computes reward terms and episodic tracking
- **TerminationManager**: Handles termination conditions and timeouts
- **EventManager**: Manages randomization and curriculum events
- **CurriculumManager**: Handles curriculum learning and difficulty progression

## 🔧 **Manager-Based vs Direct Workflow**

| Feature | Manager-Based | Direct |
|---------|---------------|--------|
| **Complexity** | Modular, easier to extend | Single-file implementation |
| **Multi-Agent** | ❌ Single-agent only | ✅ Supports multi-agent |
| **Observation Spaces** | Box spaces only | Box, Dict, Composite spaces |
| **Entry Point** | `isaaclab.envs:ManagerBasedRLEnv` | Direct class reference |
| **File Structure** | `[task]_env_cfg.py` + `mdp/` folder | `[task]_env.py` + `[task]_env_cfg.py` |
| **Learning Curve** | Steeper (more concepts) | Gentler (familiar RL pattern) |

## 🏗️ **Manager-Based Implementation**

### **1. Environment Configuration**

```python
# source/your_project/your_project/tasks/manager_based_single-agent/my_task_env_cfg.py
from isaaclab.managers import (
    ActionTermCfg, ObservationTermCfg, RewardTermCfg,
    TerminationTermCfg, EventTermCfg, CurriculumTermCfg
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class MyTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for manager-based environment."""
    
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
    
    # Curriculum management
    curriculum = CurriculumTermCfg(
        func=DifficultyCurriculum,
        params={"difficulty_range": [0.1, 1.0]}
    )
```

### **2. MDP (Markov Decision Process) Functions**

The `mdp/` folder contains the actual implementation of each manager:

```python
# source/your_project/your_project/tasks/manager_based_single-agent/my_task/mdp/actions.py
import torch
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass

@configclass
class JointPositionAction(ActionTerm):
    """Action term for joint position control."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.scale = cfg.params["scale"]
        self.joint_names = cfg.params["joint_names"]
        
        # Get joint indices
        self._joint_indices, _ = self.env.robot.find_joints(self.joint_names)
    
    def __call__(self, env, actions: torch.Tensor) -> torch.Tensor:
        """Apply joint position actions."""
        # Scale actions
        scaled_actions = actions * self.scale
        
        # Apply joint position targets
        self.env.robot.set_joint_position_target(
            scaled_actions,
            joint_ids=self._joint_indices
        )
        
        return actions
```

```python
# source/your_project/your_project/tasks/manager_based_single-agent/my_task/mdp/observations.py
import torch
from isaaclab.managers import ObservationTerm
from isaaclab.utils import configclass

@configclass
class JointPositionObservation(ObservationTerm):
    """Observation term for joint positions."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_names = cfg.params["joint_names"]
        
        # Get joint indices
        self._joint_indices, _ = self.env.robot.find_joints(self.joint_names)
    
    def __call__(self, env, actions: torch.Tensor) -> torch.Tensor:
        """Get joint position observations."""
        joint_pos = self.env.robot.data.joint_pos[:, self._joint_indices]
        return joint_pos
```

```python
# source/your_project/your_project/tasks/manager_based_single-agent/my_task/mdp/rewards.py
import torch
from isaaclab.managers import RewardTerm
from isaaclab.utils import configclass

@configclass
class JointPositionReward(RewardTerm):
    """Reward term for joint position control."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.weight = cfg.params["weight"]
        self.joint_names = cfg.params["joint_names"]
        
        # Get joint indices
        self._joint_indices, _ = self.env.robot.find_joints(self.joint_names)
    
    def __call__(self, env, actions: torch.Tensor) -> torch.Tensor:
        """Compute joint position reward."""
        joint_pos = self.env.robot.data.joint_pos[:, self._joint_indices]
        joint_pos_penalty = torch.sum(torch.abs(joint_pos), dim=1)
        return self.weight * joint_pos_penalty
```

```python
# source/your_project/your_project/tasks/manager_based_single-agent/my_task/mdp/terminations.py
import torch
from isaaclab.managers import TerminationTerm
from isaaclab.utils import configclass

@configclass
class JointPositionTermination(TerminationTerm):
    """Termination term for joint position limits."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.threshold = cfg.params["threshold"]
        self.joint_names = cfg.params["joint_names"]
        
        # Get joint indices
        self._joint_indices, _ = self.env.robot.find_joints(self.joint_names)
    
    def __call__(self, env, actions: torch.Tensor) -> torch.Tensor:
        """Check joint position termination."""
        joint_pos = self.env.robot.data.joint_pos[:, self._joint_indices]
        exceeded = torch.any(torch.abs(joint_pos) > self.threshold, dim=1)
        return exceeded
```

```python
# source/your_project/your_project/tasks/manager_based_single-agent/my_task/mdp/events.py
import torch
from isaaclab.managers import EventTerm
from isaaclab.utils import configclass

@configclass
class RandomizeJointProperties(EventTerm):
    """Event term for joint property randomization."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_names = cfg.params["joint_names"]
        self.mode = cfg.params["mode"]
        
        # Get joint indices
        self._joint_indices, _ = self.env.robot.find_joints(self.joint_names)
    
    def __call__(self, env, actions: torch.Tensor) -> torch.Tensor:
        """Randomize joint properties."""
        if self.mode == "reset":
            # Randomize joint positions on reset
            joint_pos = self.env.robot.data.default_joint_pos
            joint_pos += torch.randn_like(joint_pos) * 0.1
            self.env.robot.write_joint_state_to_sim(joint_pos, None, None)
        
        return torch.zeros(self.env.num_envs, device=self.env.device)
```

### **3. Task Registration**

```python
# source/your_project/your_project/tasks/manager_based_single-agent/my_task/__init__.py
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from .my_task_env_cfg import MyTaskEnvCfg

gym.register(
    id="Template-MyTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg": MyTaskEnvCfg()},
)
```

## 🎯 **Manager Types and Usage**

### **ActionManager**
Handles action processing and application:
- **JointPositionAction**: Position control
- **JointEffortAction**: Torque control
- **JointVelocityAction**: Velocity control
- **CartesianAction**: Cartesian space control

### **ObservationManager**
Manages observation computation and normalization:
- **JointPositionObservation**: Joint positions
- **JointVelocityObservation**: Joint velocities
- **RootPoseObservation**: Root pose (position + orientation)
- **RootVelocityObservation**: Root velocities
- **SensorObservation**: Sensor data (IMU, cameras, etc.)

### **RewardManager**
Computes reward terms and episodic tracking:
- **JointPositionReward**: Joint position penalties
- **JointVelocityReward**: Joint velocity penalties
- **ActionReward**: Action penalties
- **ProgressReward**: Task progress rewards
- **SurvivalReward**: Survival bonuses

### **TerminationManager**
Handles termination conditions and timeouts:
- **JointPositionTermination**: Joint limit violations
- **JointVelocityTermination**: Velocity limit violations
- **RootPoseTermination**: Pose violations
- **TimeoutTermination**: Episode timeouts

### **EventManager**
Manages randomization and curriculum events:
- **RandomizeJointProperties**: Joint property randomization
- **RandomizeRigidBodyProperties**: Body property randomization
- **RandomizeSceneProperties**: Scene property randomization
- **ResetJointStates**: Joint state resets

### **CurriculumManager**
Handles curriculum learning and difficulty progression:
- **DifficultyCurriculum**: Difficulty-based curriculum
- **SuccessCurriculum**: Success-based curriculum
- **PerformanceCurriculum**: Performance-based curriculum

## 🚀 **Training Manager-Based Tasks**

### **Basic Training:**
```bash
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Template-MyTask-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --headless
```

### **Playing:**
```bash
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Template-MyTask-v0 --agent=rsl_rl_ppo_cfg
```

## 💡 **When to Use Manager-Based Workflow**

### **Choose Manager-Based When:**
- ✅ **Modular design** is important
- ✅ **Easy extension** of functionality
- ✅ **Single-agent** environments only
- ✅ **Box observation spaces** are sufficient
- ✅ **Complex reward systems** with many components
- ✅ **Research requiring** modular components

### **Choose Direct Workflow When:**
- ✅ **Multi-agent** environments needed
- ✅ **Complex observation spaces** (Dict, Composite)
- ✅ **Familiar RL pattern** preferred
- ✅ **Single-file implementation** desired
- ✅ **Maximum flexibility** needed

## 🔧 **Manager-Based Project Structure**

```
your_project/
├── source/
│   └── your_project/
│       └── tasks/
│           └── manager_based_single-agent/
│               └── my_task/
│                   ├── __init__.py
│                   ├── my_task_env_cfg.py
│                   └── mdp/
│                       ├── actions.py
│                       ├── observations.py
│                       ├── rewards.py
│                       ├── terminations.py
│                       ├── events.py
│                       └── curriculum.py
```

## 🎯 **Key Benefits of Manager-Based Workflow**

1. **Modularity**: Each component is separate and reusable
2. **Extensibility**: Easy to add new managers or modify existing ones
3. **Maintainability**: Clear separation of concerns
4. **Research-Friendly**: Easy to experiment with different components
5. **Documentation**: Each manager is well-documented and tested

The Manager-Based Workflow is perfect for complex environments where you need fine-grained control over each aspect of the RL environment!
