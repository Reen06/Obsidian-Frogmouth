# 🤖 **Robot Integration - Complete Guide**

This guide covers everything you need to know about integrating custom robots into Isaac Lab, from URDF conversion to complete RL training setup.

## 🔄 **URDF to USD Conversion**

### **Command Line Conversion**

```bash
# Basic URDF to USD conversion
python scripts/tools/convert_urdf.py --input /path/to/your/robot.urdf --output /path/to/output/robot.usd

# With additional options
python scripts/tools/convert_urdf.py \
    --input /path/to/your/robot.urdf \
    --output /path/to/output/robot.usd \
    --joint_drive \
    --root_link_name="base_link" \
    --merge_fixed_joints
```

### **Programmatic Conversion**

```python
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

# Configure URDF conversion
urdf_cfg = UrdfConverterCfg(
    asset_path="/path/to/your/robot.urdf",
    usd_path="/path/to/output/robot.usd",
    joint_drive=True,  # Enable joint drives
    root_link_name="base_link",  # Specify root link
    merge_fixed_joints=True,  # Merge fixed joints for performance
    convex_decomposition=True,  # Better collision detection
    make_instanceable=True  # Enable instancing for performance
)

# Convert
converter = UrdfConverter(urdf_cfg)
converter.convert()
```

### **MJCF to USD Conversion**

```bash
# Convert MuJoCo XML to USD
python scripts/tools/convert_mjcf.py --input /path/to/robot.xml --output /path/to/robot.usd
```

## 🏗️ **Robot Configuration**

### **Basic Robot Configuration**

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
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=True,  # Enable contact sensors
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            # Add all your joint names and initial positions
        },
        joint_vel={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
        },
        pos=(0.0, 0.0, 0.0),  # Initial position
        quat=(0.0, 0.0, 0.0, 1.0),  # Initial orientation (w, x, y, z)
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # All joints
            effort_limit_sim=100.0,   # Torque limits
            velocity_limit_sim=10.0,  # Velocity limits
            stiffness=1000.0,         # Position control stiffness
            damping=100.0,            # Position control damping
            friction=0.1,             # Joint friction
        ),
    },
)
```

### **Advanced Robot Configuration**

```python
# Advanced robot with multiple actuator groups
ADVANCED_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/your/robot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            enable_gyroscopic_forces=True,
            enable_ccd=True,  # Continuous collision detection
        ),
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
            rest_offset=0.0,
            thickness=0.01,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "base_joint": 0.0,
            "shoulder_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_joint": 0.0,
        },
        joint_vel={
            "base_joint": 0.0,
            "shoulder_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.5),  # Elevated position
        quat=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=5.0,
            stiffness=2000.0,
            damping=200.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_joint", "elbow_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=1000.0,
            damping=100.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=15.0,
            stiffness=500.0,
            damping=50.0,
        ),
    },
)
```

## 🎯 **Actuator Types**

### **Implicit Actuator (Recommended)**
```python
from isaaclab.actuators import ImplicitActuatorCfg

implicit_actuator = ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit_sim=100.0,
    velocity_limit_sim=10.0,
    stiffness=1000.0,
    damping=100.0,
    friction=0.1,
    armature=0.01,
)
```

### **DC Motor Actuator**
```python
from isaaclab.actuators import DCMotorCfg

dc_motor = DCMotorCfg(
    joint_names_expr=[".*"],
    effort_limit_sim=100.0,
    velocity_limit_sim=10.0,
    motor_torque_constant=0.1,
    motor_back_emf_constant=0.1,
    motor_armature_resistance=1.0,
    motor_armature_inductance=0.01,
)
```

### **Servo Actuator**
```python
from isaaclab.actuators import ServoActuatorCfg

servo_actuator = ServoActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit_sim=100.0,
    velocity_limit_sim=10.0,
    position_gain=1000.0,
    velocity_gain=100.0,
    feedforward_gain=1.0,
)
```

## 🔧 **Robot Integration in Environment**

### **Scene Configuration**

```python
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class MyTaskSceneCfg(InteractiveSceneCfg):
    """Scene configuration with robot."""
    
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
```

### **Environment Implementation**

```python
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

class MyTaskEnv(DirectRLEnv):
    """Environment with custom robot."""
    
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
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
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

## 🎮 **Robot Control Methods**

### **Joint Torque Control**
```python
def _apply_action(self) -> None:
    """Apply joint torques."""
    self.robot.set_joint_effort_target(
        self.actions * self.cfg.action_scale,
        joint_ids=self._joint_indices
    )
```

### **Joint Position Control**
```python
def _apply_action(self) -> None:
    """Apply joint position targets."""
    joint_pos_targets = self.actions * self.cfg.max_joint_pos
    self.robot.set_joint_position_target(
        joint_pos_targets,
        joint_ids=self._joint_indices
    )
```

### **Joint Velocity Control**
```python
def _apply_action(self) -> None:
    """Apply joint velocity targets."""
    joint_vel_targets = self.actions * self.cfg.max_joint_vel
    self.robot.set_joint_velocity_target(
        joint_vel_targets,
        joint_ids=self._joint_indices
    )
```

### **Mixed Control**
```python
def _apply_action(self) -> None:
    """Apply mixed control (position + torque)."""
    # Position control for some joints
    pos_joints = self.actions[:, :3] * self.cfg.max_joint_pos
    self.robot.set_joint_position_target(
        pos_joints,
        joint_ids=self._joint_indices[:3]
    )
    
    # Torque control for other joints
    torque_joints = self.actions[:, 3:] * self.cfg.action_scale
    self.robot.set_joint_effort_target(
        torque_joints,
        joint_ids=self._joint_indices[3:]
    )
```

## 📊 **Robot Data Access**

### **Joint Data**
```python
# Joint positions: [num_envs, num_joints]
joint_pos = self.robot.data.joint_pos

# Joint velocities: [num_envs, num_joints]
joint_vel = self.robot.data.joint_vel

# Joint accelerations: [num_envs, num_joints]
joint_accel = self.robot.data.joint_acc

# Joint torques: [num_envs, num_joints]
joint_torques = self.robot.data.joint_effort
```

### **Root Data**
```python
# Root position: [num_envs, 3]
root_pos = self.robot.data.root_pos_w

# Root orientation (quaternion): [num_envs, 4]
root_quat = self.robot.data.root_quat_w

# Root linear velocity: [num_envs, 3]
root_lin_vel = self.robot.data.root_lin_vel_w

# Root angular velocity: [num_envs, 3]
root_ang_vel = self.robot.data.root_ang_vel_w
```

### **Link Data**
```python
# Link positions: [num_envs, num_links, 3]
link_pos = self.robot.data.link_pos_w

# Link orientations: [num_envs, num_links, 4]
link_quat = self.robot.data.link_quat_w

# Link velocities: [num_envs, num_links, 6] (linear + angular)
link_vel = self.robot.data.link_vel_w
```

## 🔧 **Robot Configuration Best Practices**

### **1. Joint Limits**
```python
# Set realistic joint limits
actuator = ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit_sim=100.0,  # Realistic torque limits
    velocity_limit_sim=10.0,  # Realistic velocity limits
    stiffness=1000.0,  # Appropriate stiffness
    damping=100.0,  # Appropriate damping
)
```

### **2. Collision Detection**
```python
# Enable proper collision detection
rigid_props = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=1,
)

collision_props = sim_utils.CollisionPropertiesCfg(
    collision_enabled=True,
    contact_offset=0.01,
    rest_offset=0.0,
)
```

### **3. Performance Optimization**
```python
# Use instancing for performance
spawn = sim_utils.UsdFileCfg(
    usd_path="/path/to/robot.usd",
    make_instanceable=True,  # Enable instancing
    convex_decomposition=True,  # Better collision performance
)
```

## 🚀 **Testing Your Robot**

### **Test Robot Loading**
```bash
# Test robot loading
isaaclab.bat -p scripts\benchmarks\benchmark_load_robot.py --robot_path=/path/to/robot.usd
```

### **Test with Random Actions**
```bash
# Test robot with random actions
isaaclab.bat -p scripts\environments\random_agent.py --task=Template-MyRobot-Direct-v0 --num_envs=64
```

### **Test with Zero Actions**
```bash
# Test robot with zero actions
isaaclab.bat -p scripts\environments\zero_agent.py --task=Template-MyRobot-Direct-v0 --num_envs=64
```

## 🎯 **Common Robot Types**

### **Mobile Robot**
```python
MOBILE_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="/path/to/mobile_robot.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),  # Slightly elevated
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*"],
            effort_limit_sim=50.0,
            velocity_limit_sim=20.0,
            stiffness=500.0,
            damping=50.0,
        ),
    },
)
```

### **Manipulator Arm**
```python
MANIPULATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="/path/to/manipulator.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Elevated position
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

### **Humanoid Robot**
```python
HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="/path/to/humanoid.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),  # Standing height
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["leg_.*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=15.0,
            stiffness=2000.0,
            damping=200.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["arm_.*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=1000.0,
            damping=100.0,
        ),
    },
)
```

## 💡 **Troubleshooting**

### **Common Issues**

1. **Robot Not Loading**
   - Check USD file path
   - Verify URDF conversion
   - Check joint names in configuration

2. **Physics Instability**
   - Reduce timestep
   - Increase solver iterations
   - Check joint limits

3. **Performance Issues**
   - Enable instancing
   - Use convex decomposition
   - Reduce number of environments

4. **Control Issues**
   - Check actuator configuration
   - Verify joint indices
   - Check action scaling

Isaac Lab's robot integration provides everything you need to bring your custom robots to life in simulation!
