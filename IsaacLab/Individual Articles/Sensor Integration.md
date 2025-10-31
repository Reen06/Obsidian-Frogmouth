# 📡 **Sensor Integration - Complete Guide**

Isaac Lab provides comprehensive sensor support for robotics applications, including cameras, IMUs, contact sensors, and ray casting sensors.

## 🎯 **Available Sensor Types**

### **1. Camera Sensors**
- **RGB Cameras**: Color images
- **Depth Cameras**: Depth information
- **Segmentation Cameras**: Object segmentation
- **Multi-camera setups**: Multiple camera configurations

### **2. Contact Sensors**
- **Force/Torque sensors**: Contact forces and torques
- **Contact detection**: Binary contact information
- **Pressure sensors**: Pressure distribution

### **3. IMU Sensors**
- **Accelerometers**: Linear acceleration
- **Gyroscopes**: Angular velocity
- **Magnetometers**: Magnetic field (optional)

### **4. Ray Casting Sensors**
- **Lidar**: Distance measurements
- **Range sensors**: Proximity detection
- **Custom ray patterns**: Configurable ray arrangements

## 🔧 **Sensor Configuration**

### **Camera Configuration**

```python
from isaaclab.sensors import CameraCfg

# Basic RGB camera
camera = CameraCfg(
    prim_path="/World/robot/camera",
    update_period=0.1,  # Update every 0.1 seconds
    height=480,
    width=640,
    data_types=["rgb"]  # Only RGB data
)

# Multi-modal camera
multi_camera = CameraCfg(
    prim_path="/World/robot/camera",
    update_period=0.1,
    height=480,
    width=640,
    data_types=["rgb", "depth", "segmentation"]  # Multiple data types
)

# High-resolution camera
hd_camera = CameraCfg(
    prim_path="/World/robot/camera",
    update_period=0.05,  # Higher frequency
    height=1080,
    width=1920,
    data_types=["rgb", "depth"]
)
```

### **Contact Sensor Configuration**

```python
from isaaclab.sensors import ContactSensorCfg

# Basic contact sensor
contact_sensor = ContactSensorCfg(
    prim_path="/World/robot/feet_*",  # All feet
    update_period=0.0,  # Every simulation step
    history_length=2  # Keep 2 frames of history
)

# Force/torque sensor
ft_sensor = ContactSensorCfg(
    prim_path="/World/robot/end_effector",
    update_period=0.0,
    history_length=1,
    data_types=["force", "torque"]
)
```

### **IMU Configuration**

```python
from isaaclab.sensors import ImuCfg

# Basic IMU
imu = ImuCfg(
    prim_path="/World/robot/base",
    update_period=0.0,  # Every simulation step
    data_types=["accel", "gyro"]  # Accelerometer and gyroscope
)

# Full IMU with magnetometer
full_imu = ImuCfg(
    prim_path="/World/robot/base",
    update_period=0.0,
    data_types=["accel", "gyro", "mag"]  # All IMU data
)
```

### **Ray Caster Configuration**

```python
from isaaclab.sensors import RayCasterCfg

# Basic ray caster
ray_caster = RayCasterCfg(
    prim_path="/World/robot/base",
    update_period=0.1,
    max_distance=10.0,
    num_rays=16,
    pattern="circle"  # Circular pattern
)

# Lidar-style ray caster
lidar = RayCasterCfg(
    prim_path="/World/robot/base",
    update_period=0.05,
    max_distance=50.0,
    num_rays=360,
    pattern="circle",
    horizontal_fov=360.0,
    vertical_fov=30.0
)
```

## 🏗️ **Sensor Integration in Environment**

### **Scene Configuration**

```python
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ImuCfg, ContactSensorCfg
from isaaclab.utils import configclass

@configclass
class MyTaskSceneCfg(InteractiveSceneCfg):
    """Scene configuration with sensors."""
    
    # Robot with sensors
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(usd_path="/path/to/robot.usd"),
        sensors={
            "camera": CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/camera",
                height=480,
                width=640,
                data_types=["rgb", "depth"]
            ),
            "imu": ImuCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base",
                data_types=["accel", "gyro"]
            ),
            "contact": ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/feet_*",
                history_length=2
            )
        }
    )
```

### **Environment Implementation**

```python
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

class MyTaskEnv(DirectRLEnv):
    """Environment with sensor integration."""
    
    cfg: MyTaskEnvCfg
    
    def __init__(self, cfg: MyTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Access sensors
        self.camera = self.scene.sensors["camera"]
        self.imu = self.scene.sensors["imu"]
        self.contact = self.scene.sensors["contact"]
    
    def _get_observations(self) -> dict:
        """Get observations including sensor data."""
        
        # Basic robot state
        joint_pos = self.robot.data.joint_pos[:, self._joint_indices]
        joint_vel = self.robot.data.joint_vel[:, self._joint_indices]
        
        # IMU data
        imu_accel = self.imu.data.accel
        imu_gyro = self.imu.data.gyro
        
        # Contact data
        contact_forces = self.contact.data.force
        
        # Camera data
        rgb_images = self.camera.data.rgb
        
        # Combine observations
        obs = torch.cat([
            joint_pos,
            joint_vel,
            imu_accel.flatten(start_dim=1),
            imu_gyro.flatten(start_dim=1),
            contact_forces.flatten(start_dim=1),
        ], dim=-1)
        
        return {
            "policy": obs,
            "camera": rgb_images,  # Separate camera observation
        }
```

## 📊 **Sensor Data Access**

### **Camera Data**

```python
# Access camera data
camera_data = self.scene.sensors["camera"].data

# RGB images: [num_envs, height, width, 3]
rgb_images = camera_data.rgb

# Depth images: [num_envs, height, width, 1]
depth_images = camera_data.depth

# Segmentation images: [num_envs, height, width, 1]
seg_images = camera_data.segmentation

# Camera intrinsics
intrinsics = camera_data.intrinsics  # [num_envs, 3, 3]

# Camera extrinsics
extrinsics = camera_data.extrinsics  # [num_envs, 4, 4]
```

### **IMU Data**

```python
# Access IMU data
imu_data = self.scene.sensors["imu"].data

# Accelerometer: [num_envs, 3]
accel = imu_data.accel

# Gyroscope: [num_envs, 3]
gyro = imu_data.gyro

# Magnetometer: [num_envs, 3] (if enabled)
mag = imu_data.mag
```

### **Contact Sensor Data**

```python
# Access contact data
contact_data = self.scene.sensors["contact"].data

# Contact forces: [num_envs, num_contacts, 3]
forces = contact_data.force

# Contact torques: [num_envs, num_contacts, 3]
torques = contact_data.torque

# Contact positions: [num_envs, num_contacts, 3]
positions = contact_data.position

# Contact normals: [num_envs, num_contacts, 3]
normals = contact_data.normal
```

### **Ray Caster Data**

```python
# Access ray caster data
ray_data = self.scene.sensors["ray_caster"].data

# Ray hits: [num_envs, num_rays, 3]
hits = ray_data.hits

# Ray distances: [num_envs, num_rays]
distances = ray_data.distances

# Ray normals: [num_envs, num_rays, 3]
normals = ray_data.normals
```

## 🎮 **Vision-Based Tasks**

### **Training with Cameras**

```bash
# Enable cameras for vision tasks
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task=Isaac-Cartpole-RGB-Camera-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --enable_cameras \
    --num_envs=2048 \
    --headless
```

### **Playing with Cameras**

```bash
# Play vision-based task
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py \
    --task=Isaac-Cartpole-RGB-Camera-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --enable_cameras
```

## 🔧 **Custom Sensor Implementation**

### **Creating Custom Sensors**

```python
from isaaclab.sensors import SensorCfg
from isaaclab.utils import configclass

@configclass
class CustomSensorCfg(SensorCfg):
    """Configuration for custom sensor."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_types = ["custom_data"]
        self.update_period = 0.1

class CustomSensor:
    """Custom sensor implementation."""
    
    def __init__(self, cfg: CustomSensorCfg, env):
        self.cfg = cfg
        self.env = env
        self.data = CustomSensorData()
    
    def update(self, dt: float):
        """Update sensor data."""
        # Custom sensor logic here
        pass

class CustomSensorData:
    """Custom sensor data container."""
    
    def __init__(self):
        self.custom_data = None
```

## 📈 **Sensor Performance Optimization**

### **Update Frequency Guidelines**

```python
# High-frequency sensors (every step)
imu = ImuCfg(update_period=0.0)  # 120Hz
contact = ContactSensorCfg(update_period=0.0)  # 120Hz

# Medium-frequency sensors
ray_caster = RayCasterCfg(update_period=0.05)  # 20Hz

# Low-frequency sensors
camera = CameraCfg(update_period=0.1)  # 10Hz
```

### **Memory Management**

```python
# Limit history for memory efficiency
contact_sensor = ContactSensorCfg(
    prim_path="/World/robot/feet_*",
    history_length=1  # Only current frame
)

# Reduce image resolution for performance
camera = CameraCfg(
    prim_path="/World/robot/camera",
    height=240,  # Lower resolution
    width=320,
    data_types=["rgb"]  # Only RGB, no depth
)
```

## 🎯 **Sensor Integration Best Practices**

### **1. Start Simple**
- Begin with basic sensors (IMU, contact)
- Add cameras only when needed
- Use low resolution initially

### **2. Optimize Performance**
- Use appropriate update frequencies
- Limit history length
- Choose necessary data types only

### **3. Handle Missing Data**
```python
def _get_observations(self) -> dict:
    """Robust observation handling."""
    
    # Check if sensor data is available
    if hasattr(self, 'camera') and self.camera.data.rgb is not None:
        rgb_images = self.camera.data.rgb
    else:
        rgb_images = torch.zeros(self.num_envs, 480, 640, 3)
    
    return {"policy": obs, "camera": rgb_images}
```

### **4. Sensor Calibration**
- Calibrate camera intrinsics
- Align IMU coordinate frames
- Verify contact sensor positions

## 🚀 **Common Sensor Configurations**

### **Mobile Robot Sensors**
```python
mobile_robot_sensors = {
    "imu": ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base"),
    "lidar": RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        num_rays=360,
        max_distance=50.0
    ),
    "contact": ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wheels_*"
    )
}
```

### **Manipulator Sensors**
```python
manipulator_sensors = {
    "camera": CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera",
        data_types=["rgb", "depth"]
    ),
    "ft_sensor": ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/end_effector",
        data_types=["force", "torque"]
    ),
    "joint_encoders": JointPositionObservation(
        joint_names=[".*"]
    )
}
```

### **Humanoid Sensors**
```python
humanoid_sensors = {
    "imu": ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/torso"),
    "foot_contact": ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/feet_*"
    ),
    "hand_contact": ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/hands_*"
    )
}
```

Isaac Lab's sensor integration provides everything you need for sophisticated robotics applications with rich sensory feedback!
