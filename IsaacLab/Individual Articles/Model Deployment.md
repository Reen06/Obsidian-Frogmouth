# 🚀 **Model Deployment - Complete Guide**

This guide covers everything you need to know about deploying trained Isaac Lab models for real-world applications, including model export, real-time inference, and integration with real robots.

## 📦 **Exporting Trained Models**

### **RSL-RL Model Export**
```bash
# Export RSL-RL model
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --checkpoint=logs/rsl_rl/ant/model_1500.pt \
    --output=deployed_models/ant_model.pt

# Export with specific parameters
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --checkpoint=logs/rsl_rl/ant/model_1500.pt \
    --output=deployed_models/ant_model.pt \
    --format=torch \
    --include_config=True
```

### **RL-Games Model Export**
```bash
# Export RL-Games model
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rl_games_ppo_cfg \
    --checkpoint=logs/rl_games/ant/model.pth \
    --output=deployed_models/ant_model.pth

# Export with configuration
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rl_games_ppo_cfg \
    --checkpoint=logs/rl_games/ant/model.pth \
    --output=deployed_models/ant_model.pth \
    --include_config=True \
    --include_env_config=True
```

### **SKRL Model Export**
```bash
# Export SKRL model
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=skrl_ppo_cfg \
    --checkpoint=logs/skrl/ant/agent_1500.pt \
    --output=deployed_models/ant_model.pt

# Export with metadata
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=skrl_ppo_cfg \
    --checkpoint=logs/skrl/ant/agent_1500.pt \
    --output=deployed_models/ant_model.pt \
    --include_metadata=True
```

### **SB3 Model Export**
```bash
# Export SB3 model
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=sb3_ppo_cfg \
    --checkpoint=logs/sb3/ant/model.zip \
    --output=deployed_models/ant_model.zip

# Export with normalization
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=sb3_ppo_cfg \
    --checkpoint=logs/sb3/ant/model.zip \
    --output=deployed_models/ant_model.zip \
    --include_normalization=True
```

## 🎮 **Real-Time Inference**

### **Real-Time Playing**
```bash
# Run in real-time mode
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --real-time \
    --num_envs=1

# Real-time with specific checkpoint
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --checkpoint=deployed_models/ant_model.pt \
    --real-time \
    --num_envs=1
```

### **Real-Time with Video Recording**
```bash
# Real-time with video recording
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --real-time \
    --video \
    --video_length=1000 \
    --num_envs=1
```

## 🤖 **Integration with Real Robots**

### **Model Loading for Real Robot Control**
```python
# Example: Load trained model for real robot control
import torch
import numpy as np
from isaaclab.envs import DirectRLEnv
from your_project.tasks.your_task_env import YourTaskEnv
from your_project.tasks.your_task_env_cfg import YourTaskEnvCfg

class RealRobotController:
    """Real robot controller using trained model."""
    
    def __init__(self, model_path: str, config_path: str):
        """Initialize controller."""
        
        # Load environment configuration
        self.cfg = YourTaskEnvCfg()
        
        # Load trained model
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        
        # Initialize robot interface
        self.robot_interface = self._init_robot_interface()
        
        # Initialize observation buffer
        self.obs_buffer = None
        
    def _init_robot_interface(self):
        """Initialize robot interface."""
        # Initialize your robot interface here
        # This could be ROS, direct hardware interface, etc.
        pass
    
    def get_observations(self) -> torch.Tensor:
        """Get observations from real robot."""
        
        # Get sensor data from real robot
        joint_pos = self.robot_interface.get_joint_positions()
        joint_vel = self.robot_interface.get_joint_velocities()
        imu_data = self.robot_interface.get_imu_data()
        
        # Convert to torch tensors
        obs = torch.tensor([
            joint_pos,
            joint_vel,
            imu_data
        ], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        return obs
    
    def apply_actions(self, actions: torch.Tensor) -> None:
        """Apply actions to real robot."""
        
        # Convert actions to robot commands
        joint_torques = actions.squeeze(0).numpy()
        
        # Apply to real robot
        self.robot_interface.set_joint_torques(joint_torques)
    
    def control_loop(self):
        """Main control loop."""
        
        while True:
            # Get observations
            obs = self.get_observations()
            
            # Compute actions
            with torch.no_grad():
                actions = self.model(obs)
            
            # Apply actions
            self.apply_actions(actions)
            
            # Control frequency
            time.sleep(0.01)  # 100Hz control loop
```

### **ROS Integration**
```python
# ROS integration example
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import torch

class ROSRobotController:
    """ROS-based robot controller."""
    
    def __init__(self, model_path: str):
        """Initialize ROS controller."""
        
        # Initialize ROS node
        rospy.init_node('isaac_lab_controller')
        
        # Load trained model
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        
        # ROS subscribers and publishers
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.cmd_pub = rospy.Publisher('/joint_commands', Float32MultiArray, queue_size=1)
        
        # Data buffers
        self.joint_pos = None
        self.joint_vel = None
        self.imu_data = None
        
    def joint_callback(self, msg):
        """Joint state callback."""
        self.joint_pos = np.array(msg.position)
        self.joint_vel = np.array(msg.velocity)
    
    def imu_callback(self, msg):
        """IMU callback."""
        self.imu_data = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
    
    def control_loop(self):
        """Main control loop."""
        
        rate = rospy.Rate(100)  # 100Hz
        
        while not rospy.is_shutdown():
            if self.joint_pos is not None and self.imu_data is not None:
                # Prepare observations
                obs = torch.tensor([
                    self.joint_pos,
                    self.joint_vel,
                    self.imu_data
                ], dtype=torch.float32).unsqueeze(0)
                
                # Compute actions
                with torch.no_grad():
                    actions = self.model(obs)
                
                # Publish commands
                cmd_msg = Float32MultiArray()
                cmd_msg.data = actions.squeeze(0).numpy().tolist()
                self.cmd_pub.publish(cmd_msg)
            
            rate.sleep()
```

## 🔧 **Model Optimization for Deployment**

### **Model Quantization**
```python
# Model quantization for deployment
import torch.quantization as quantization

def quantize_model(model_path: str, output_path: str):
    """Quantize model for deployment."""
    
    # Load model
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # Set quantization configuration
    quantization_config = quantization.QConfig(
        activation=quantization.observer.MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        ),
        weight=quantization.observer.MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
    )
    
    # Prepare model for quantization
    model.qconfig = quantization_config
    model_prepared = quantization.prepare(model)
    
    # Calibrate model
    # Run inference on calibration data
    for _ in range(100):
        dummy_input = torch.randn(1, obs_dim)
        model_prepared(dummy_input)
    
    # Convert to quantized model
    quantized_model = quantization.convert(model_prepared)
    
    # Save quantized model
    torch.save(quantized_model, output_path)
```

### **Model Pruning**
```python
# Model pruning for deployment
import torch.nn.utils.prune as prune

def prune_model(model_path: str, output_path: str, pruning_ratio: float = 0.2):
    """Prune model for deployment."""
    
    # Load model
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # Prune model
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')
    
    # Save pruned model
    torch.save(model, output_path)
```

### **Model Compilation**
```python
# Model compilation for deployment
import torch.jit

def compile_model(model_path: str, output_path: str):
    """Compile model for deployment."""
    
    # Load model
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, obs_dim)
    
    # Trace model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save compiled model
    traced_model.save(output_path)
```

## 📊 **Deployment Monitoring**

### **Performance Monitoring**
```python
# Performance monitoring for deployment
import time
import psutil
import GPUtil

class DeploymentMonitor:
    """Monitor deployment performance."""
    
    def __init__(self):
        """Initialize monitor."""
        self.start_time = time.time()
        self.step_count = 0
        self.total_inference_time = 0
        
    def log_step(self, inference_time: float):
        """Log step performance."""
        self.step_count += 1
        self.total_inference_time += inference_time
        
        # Log every 100 steps
        if self.step_count % 100 == 0:
            avg_inference_time = self.total_inference_time / self.step_count
            fps = 1.0 / avg_inference_time
            
            print(f"Step {self.step_count}:")
            print(f"  Average inference time: {avg_inference_time:.4f}s")
            print(f"  FPS: {fps:.2f}")
            print(f"  CPU usage: {psutil.cpu_percent():.1f}%")
            print(f"  Memory usage: {psutil.virtual_memory().percent:.1f}%")
            
            # GPU usage if available
            if GPUtil.getGPUs():
                gpu = GPUtil.getGPUs()[0]
                print(f"  GPU usage: {gpu.load * 100:.1f}%")
                print(f"  GPU memory: {gpu.memoryUsed / gpu.memoryTotal * 100:.1f}%")
```

### **Error Handling**
```python
# Error handling for deployment
import logging
import traceback

class DeploymentErrorHandler:
    """Handle deployment errors."""
    
    def __init__(self):
        """Initialize error handler."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def handle_inference_error(self, error: Exception, obs: torch.Tensor):
        """Handle inference errors."""
        self.logger.error(f"Inference error: {error}")
        self.logger.error(f"Observation shape: {obs.shape}")
        self.logger.error(f"Observation range: [{obs.min():.4f}, {obs.max():.4f}]")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return safe default action
        return torch.zeros(obs.shape[0], action_dim)
    
    def handle_robot_error(self, error: Exception):
        """Handle robot communication errors."""
        self.logger.error(f"Robot error: {error}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Implement safety measures
        self.emergency_stop()
    
    def emergency_stop(self):
        """Emergency stop procedure."""
        self.logger.critical("Emergency stop activated")
        # Implement emergency stop logic
        pass
```

## 🚀 **Deployment Best Practices**

### **1. Model Validation**
```python
# Validate model before deployment
def validate_model(model_path: str, test_data: torch.Tensor):
    """Validate model before deployment."""
    
    # Load model
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # Test inference
    with torch.no_grad():
        for i in range(100):
            obs = test_data[i:i+1]
            action = model(obs)
            
            # Check for NaN or Inf
            if torch.isnan(action).any() or torch.isinf(action).any():
                raise ValueError("Model produces NaN or Inf values")
            
            # Check action bounds
            if torch.abs(action).max() > 10.0:
                raise ValueError("Model produces actions outside expected range")
    
    print("Model validation passed")
```

### **2. Safety Measures**
```python
# Safety measures for deployment
class SafetyController:
    """Safety controller for deployment."""
    
    def __init__(self, max_action: float = 1.0, max_velocity: float = 10.0):
        """Initialize safety controller."""
        self.max_action = max_action
        self.max_velocity = max_velocity
        self.action_history = []
        
    def check_action_safety(self, action: torch.Tensor) -> torch.Tensor:
        """Check and limit action safety."""
        
        # Limit action magnitude
        action = torch.clamp(action, -self.max_action, self.max_action)
        
        # Check for sudden changes
        if len(self.action_history) > 0:
            prev_action = self.action_history[-1]
            action_diff = torch.abs(action - prev_action)
            if action_diff.max() > 0.5:  # Sudden change threshold
                action = prev_action + torch.sign(action - prev_action) * 0.5
        
        # Store action history
        self.action_history.append(action.clone())
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        
        return action
```

### **3. Configuration Management**
```python
# Configuration management for deployment
import json
from dataclasses import dataclass

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    
    model_path: str
    obs_dim: int
    action_dim: int
    control_frequency: float = 100.0
    max_action: float = 1.0
    safety_enabled: bool = True
    logging_enabled: bool = True
    
    def save(self, path: str):
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
```

## 💡 **Deployment Checklist**

### **Pre-Deployment**
- [ ] Model validation completed
- [ ] Performance benchmarking done
- [ ] Safety measures implemented
- [ ] Error handling configured
- [ ] Configuration files prepared
- [ ] Monitoring system set up

### **Deployment**
- [ ] Model loaded successfully
- [ ] Real-time inference working
- [ ] Robot communication established
- [ ] Safety systems active
- [ ] Monitoring active
- [ ] Error handling tested

### **Post-Deployment**
- [ ] Performance monitoring
- [ ] Error logging
- [ ] Safety system validation
- [ ] User feedback collection
- [ ] Performance optimization
- [ ] Model updates planned

## 🚀 **Quick Deployment Commands**

```bash
# Export model
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py --task=<TASK> --checkpoint=<CHECKPOINT> --output=<OUTPUT>

# Real-time inference
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task=<TASK> --agent=<AGENT> --real-time

# Test deployment
python test_deployment.py --model_path=<MODEL> --test_data=<DATA>

# Monitor deployment
python monitor_deployment.py --config=<CONFIG>
```

Isaac Lab's model deployment capabilities provide everything you need to bring your trained models to real-world applications!
