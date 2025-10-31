# 🛠️ **Non-Training Scripts and Tools - Complete Guide**

Isaac Lab provides a comprehensive suite of non-training scripts and tools for demonstrations, testing, benchmarking, and utility operations.

## 🎮 **Demonstration Scripts**

### **Robot Demonstrations**

#### **Arms (Manipulators)**
```bash
# Single-arm manipulator demonstrations
isaaclab.bat -p scripts/demos/arms.py

# Specific arm demonstrations
isaaclab.bat -p scripts/demos/arms.py --robot=franka
isaaclab.bat -p scripts/demos/arms.py --robot=ur10
```

#### **Quadrupeds (Four-legged robots)**
```bash
# Quadruped locomotion demonstrations
isaaclab.bat -p scripts/demos/quadrupeds.py

# Specific quadruped demonstrations
isaaclab.bat -p scripts/demos/quadrupeds.py --robot=anymal_c
isaaclab.bat -p scripts/demos/quadrupeds.py --robot=spot
```

#### **Hands (Manipulation)**
```bash
# Hand manipulation demonstrations
isaaclab.bat -p scripts/demos/hands.py

# Specific hand demonstrations
isaaclab.bat -p scripts/demos/hands.py --robot=allegro_hand
isaaclab.bat -p scripts/demos/hands.py --robot=shadow_hand
```

#### **Bipeds (Two-legged robots)**
```bash
# Bipedal locomotion demonstrations
isaaclab.bat -p scripts/demos/bipeds.py

# Specific biped demonstrations
isaaclab.bat -p scripts/demos/bipeds.py --robot=humanoid
isaaclab.bat -p scripts/demos/bipeds.py --robot=h1
```

### **Sensor Demonstrations**

#### **Camera Demo**
```bash
# Camera sensor demonstration
isaaclab.bat -p scripts/demos/sensors/camera_demo.py

# RGB camera demo
isaaclab.bat -p scripts/demos/sensors/camera_demo.py --camera_type=rgb

# Depth camera demo
isaaclab.bat -p scripts/demos/sensors/camera_demo.py --camera_type=depth

# Multi-modal camera demo
isaaclab.bat -p scripts/demos/sensors/camera_demo.py --camera_type=multi
```

#### **IMU Demo**
```bash
# IMU sensor demonstration
isaaclab.bat -p scripts/demos/sensors/imu_demo.py

# With specific robot
isaaclab.bat -p scripts/demos/sensors/imu_demo.py --robot=anymal_c
```

#### **Contact Sensor Demo**
```bash
# Contact sensor demonstration
isaaclab.bat -p scripts/demos/sensors/contact_demo.py

# Force/torque sensor demo
isaaclab.bat -p scripts/demos/sensors/contact_demo.py --sensor_type=force_torque
```

## 🔍 **Environment Testing Scripts**

### **List Available Environments**
```bash
# List all available environments
isaaclab.bat -p scripts/environments/list_envs.py

# Filter by specific criteria
isaaclab.bat -p scripts/environments/list_envs.py | findstr "Direct"
isaaclab.bat -p scripts/environments/list_envs.py | findstr "Franka"
isaaclab.bat -p scripts/environments/list_envs.py | findstr "Camera"
```

### **Random Agent Testing**
```bash
# Test environment with random actions
isaaclab.bat -p scripts/environments/random_agent.py --task=Isaac-Cartpole-Direct-v0

# Test with specific number of environments
isaaclab.bat -p scripts/environments/random_agent.py --task=Isaac-Cartpole-Direct-v0 --num_envs=64

# Test with headless mode
isaaclab.bat -p scripts/environments/random_agent.py --task=Isaac-Cartpole-Direct-v0 --headless

# Test for specific duration
isaaclab.bat -p scripts/environments/random_agent.py --task=Isaac-Cartpole-Direct-v0 --num_steps=1000
```

### **Zero Agent Testing**
```bash
# Test environment with zero actions
isaaclab.bat -p scripts/environments/zero_agent.py --task=Isaac-Cartpole-Direct-v0

# Test with specific number of environments
isaaclab.bat -p scripts/environments/zero_agent.py --task=Isaac-Cartpole-Direct-v0 --num_envs=64

# Test with headless mode
isaaclab.bat -p scripts/environments/zero_agent.py --task=Isaac-Cartpole-Direct-v0 --headless
```

## 📊 **Performance Benchmarks**

### **Environment Performance Benchmark**
```bash
# Benchmark environment performance
isaaclab.bat -p scripts/benchmarks/benchmark_non_rl.py --task=Isaac-Ant-Direct-v0

# Benchmark with specific number of environments
isaaclab.bat -p scripts/benchmarks/benchmark_non_rl.py --task=Isaac-Ant-Direct-v0 --num_envs=4096

# Benchmark with headless mode
isaaclab.bat -p scripts/benchmarks/benchmark_non_rl.py --task=Isaac-Ant-Direct-v0 --headless

# Benchmark for specific duration
isaaclab.bat -p scripts/benchmarks/benchmark_non_rl.py --task=Isaac-Ant-Direct-v0 --num_steps=10000
```

### **Camera Performance Benchmark**
```bash
# Benchmark camera performance
isaaclab.bat -p scripts/benchmarks/benchmark_cameras.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0

# Benchmark with specific camera settings
isaaclab.bat -p scripts/benchmarks/benchmark_cameras.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --height=1080 --width=1920

# Benchmark with headless mode
isaaclab.bat -p scripts/benchmarks/benchmark_cameras.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --headless
```

### **Robot Loading Benchmark**
```bash
# Benchmark robot loading performance
isaaclab.bat -p scripts/benchmarks/benchmark_load_robot.py --robot_path=/path/to/robot.usd

# Benchmark with specific number of instances
isaaclab.bat -p scripts/benchmarks/benchmark_load_robot.py --robot_path=/path/to/robot.usd --num_instances=1000

# Benchmark with headless mode
isaaclab.bat -p scripts/benchmarks/benchmark_load_robot.py --robot_path=/path/to/robot.usd --headless
```

## 🔧 **Utility Tools**

### **Asset Conversion Tools**

#### **URDF to USD Conversion**
```bash
# Basic URDF to USD conversion
isaaclab.bat -p scripts/tools/convert_urdf.py --input=/path/to/robot.urdf --output=/path/to/robot.usd

# With additional options
isaaclab.bat -p scripts/tools/convert_urdf.py \
    --input=/path/to/robot.urdf \
    --output=/path/to/robot.usd \
    --joint_drive \
    --root_link_name="base_link" \
    --merge_fixed_joints \
    --convex_decomposition
```

#### **MJCF to USD Conversion**
```bash
# Convert MuJoCo XML to USD
isaaclab.bat -p scripts/tools/convert_mjcf.py --input=/path/to/robot.xml --output=/path/to/robot.usd

# With additional options
isaaclab.bat -p scripts/tools/convert_mjcf.py \
    --input=/path/to/robot.xml \
    --output=/path/to/robot.usd \
    --joint_drive \
    --merge_fixed_joints
```

### **Demonstration Tools**

#### **Record Demonstrations**
```bash
# Record demonstrations for imitation learning
isaaclab.bat -p scripts/tools/record_demos.py --task=Isaac-Reach-Franka-v0

# Record specific number of demonstrations
isaaclab.bat -p scripts/tools/record_demos.py --task=Isaac-Reach-Franka-v0 --num_demos=100

# Record to specific file
isaaclab.bat -p scripts/tools/record_demos.py --task=Isaac-Reach-Franka-v0 --output_file=demos/reach_franka.hdf5

# Record with specific duration
isaaclab.bat -p scripts/tools/record_demos.py --task=Isaac-Reach-Franka-v0 --demo_length=1000
```

#### **Replay Demonstrations**
```bash
# Replay recorded demonstrations
isaaclab.bat -p scripts/tools/replay_demos.py --dataset_file=demos/reach_franka.hdf5

# Replay specific demonstration
isaaclab.bat -p scripts/tools/replay_demos.py --dataset_file=demos/reach_franka.hdf5 --demo_index=0

# Replay with visualization
isaaclab.bat -p scripts/tools/replay_demos.py --dataset_file=demos/reach_franka.hdf5 --visualize
```

### **Checkpoint Management**

#### **Pre-trained Checkpoint Tool**
```bash
# Download pre-trained checkpoint
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py --train=rl_games:Isaac-Cartpole-v0

# Download specific checkpoint
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py --train=rsl_rl:Isaac-Ant-Direct-v0 --checkpoint=model_1500.pt

# List available checkpoints
isaaclab.bat -p scripts/tools/pretrained_checkpoint.py --list
```

## 📚 **Learning Tutorials**

### **Simulation Basics**
```bash
# Basic simulation tutorial
isaaclab.bat -p scripts/tutorials/00_sim/sim_01_basic.py

# Advanced simulation features
isaaclab.bat -p scripts/tutorials/00_sim/sim_02_advanced.py
```

### **Asset Management**
```bash
# Basic asset tutorial
isaaclab.bat -p scripts/tutorials/01_assets/assets_01_articulation.py

# Advanced asset features
isaaclab.bat -p scripts/tutorials/01_assets/assets_02_rigid_object.py
```

### **Scene Setup**
```bash
# Basic scene tutorial
isaaclab.bat -p scripts/tutorials/02_scene/scene_01_basic.py

# Advanced scene features
isaaclab.bat -p scripts/tutorials/02_scene/scene_02_advanced.py
```

### **Environment Creation**
```bash
# Basic environment tutorial
isaaclab.bat -p scripts/tutorials/03_envs/env_01_basic.py

# Advanced environment features
isaaclab.bat -p scripts/tutorials/03_envs/env_02_advanced.py
```

### **Sensor Integration**
```bash
# Basic sensor tutorial
isaaclab.bat -p scripts/tutorials/04_sensors/sensors_01_camera.py

# Advanced sensor features
isaaclab.bat -p scripts/tutorials/04_sensors/sensors_02_imu.py
```

## 🎯 **Common Use Cases**

### **Testing New Environment**
```bash
# 1. List available environments
isaaclab.bat -p scripts/environments/list_envs.py

# 2. Test with random actions
isaaclab.bat -p scripts/environments/random_agent.py --task=Your-Task-Name-v0

# 3. Test with zero actions
isaaclab.bat -p scripts/environments/zero_agent.py --task=Your-Task-Name-v0

# 4. Benchmark performance
isaaclab.bat -p scripts/benchmarks/benchmark_non_rl.py --task=Your-Task-Name-v0
```

### **Converting Robot Assets**
```bash
# 1. Convert URDF to USD
isaaclab.bat -p scripts/tools/convert_urdf.py --input=robot.urdf --output=robot.usd

# 2. Test robot loading
isaaclab.bat -p scripts/benchmarks/benchmark_load_robot.py --robot_path=robot.usd

# 3. Test in environment
isaaclab.bat -p scripts/environments/random_agent.py --task=Your-Robot-Task-v0
```

### **Recording Demonstrations**
```bash
# 1. Record demonstrations
isaaclab.bat -p scripts/tools/record_demos.py --task=Isaac-Reach-Franka-v0 --num_demos=50

# 2. Replay demonstrations
isaaclab.bat -p scripts/tools/replay_demos.py --dataset_file=demos/reach_franka.hdf5

# 3. Use for imitation learning
isaaclab.bat -p scripts/imitation_learning/isaaclab_mimic/train.py --demo_path=demos/reach_franka.hdf5
```

## 💡 **Best Practices**

### **1. Testing Workflow**
- Always test with random and zero agents first
- Use benchmarks to verify performance
- Test with different numbers of environments

### **2. Asset Conversion**
- Use appropriate conversion options
- Test converted assets thoroughly
- Keep original files as backup

### **3. Demonstration Recording**
- Record multiple demonstrations for robustness
- Use consistent recording parameters
- Validate demonstrations before training

### **4. Performance Optimization**
- Use headless mode for benchmarks
- Test with realistic environment counts
- Monitor memory usage during testing

## 🚀 **Quick Reference Commands**

```bash
# List all environments
isaaclab.bat -p scripts/environments/list_envs.py

# Test any environment
isaaclab.bat -p scripts/environments/random_agent.py --task=<TASK_NAME>

# Benchmark any environment
isaaclab.bat -p scripts/benchmarks/benchmark_non_rl.py --task=<TASK_NAME>

# Convert URDF to USD
isaaclab.bat -p scripts/tools/convert_urdf.py --input=<INPUT> --output=<OUTPUT>

# Record demonstrations
isaaclab.bat -p scripts/tools/record_demos.py --task=<TASK_NAME>

# Run tutorials
isaaclab.bat -p scripts/tutorials/00_sim/sim_01_basic.py
```

These non-training scripts and tools provide everything you need for testing, benchmarking, and utility operations in Isaac Lab!
