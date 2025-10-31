# 🎮 **Isaac Lab Demo Programs - Quick Reference**

Here are my favorite most commonly used demo programs with proper command examples:

python scripts/reinforcement_learning/skrl/play.py --task=Isaac-Repose-Cube-Allegro-Direct-v0 --rendering_mode=performance --num_envs=2

python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Ant-v0 --rendering_mode=performance --num_envs=2048

## 🤖 **Locomotion Tasks**

### **Ant Locomotion (Manager-based)**
```bash
# Training with GUI
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Ant-v0 --agent=rsl_rl_ppo_cfg --num_envs=2048

# Training headless (faster)
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Ant-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --headless

# Playing trained model
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Ant-v0 --agent=rsl_rl_ppo_cfg
```

### **Ant Locomotion (Direct workflow)**
```bash
# Training
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Ant-Direct-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --headless

# Playing
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Ant-Direct-v0 --agent=rsl_rl_ppo_cfg
```

## 🎯 **Classic Control Tasks**

### **Cartpole Balancing**
```bash
# Manager-based workflow
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Cartpole-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --headless

# Direct workflow
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Cartpole-Direct-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --headless

# Playing
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Cartpole-Direct-v0 --agent=rsl_rl_ppo_cfg
```

### **Cartpole with RGB Camera**
```bash
# Training with camera
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --agent=rsl_rl_ppo_cfg --enable_cameras --num_envs=2048 --headless

# Playing with camera
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --agent=rsl_rl_ppo_cfg --enable_cameras
```

## 🦾 **Manipulation Tasks**

### **Franka Reach Task**
```bash
# Training
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Reach-Franka-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --headless

# Playing
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Reach-Franka-v0 --agent=rsl_rl_ppo_cfg
```

### **Franka Lift Task**
```bash
# Training
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Lift-Franka-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --headless

# Playing
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Lift-Franka-v0 --agent=rsl_rl_ppo_cfg
```

## 🐕 **Quadruped Locomotion**

### **ANYmal-C Flat Terrain**
```bash
# Training
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Velocity-Flat-Anymal-C-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --headless

# Playing
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Velocity-Flat-Anymal-C-v0 --agent=rsl_rl_ppo_cfg
```

### **ANYmal-C Rough Terrain**
```bash
# Training
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --headless

# Playing
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --agent=rsl_rl_ppo_cfg
```

## 🤖 **Humanoid Tasks**

### **Humanoid Locomotion**
```bash
# Training
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Humanoid-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --headless

# Playing
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Humanoid-v0 --agent=rsl_rl_ppo_cfg
```

## 🎮 **Hand Manipulation**

### **Allegro Hand In-Hand Manipulation**
```bash
# Training
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Inhand-Allegro-Hand-v0 --agent=rsl_rl_ppo_cfg --num_envs=2048 --headless

# Playing
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Inhand-Allegro-Hand-v0 --agent=rsl_rl_ppo_cfg
```

## 🔧 **Common Training Flags**

### **Essential Flags:**
- `--task`: Task name (e.g., `Isaac-Ant-v0`, `Isaac-Cartpole-Direct-v0`)
- `--agent`: Agent configuration (e.g., `rsl_rl_ppo_cfg`, `rl_games_ppo_cfg`)
- `--num_envs`: Number of parallel environments (2048-4096 for training)
- `--headless`: Run without GUI for faster training
- `--max_iterations`: Maximum training iterations (default: 1500)

### **Optional Flags:**
- `--seed`: Random seed for reproducibility
- `--video`: Enable video recording during training
- `--video_length`: Length of recorded videos
- `--video_interval`: Interval between video recordings
- `--enable_cameras`: Enable camera sensors for vision tasks
- `--checkpoint`: Path to checkpoint for resuming training

## 🎯 **Playing Flags**

### **Essential Flags:**
- `--task`: Task name
- `--agent`: Agent configuration
- `--checkpoint`: Path to trained model (optional - uses latest if not specified)

### **Optional Flags:**
- `--use_pretrained_checkpoint`: Use pre-trained models from Nucleus
- `--real-time`: Run in real-time mode
- `--video`: Enable video recording
- `--disable_fabric`: Disable fabric acceleration
- `--enable_cameras`: Enable camera sensors for vision tasks

## 📋 **Task Naming Conventions**

- **Isaac Lab built-in tasks**: `Isaac-<TaskName>-v0` (manager-based) or `Isaac-<TaskName>-Direct-v0` (direct workflow)
- **External project tasks**: `Template-<TaskName>-<Workflow>-v0`
- **Camera tasks**: `Isaac-<TaskName>-RGB-Camera-Direct-v0` or `Isaac-<TaskName>-Depth-Camera-Direct-v0`

## 🚀 **Quick Start Examples**

### **Test Environment First:**
```bash
# List all available tasks
isaaclab.bat -p scripts\environments\list_envs.py

# Test with random actions
isaaclab.bat -p scripts\environments\random_agent.py --task=Isaac-Cartpole-Direct-v0

# Test with zero actions
isaaclab.bat -p scripts\environments\zero_agent.py --task=Isaac-Cartpole-Direct-v0
```

### **Complete Training Workflow:**
```bash
# 1. Train
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Cartpole-Direct-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --headless --max_iterations=1500

# 2. Play trained model
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Cartpole-Direct-v0 --agent=rsl_rl_ppo_cfg

# 3. Play with specific checkpoint
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Cartpole-Direct-v0 --agent=rsl_rl_ppo_cfg --checkpoint=logs\rsl_rl\cartpole_direct\model_1500.pt
```

## 💡 **Pro Tips:**

1. **Start Simple**: Begin with `Isaac-Cartpole-Direct-v0` for learning
2. **Use Headless**: Always use `--headless` for training to maximize performance
3. **Large Environments**: Use `--num_envs=4096` for faster training
4. **Test First**: Always test with random/zero agents before training
5. **Monitor Progress**: Use TensorBoard to monitor training: `tensorboard --logdir logs/`