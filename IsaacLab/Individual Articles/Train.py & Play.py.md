
# 🎮 **Isaac Lab Training and Playing Scripts**

Here's a comprehensive overview of all the `train.py` and `play.py` scripts available in Isaac Lab:

## 🚀 **Training Scripts** (`train.py`)

### **1. RSL-RL Training** (`scripts/reinforcement_learning/rsl_rl/train.py`)
- **Description**: Train RL agents using RSL-RL library
- **Features**:
  - Video recording during training
  - Distributed training support
  - Checkpoint resuming
  - IO descriptor export
- **Usage**:
  ```cmd
  isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Cartpole-v0 --agent=rsl_rl_ppo_cfg
  isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Cartpole-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --max_iterations=1500
  isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Cartpole-v0 --agent=rsl_rl_ppo_cfg --video --video_length=200 --video_interval=2000
  ```

### **2. RL-Games Training** (`scripts/reinforcement_learning/rl_games/train.py`)
- **Description**: Train RL agents using RL-Games library
- **Features**:
  - Video recording during training
  - Distributed training support
  - Checkpoint resuming
  - Wandb integration
- **Usage**:
  ```cmd
  isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task=Isaac-Cartpole-v0 --agent=rl_games_ppo_cfg
  isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task=Isaac-Cartpole-v0 --agent=rl_games_ppo_cfg --distributed
  isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task=Isaac-Cartpole-v0 --agent=rl_games_ppo_cfg --wandb-project-name=my_project
  ```

### **3. SKRL Training** (`scripts/reinforcement_learning/skrl/train.py`)
- **Description**: Train RL agents using SKRL library
- **Features**:
  - Video recording during training
  - Distributed training support
  - Checkpoint resuming
  - Multi-agent support
- **Usage**:
  ```cmd
  isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task=Isaac-Cartpole-v0 --agent=skrl_ppo_cfg
  isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task=Isaac-Cartpole-v0 --agent=skrl_ppo_cfg --algorithm=PPO
  ```

### **4. Stable Baselines3 Training** (`scripts/reinforcement_learning/sb3/train.py`)
- **Description**: Train RL agents using Stable Baselines3 library
- **Features**:
  - Video recording during training
  - Checkpoint resuming
  - Logging intervals
  - Keep all training info option
- **Usage**:
  ```cmd
  isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task=Isaac-Cartpole-v0 --agent=sb3_ppo_cfg
  isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task=Isaac-Cartpole-v0 --agent=sb3_ppo_cfg --log_interval=100000
  ```

## 🎮 **Playing Scripts** (`play.py`)

### **1. RSL-RL Playing** (`scripts/reinforcement_learning/rsl_rl/play.py`)
- **Description**: Play trained RSL-RL checkpoints
- **Features**:
  - Video recording
  - Real-time mode
  - Pre-trained checkpoint support
  - Fabric disable option
- **Usage**:
  ```cmd
  isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Cartpole-v0 --agent=rsl_rl_ppo_cfg
  isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Cartpole-v0 --agent=rsl_rl_ppo_cfg --checkpoint=logs\rsl_rl\cartpole_direct\model_1500.pt
  isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Cartpole-v0 --agent=rsl_rl_ppo_cfg --use_pretrained_checkpoint
  ```

### **2. RL-Games Playing** (`scripts/reinforcement_learning/rl_games/play.py`)
- **Description**: Play trained RL-Games checkpoints
- **Features**:
  - Video recording
  - Pre-trained checkpoint support
  - Fabric disable option
- **Usage**:
  ```cmd
  isaaclab.bat -p scripts\reinforcement_learning\rl_games\play.py --task=Isaac-Cartpole-v0 --agent=rl_games_ppo_cfg
  isaaclab.bat -p scripts\reinforcement_learning\rl_games\play.py --task=Isaac-Cartpole-v0 --agent=rl_games_ppo_cfg --checkpoint=logs\rl_games\cartpole_direct\model.pth
  ```

### **3. SKRL Playing** (`scripts/reinforcement_learning/skrl/play.py`)
- **Description**: Play trained SKRL checkpoints
- **Features**:
  - Video recording
  - Pre-trained checkpoint support
  - Fabric disable option
- **Usage**:
  ```cmd
  isaaclab.bat -p scripts\reinforcement_learning\skrl\play.py --task=Isaac-Cartpole-v0 --agent=skrl_ppo_cfg
  isaaclab.bat -p scripts\reinforcement_learning\skrl\play.py --task=Isaac-Cartpole-v0 --agent=skrl_ppo_cfg --checkpoint=logs\skrl\cartpole_direct\model.zip
  ```

### **4. Stable Baselines3 Playing** (`scripts/reinforcement_learning/sb3/play.py`)
- **Description**: Play trained Stable Baselines3 checkpoints
- **Features**:
  - Video recording
  - Pre-trained checkpoint support
  - Fabric disable option
- **Usage**:
  ```cmd
  isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py --task=Isaac-Cartpole-v0 --agent=sb3_ppo_cfg
  isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py --task=Isaac-Cartpole-v0 --agent=sb3_ppo_cfg --checkpoint=logs\sb3\cartpole_direct\model.zip
  ```

## 🔍 **Environment Testing Scripts**

### **1. List Environments** (`scripts/environments/list_envs.py`)
- **Description**: List all available environments
- **Usage**:
  ```cmd
  isaaclab.bat -p scripts\environments\list_envs.py
  ```

### **2. Random Agent** (`scripts/environments/random_agent.py`)
- **Description**: Test environment with random actions
- **Usage**:
  ```cmd
  isaaclab.bat -p scripts\environments\random_agent.py --task=Isaac-Cartpole-v0
  isaaclab.bat -p scripts\environments\random_agent.py --task=Isaac-Cartpole-v0 --num_envs=64
  ```

### **3. Zero Agent** (`scripts/environments/zero_agent.py`)
- **Description**: Test environment with zero actions
- **Usage**:
  ```cmd
  isaaclab.bat -p scripts\environments\zero_agent.py --task=Isaac-Cartpole-v0
  isaaclab.bat -p scripts\environments\zero_agent.py --task=Isaac-Cartpole-v0 --num_envs=64
  ```

## 🎯 **Common Command-Line Arguments**

### **Training Arguments:**
- `--task`: Name of the task/environment
- `--agent`: Name of the RL agent configuration
- `--num_envs`: Number of parallel environments
- `--max_iterations`: Maximum training iterations
- `--seed`: Random seed
- `--video`: Enable video recording
- `--video_length`: Length of recorded videos
- `--video_interval`: Interval between video recordings
- `--checkpoint`: Path to checkpoint for resuming
- `--distributed`: Enable distributed training

### **Playing Arguments:**
- `--task`: Name of the task/environment
- `--agent`: Name of the RL agent configuration
- `--checkpoint`: Path to trained model checkpoint
- `--use_pretrained_checkpoint`: Use pre-trained models from Nucleus
- `--real-time`: Run in real-time mode
- `--video`: Enable video recording
- `--disable_fabric`: Disable fabric acceleration

## 🚀 **Complete Workflow Examples**

### **Training Workflow:**
```cmd
# 1. List available environments
isaaclab.bat -p scripts\environments\list_envs.py

# 2. Test environment with random actions
isaaclab.bat -p scripts\environments\random_agent.py --task=Isaac-Cartpole-v0

# 3. Train with RSL-RL
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Cartpole-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --max_iterations=1500

# 4. Play trained model
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Cartpole-v0 --agent=rsl_rl_ppo_cfg --checkpoint=logs\rsl_rl\cartpole_direct\model_1500.pt
```

### **Multi-Library Comparison:**
```cmd
# Train with different libraries
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Cartpole-v0 --agent=rsl_rl_ppo_cfg
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task=Isaac-Cartpole-v0 --agent=rl_games_ppo_cfg
isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task=Isaac-Cartpole-v0 --agent=skrl_ppo_cfg
isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task=Isaac-Cartpole-v0 --agent=sb3_ppo_cfg
```

## 📊 **RL Library Comparison**

| Library | Multi-Agent | Distributed | Vectorized | Performance | Algorithms |
|---------|-------------|-------------|-----------|-------------|-----------|
| **RSL-RL** | ❌ | ❌ | ✅ | ~1X | PPO, AMP |
| **RL-Games** | ❌ | ✅ | ✅ | ~1X | PPO, AMP |
| **SKRL** | ✅ | ✅ | ✅ | ~1X | PPO, AMP, IPPO, MAPPO |
| **SB3** | ❌ | ❌ | ❌ | ~1.6X | PPO (primary) |

## 💡 **Key Features:**

- **Unified Interface**: All libraries use the same command-line interface
- **Video Recording**: Built-in video recording for training and playing
- **Checkpoint Management**: Easy checkpoint saving and loading
- **Pre-trained Models**: Access to pre-trained models via Nucleus
- **Environment Testing**: Built-in random and zero agents for testing
- **Distributed Training**: Support for multi-GPU and multi-node training
- **Real-time Playing**: Real-time inference mode for playing

These scripts provide everything you need to train, test, and play with RL agents in Isaac Lab!