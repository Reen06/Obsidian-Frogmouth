
# Where to Find Your Trained Models and Checkpoints

## 📁 **Default Storage Locations**

Your trained models are stored in the `logs/` directory in your Isaac Lab project root. The structure varies by RL library:

### **RSL-RL Models:**
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

### **RL-Games Models:**
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

### **SKRL Models:**
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

### **SB3 Models:**
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

## 🔍 **How to Find Your Specific Models**

### **Method 1: Check the Training Output**
When you run training, look for these lines in the console:
```bash
[INFO] Logging experiment in directory: /path/to/logs/rsl_rl/my_robot_task
Exact experiment name requested from command line: 2024-01-15_10-30-45
```

### **Method 2: List All Available Checkpoints**
```bash
# Navigate to your project directory
cd /path/to/your/project

# List all training logs
ls -la logs/

# List specific RL library logs
ls -la logs/rsl_rl/
ls -la logs/rl_games/
ls -la logs/skrl/
ls -la logs/sb3/
```

### **Method 3: Find Latest Checkpoint Programmatically**
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

## 🎮 **How to Use Your Trained Models**

### **Playing with RSL-RL:**
```bash
# Use latest checkpoint automatically
python scripts/rsl_rl/play.py --task=Template-My-Task-Direct-v0 --agent=rsl_rl_ppo_cfg

# Use specific checkpoint
python scripts/rsl_rl/play.py --task=Template-My-Task-Direct-v0 --agent=rsl_rl_ppo_cfg --checkpoint=logs/rsl_rl/my_robot_task/2024-01-15_10-30-45/model_1500.pt

# Use last checkpoint
python scripts/rsl_rl/play.py --task=Template-My-Task-Direct-v0 --agent=rsl_rl_ppo_cfg --load_checkpoint=-1
```

### **Playing with RL-Games:**
```bash
# Use latest checkpoint automatically
python scripts/rl_games/play.py --task=Template-My-Task-Direct-v0 --agent=rl_games_ppo_cfg

# Use specific checkpoint
python scripts/rl_games/play.py --task=Template-My-Task-Direct-v0 --agent=rl_games_ppo_cfg --checkpoint=logs/rl_games/my_config/2024-01-15_10-30-45/my_config.pth
```

### **Playing with SKRL:**
```bash
# Use latest checkpoint automatically
python scripts/skrl/play.py --task=Template-My-Task-Direct-v0 --agent=skrl_ppo_cfg

# Use specific checkpoint
python scripts/skrl/play.py --task=Template-My-Task-Direct-v0 --agent=skrl_ppo_cfg --checkpoint=logs/skrl/my_task/2024-01-15_10-30-45_ppo_torch/checkpoints/agent_1500.pt
```

### **Playing with SB3:**
```bash
# Use latest checkpoint automatically
python scripts/sb3/play.py --task=Template-My-Task-Direct-v0 --agent=sb3_ppo_cfg

# Use specific checkpoint
python scripts/sb3/play.py --task=Template-My-Task-Direct-v0 --agent=sb3_ppo_cfg --checkpoint=logs/sb3/my_task/2024-01-15_10-30-45/model.zip
```

## 📊 **Monitoring Training Progress**

### **TensorBoard:**
```bash
# Start TensorBoard
tensorboard --logdir logs/

# Or for specific experiment
tensorboard --logdir logs/rsl_rl/my_robot_task/2024-01-15_10-30-45/
```

### **Check Training Logs:**
```bash
# View training output
tail -f logs/rsl_rl/my_robot_task/2024-01-15_10-30-45/training.log

# Check configuration files
cat logs/rsl_rl/my_robot_task/2024-01-15_10-30-45/params/agent.yaml
cat logs/rsl_rl/my_robot_task/2024-01-15_10-30-45/params/env.yaml
```

## 🔄 **Resuming Training**

### **Resume from Checkpoint:**
```bash
# RSL-RL
python scripts/rsl_rl/train.py --task=Template-My-Task-Direct-v0 --agent=rsl_rl_ppo_cfg --resume --load_run=2024-01-15_10-30-45 --load_checkpoint=1500

# RL-Games
python scripts/rl_games/train.py --task=Template-My-Task-Direct-v0 --agent=rl_games_ppo_cfg --checkpoint=logs/rl_games/my_config/2024-01-15_10-30-45/my_config.pth

# SKRL
python scripts/skrl/train.py --task=Template-My-Task-Direct-v0 --agent=skrl_ppo_cfg --checkpoint=logs/skrl/my_task/2024-01-15_10-30-45_ppo_torch/checkpoints/agent_1500.pt

# SB3
python scripts/sb3/train.py --task=Template-My-Task-Direct-v0 --agent=sb3_ppo_cfg --checkpoint=logs/sb3/my_task/2024-01-15_10-30-45/model.zip
```

## 💾 **Model File Formats**

- **RSL-RL**: `.pt` files (PyTorch checkpoints)
- **RL-Games**: `.pth` files (PyTorch checkpoints)
- **SKRL**: `.pt` files (PyTorch checkpoints)
- **SB3**: `.zip` files (compressed model + normalization)

## 🎯 **Quick Reference Commands**

```bash
# Find all your trained models
find logs/ -name "*.pt" -o -name "*.pth" -o -name "*.zip"

# Find latest checkpoint for any RL library
find logs/ -name "model_*.pt" -o -name "*.pth" -o -name "model.zip" | xargs ls -lt | head -1

# List all your experiments
ls logs/*/

# Check training progress
tensorboard --logdir logs/ --port=6006
```

Your models are automatically saved during training, and the play scripts can automatically find the latest checkpoint if you don't specify one explicitly!