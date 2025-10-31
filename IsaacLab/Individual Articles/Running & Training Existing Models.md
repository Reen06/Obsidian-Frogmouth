# Working with Multiple Isaac Lab Projects

## 🔍 **Step 1: List All Available Tasks**

First, see what tasks are available from all your installed projects:

```bash
# Navigate to your Isaac Lab directory
cd /path/to/IsaacLab

# List ALL available environments from ALL projects
python scripts/environments/list_envs.py
```

This will show you a table like:
```
+-------+--------------------------------+--------------------------------+--------------------------------+
| S. No.| Task Name                       | Entry Point                    | Config                         |
+-------+--------------------------------+--------------------------------+--------------------------------+
| 1     | Isaac-Cartpole-v0              | isaaclab_tasks.cartpole:...    | isaaclab_tasks.cartpole:...    |
| 2     | Template-MyRobot-Direct-v0     | my_project.tasks:...           | my_project.tasks:...           |
| 3     | Template-AnotherBot-Direct-v0  | another_project.tasks:...      | another_project.tasks:...      |
+-------+--------------------------------+--------------------------------+--------------------------------+
```

## 🎯 **Step 2: Identify Your Project Tasks**

Your external projects will have tasks that start with **`Template-`** instead of **`Isaac-`**:

- **Isaac Lab built-in tasks**: `Isaac-Cartpole-v0`, `Isaac-Ant-v0`, etc.
- **Your external projects**: `Template-MyRobot-Direct-v0`, `Template-AnotherBot-Direct-v0`, etc.

## 🚀 **Step 3: Training Commands**

### **For Your External Project Tasks:**

```bash
# RSL-RL Training
python scripts/reinforcement_learning/rsl_rl/train.py --task=Template-MyRobot-Direct-v0 --agent=rsl_rl_ppo_cfg

# RL-Games Training  
python scripts/reinforcement_learning/rl_games/train.py --task=Template-MyRobot-Direct-v0 --agent=rl_games_ppo_cfg

# SKRL Training
python scripts/reinforcement_learning/skrl/train.py --task=Template-MyRobot-Direct-v0 --agent=skrl_ppo_cfg

# SB3 Training
python scripts/reinforcement_learning/sb3/train.py --task=Template-MyRobot-Direct-v0 --agent=sb3_ppo_cfg
```

### **For Isaac Lab Built-in Tasks:**

```bash
# RSL-RL Training
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --agent=rsl_rl_ppo_cfg

# RL-Games Training
python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --agent=rl_games_ppo_cfg
```

## 🎮 **Step 4: Playing Commands**

### **For Your External Project Tasks:**

```bash
# RSL-RL Playing
python scripts/reinforcement_learning/rsl_rl/play.py --task=Template-MyRobot-Direct-v0 --agent=rsl_rl_ppo_cfg

# RL-Games Playing
python scripts/reinforcement_learning/rl_games/play.py --task=Template-MyRobot-Direct-v0 --agent=rl_games_ppo_cfg

# SKRL Playing
python scripts/reinforcement_learning/skrl/play.py --task=Template-MyRobot-Direct-v0 --agent=skrl_ppo_cfg

# SB3 Playing
python scripts/reinforcement_learning/sb3/play.py --task=Template-MyRobot-Direct-v0 --agent=sb3_ppo_cfg
```

### **For Isaac Lab Built-in Tasks:**

```bash
# RSL-RL Playing
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Cartpole-v0 --agent=rsl_rl_ppo_cfg

# RL-Games Playing
python scripts/reinforcement_learning/rl_games/play.py --task=Isaac-Cartpole-v0 --agent=rl_games_ppo_cfg
```

## 📁 **Step 5: Finding Your Project's Models**

Your external project models will be stored in the same `logs/` directory structure:

```bash
# Your external project models
logs/rsl_rl/my_robot_task/2024-01-15_10-30-45/model_1500.pt
logs/rl_games/my_robot_config/2024-01-15_10-30-45/my_robot_config.pth

# Isaac Lab built-in models  
logs/rsl_rl/cartpole_direct/2024-01-15_10-30-45/model_1500.pt
logs/rl_games/cartpole_direct/2024-01-15_10-30-45/cartpole_direct.pth
```

## 🔧 **Step 6: Testing Your Projects**

### **Test with Random Actions:**
```bash
# Test your external project
python scripts/environments/random_agent.py --task=Template-MyRobot-Direct-v0

# Test Isaac Lab built-in task
python scripts/environments/random_agent.py --task=Isaac-Cartpole-v0
```

### **Test with Zero Actions:**
```bash
# Test your external project
python scripts/environments/zero_agent.py --task=Template-MyRobot-Direct-v0

# Test Isaac Lab built-in task
python scripts/environments/zero_agent.py --task=Isaac-Cartpole-v0
```

## 🎯 **Step 7: Complete Workflow Example**

Let's say you have two projects: `my_robot_project` and `another_robot_project`:

```bash
# 1. List all available tasks
python scripts/environments/list_envs.py

# 2. Train your first robot
python scripts/reinforcement_learning/rsl_rl/train.py --task=Template-MyRobot-Direct-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096

# 3. Train your second robot  
python scripts/reinforcement_learning/rsl_rl/train.py --task=Template-AnotherBot-Direct-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096

# 4. Play with your first robot
python scripts/reinforcement_learning/rsl_rl/play.py --task=Template-MyRobot-Direct-v0 --agent=rsl_rl_ppo_cfg

# 5. Play with your second robot
python scripts/reinforcement_learning/rsl_rl/play.py --task=Template-AnotherBot-Direct-v0 --agent=rsl_rl_ppo_cfg

# 6. Resume training your first robot
python scripts/reinforcement_learning/rsl_rl/train.py --task=Template-MyRobot-Direct-v0 --agent=rsl_rl_ppo_cfg --resume --load_run=2024-01-15_10-30-45 --load_checkpoint=1500
```

## 🔍 **Step 8: Quick Reference Commands**

```bash
# List all tasks from all projects
python scripts/environments/list_envs.py

# Find all your external project tasks
python scripts/environments/list_envs.py | grep "Template-"

# Find all Isaac Lab built-in tasks
python scripts/environments/list_envs.py | grep "Isaac-"

# Test any task with random actions
python scripts/environments/random_agent.py --task=<TASK_NAME>

# Train any task
python scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --agent=rsl_rl_ppo_cfg

# Play any task
python scripts/reinforcement_learning/rsl_rl/play.py --task=<TASK_NAME> --agent=rsl_rl_ppo_cfg
```

## 💡 **Key Points:**

1. **Task Names**: Your external projects use `Template-` prefix, Isaac Lab uses `Isaac-` prefix
2. **Same Commands**: Use the same training/playing commands regardless of which project the task comes from
3. **Same Logs**: All models are stored in the same `logs/` directory structure
4. **Automatic Discovery**: Isaac Lab automatically finds all installed projects and their tasks
5. **No Path Needed**: You don't need to specify project paths - just use the task name

The beauty of Isaac Lab's extension system is that once you install your projects (`pip install -e source/your_project`), they become seamlessly integrated and you can use them just like built-in tasks!