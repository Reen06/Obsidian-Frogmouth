
# Using Existing Isaac Lab Projects from External Storage

## 🔌 **Step 1: Connect Your Hard Drive**

First, plug in your hard drive and locate your existing projects:

```bash
# Navigate to your hard drive
cd /path/to/your/harddrive

# List your existing projects
ls -la
# You should see directories like:
# my_robot_project/
# another_robot_project/
# third_robot_project/
```

## 📁 **Step 2: Project Structure Check**

Make sure your existing projects have the correct structure:

```
your_project/
├── source/
│   └── your_project/
│       ├── setup.py
│       ├── pyproject.toml
│       ├── config/
│       │   └── extension.toml
│       └── your_project/
│           ├── tasks/
│           ├── assets/
│           └── ...
├── scripts/
├── README.md
└── .git/
```

## 🚀 **Step 3: Install Existing Projects**

### **Method 1: Direct Installation (Recommended)**

```bash
# Navigate to your Isaac Lab directory
cd /path/to/IsaacLab

# Install your existing project directly
python -m pip install -e /path/to/your/harddrive/my_robot_project/source/my_robot_project

# Or using isaaclab.sh/bat
./isaaclab.sh -p -m pip install -e /path/to/your/harddrive/my_robot_project/source/my_robot_project
```

### **Method 2: Copy to Local Machine First**

```bash
# Copy project to local machine
cp -r /path/to/your/harddrive/my_robot_project /home/user/projects/

# Navigate to Isaac Lab
cd /path/to/IsaacLab

# Install from local copy
python -m pip install -e /home/user/projects/my_robot_project/source/my_robot_project
```

### **Method 3: Multiple Projects at Once**

```bash
# Install multiple projects from hard drive
python -m pip install -e /path/to/your/harddrive/project1/source/project1
python -m pip install -e /path/to/your/harddrive/project2/source/project2
python -m pip install -e /path/to/your/harddrive/project3/source/project3
```

## 🔍 **Step 4: Verify Installation**

```bash
# Navigate to Isaac Lab directory
cd /path/to/IsaacLab

# List all available tasks (should include your projects)
python scripts/environments/list_envs.py

# You should see tasks like:
# Template-MyRobot-Direct-v0
# Template-AnotherBot-Direct-v0
# etc.
```

## 🎯 **Step 5: Use Your Existing Projects**

### **Training:**
```bash
# Train your existing robot
python scripts/reinforcement_learning/rsl_rl/train.py --task=Template-MyRobot-Direct-v0 --agent=rsl_rl_ppo_cfg

# Train another robot
python scripts/reinforcement_learning/rsl_rl/train.py --task=Template-AnotherBot-Direct-v0 --agent=rsl_rl_ppo_cfg
```

### **Playing:**
```bash
# Play with your existing robot
python scripts/reinforcement_learning/rsl_rl/play.py --task=Template-MyRobot-Direct-v0 --agent=rsl_rl_ppo_cfg

# Play with another robot
python scripts/reinforcement_learning/rsl_rl/play.py --task=Template-AnotherBot-Direct-v0 --agent=rsl_rl_ppo_cfg
```

## 📊 **Step 6: Using Existing Trained Models**

If your hard drive contains trained models, you can use them directly:

```bash
# Play with existing trained model
python scripts/reinforcement_learning/rsl_rl/play.py --task=Template-MyRobot-Direct-v0 --agent=rsl_rl_ppo_cfg --checkpoint=/path/to/your/harddrive/logs/rsl_rl/my_robot_task/2024-01-15_10-30-45/model_1500.pt

# Resume training from existing checkpoint
python scripts/reinforcement_learning/rsl_rl/train.py --task=Template-MyRobot-Direct-v0 --agent=rsl_rl_ppo_cfg --resume --load_run=2024-01-15_10-30-45 --load_checkpoint=1500
```

## 🔧 **Step 7: Complete Workflow Example**

Let's say you have a hard drive with 3 projects:

```bash
# 1. Connect hard drive and list projects
ls /media/usb/projects/
# Output: robot_arm/  mobile_robot/  humanoid_robot/

# 2. Navigate to Isaac Lab
cd /home/user/IsaacLab

# 3. Install all projects
python -m pip install -e /media/usb/projects/robot_arm/source/robot_arm
python -m pip install -e /media/usb/projects/mobile_robot/source/mobile_robot  
python -m pip install -e /media/usb/projects/humanoid_robot/source/humanoid_robot

# 4. Verify installation
python scripts/environments/list_envs.py

# 5. Train robot arm
python scripts/reinforcement_learning/rsl_rl/train.py --task=Template-Robot-Arm-Direct-v0 --agent=rsl_rl_ppo_cfg

# 6. Train mobile robot
python scripts/reinforcement_learning/rsl_rl/train.py --task=Template-Mobile-Robot-Direct-v0 --agent=rsl_rl_ppo_cfg

# 7. Play with humanoid robot (if it has trained models)
python scripts/reinforcement_learning/rsl_rl/play.py --task=Template-Humanoid-Robot-Direct-v0 --agent=rsl_rl_ppo_cfg --checkpoint=/media/usb/logs/rsl_rl/humanoid_task/model_1500.pt
```

## 🗂️ **Step 8: Managing Multiple Projects**

### **List Installed Projects:**
```bash
# See what's installed
pip list | grep -E "(robot|template)"

# Or check Python path
python -c "import sys; print([p for p in sys.path if 'robot' in p])"
```

### **Uninstall Projects:**
```bash
# Uninstall specific project
pip uninstall robot_arm

# Uninstall all custom projects
pip uninstall robot_arm mobile_robot humanoid_robot
```

### **Update Projects:**
```bash
# Reinstall if project was updated on hard drive
python -m pip install -e /path/to/your/harddrive/updated_project/source/updated_project --force-reinstall
```

## 💡 **Key Points:**

1. **No `--new` Command**: You don't use `isaaclab.bat --new` for existing projects
2. **Direct Installation**: Use `pip install -e` with the path to your project's `source/` directory
3. **Same Commands**: Once installed, use the same training/playing commands as if you created the project locally
4. **Path Independence**: The project can stay on the hard drive - you don't need to copy it locally
5. **Multiple Projects**: You can install as many projects as you want from the hard drive
6. **Existing Models**: You can use existing trained models directly from the hard drive

## 🚨 **Troubleshooting:**

### **If Installation Fails:**
```bash
# Check if project has setup.py
ls /path/to/your/harddrive/project/source/project/setup.py

# Check if Isaac Lab is properly installed
python -c "import isaaclab; print('Isaac Lab installed')"

# Try with verbose output
python -m pip install -e /path/to/project/source/project -v
```

### **If Tasks Don't Appear:**
```bash
# Check if project was installed
pip list | grep your_project_name

# Check Python path
python -c "import your_project_name; print('Project imported successfully')"

# Restart Python environment
python scripts/environments/list_envs.py
```

This approach lets you use any existing Isaac Lab project from external storage without recreating it!