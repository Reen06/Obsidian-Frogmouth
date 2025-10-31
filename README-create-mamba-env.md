# create-mamba-env Usage

## Overview
The `create-mamba-env` command sets up a new micromamba environment linked to a project directory with automatic PATH management and optional auto-activation.

## Usage

### Basic Usage

1. **Navigate to your project directory** (or where you want to create the project):
   ```bash
   cd /home/jetson/Projects
   ```

2. **Run the command**:
   ```bash
   create-mamba-env
   ```

3. **Follow the prompts**:
   - Choose: Use current directory or create subdirectory
   - Enter environment name (no spaces)
   - Enter Python version (default: 3.11)
   - Enable auto-activation? (y/n)

### What it does

- Creates a micromamba environment with your specified Python version
- Creates `activate.d/path.sh` and `deactivate.d/path.sh` files that add your project directory to PATH
- Creates a `.mamba-env` marker file in your project directory
- (Optional) Sets up auto-activation when entering the directory

### Auto-Activation Feature

If you enabled auto-activation, when you:
- `cd` into the project directory → micromamba environment activates automatically
- `cd` out of the project directory → micromamba environment deactivates

### After Setup

```bash
# Activate the environment
micromamba activate <env-name>

# Your scripts in the project directory are now in PATH
my-script.py

# Deactivate
micromamba deactivate
```

### Example

```bash
cd ~/Projects
create-mamba-env

# Prompts:
# 1) Use current dir: ~/Projects
# 2) Enter env name: myproject
# 3) Enter Python version: 3.11
# 4) Enable auto-activation? y

# Now when you:
cd ~/Projects
# Environment activates automatically!

# And when you leave:
cd ~
# Environment deactivates automatically
```

## Managing Environments

### List All Environments

```bash
micromamba env list
```

Output:
```
# conda environments:
#
base                  /home/jetson/miniconda
myproject          *  /home/jetson/.micromamba/envs/myproject
testenv                /home/jetson/.micromamba/envs/testenv
sysinfoenv             /home/jetson/.micromamba/envs/sysinfoenv
```

The `*` indicates the currently active environment.

### Remove an Environment

```bash
micromamba env remove -n <env-name>
```

**Example:**
```bash
micromamba env remove -n myproject
```

**Important:** Always delete the micromamba environment before deleting the project directory to keep things clean.

### Complete Workflow: Creating and Removing Environments

**1. Create a New Project with Environment**

```bash
# Navigate to where you want your project
cd ~/Projects

# Create the project and environment
create-mamba-env

# Follow the prompts:
# > 2 (create subdirectory)
# > Enter name: mynewproject
# > Environment name: mynewproject
# > Python version: 3.11
# > Auto-activation: y

# This creates:
# - Directory: ~/Projects/mynewproject/
# - Environment: mynewproject
# - PATH configuration files
```

**2. Working with Your Project**

```bash
# Navigate to project
cd ~/Projects/mynewproject

# If auto-activation is enabled, environment activates automatically!
# Otherwise, activate manually:
micromamba activate mynewproject

# Create your script
nano my_script.py

# Make it executable
chmod +x my_script.py

# Run it from anywhere (environment must be activated)
my_script.py
```

**3. Clean Up (Removing Everything)**

```bash
# First, remove the micromamba environment
micromamba env remove -n mynewproject

# Then delete the project directory
rm -rf ~/Projects/mynewproject

# Or do both at once:
micromamba env remove -n mynewproject && rm -rf ~/Projects/mynewproject
```

### Quick Reference

```bash
# List environments
micromamba env list

# Activate environment
micromamba activate <env-name>

# Deactivate environment
micromamba deactivate

# Create environment (via script)
create-mamba-env

# Remove environment
micromamba env remove -n <env-name>

# Check which Python you're using
which python

# See what's installed in current environment
micromamba list
```
