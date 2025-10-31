# Micromamba Environment Setup Guide

This guide shows you how to create a micromamba environment and set up auto-executable programs that are accessible from anywhere.

## Step 1: Create Micromamba Environment

```bash
micromamba create -n sysinfoenv python=3.11
micromamba activate sysinfoenv
```

## Step 2: Create Your Project Directory

```bash
# Go to your project folder (use absolute path)
cd /absolute/path/to/your/projectfolder
```

## Step 3: Create Your Program

Create a Python script:

```bash
nano sysinfo.py
```

Add this code:

```python
#!/usr/bin/env python3
import platform, os

def main():
    print("=== System Info ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python version: {platform.python_version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Current working directory: {os.getcwd()}")

if __name__ == "__main__":
    main()
```

Make it executable:

```bash
chmod +x sysinfo.py
```

## Step 4: Configure PATH for Auto-Access

Set up the environment to add your project folder to PATH when activated:

```bash
# Create the activation directory
mkdir -p ~/.micromamba/envs/sysinfoenv/etc/conda/activate.d

# Create the activation script
nano ~/.micromamba/envs/sysinfoenv/etc/conda/activate.d/path.sh
```

Add this content (replace with your actual path):

```bash
export PATH="/absolute/path/to/your/projectfolder:$PATH"
```

## Step 5: Test Your Setup

```bash
# Deactivate and reactivate to load the changes
micromamba deactivate
micromamba activate sysinfoenv

# Test from any directory
cd ~
python /absolute/path/to/your/projectfolder/sysinfo.py

# Or if you're in the project folder:
cd /absolute/path/to/your/projectfolder
python sysinfo.py
```

## Important Notes

- You need to use `python sysinfo.py` to run the script
- The PATH configuration makes the file accessible, but Python still needs to execute it
- Always use absolute paths in the path.sh configuration file

## Alternative: Use the Setup Script

For a more automated approach, use the `create-mamba-env` command:

```bash
cd ~/Projects
create-mamba-env
```

See [README-create-mamba-env.md](README-create-mamba-env.md) for more details.
