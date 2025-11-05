# Jetson Containers - Quick User Guide

Simple guide for using jetson-containers to run AI/ML containers.

## Overview

Jetson-containers provides pre-built Docker containers with AI/ML packages for NVIDIA Jetson devices. Use it to run PyTorch, LLMs, vision models, and more without building from source.

## Quick Start

### Run a Container (Auto-Pull)

```bash
# This will automatically find and pull the right container for your system
jetson-containers run $(autotag pytorch)

# Interactive session starts immediately
```

### Available Containers

Browse available containers:
- **ML**: `pytorch`, `tensorflow`, `jax`, `transformers`
- **LLM**: `vllm`, `ollama`, `text-generation-webui`, `llama-factory`
- **Vision**: `llava`, `nanoowl`, `nanosam`, `clip_trt`
- **Code**: `jupyterlab`, `vscode`
- **Full list**: Check `packages/` directory

## Common Commands

### Starting Containers

```bash
# Run PyTorch container
jetson-containers run $(autotag pytorch)

# Run JupyterLab (runs on port 8888)
jetson-containers run $(autotag jupyterlab)

# Run a specific container
jetson-containers run dustynv/pytorch:2.8-r36.2.0-cu128-24.04
```

### Interactive vs Command Mode

```bash
# Interactive shell (default)
jetson-containers run $(autotag pytorch)

# Run a command directly
jetson-containers run $(autotag pytorch) python --version

# Run your own script
jetson-containers run $(autotag pytorch) python /path/to/your/script.py
```

### Mounting Directories

```bash
# Mount your project directory
jetson-containers run -v /home/jetson/Projects:/workspace $(autotag pytorch)

# Mount multiple directories
jetson-containers run -v /data/models:/models -v /data/datasets:/datasets $(autotag pytorch)
```

### Background Containers

```bash
# Run in detached mode
jetson-containers run -d $(autotag jupyterlab)

# Check running containers
docker ps

# Stop a container
docker stop <container-id>
```

## Building Containers

### Build Specific Packages

```bash
# Build PyTorch
jetson-containers build pytorch

# Build multiple packages
jetson-containers build pytorch transformers

# Build with custom CUDA version
CUDA_VERSION=12.6 jetson-containers build pytorch
```

### List Available Packages

```bash
# List all available packages
jetson-containers list

# Search for specific packages
jetson-containers list pytorch
```

## Common Use Cases

### 1. PyTorch Development

```bash
# Start PyTorch container
jetson-containers run -v /home/jetson/Projects:/workspace $(autotag pytorch)

# Inside container:
python3 -c "import torch; print(torch.__version__)"
```

### 2. JupyterLab

```bash
# Start JupyterLab
jetson-containers run $(autotag jupyterlab)

# Access at: http://localhost:8888
# Token shown in container output
```

### 3. LLM Models

```bash
# Run Llama with Ollama
jetson-containers run $(autotag ollama)

# Inside container:
ollama run llama2
```

### 4. Vision Models

```bash
# Run NanoOWL (object detection)
jetson-containers run $(autotag nanoowl)

# Run NanoSAM (segment anything)
jetson-containers run $(autotag nanosam)
```

## Managing Containers

### View Running Containers

```bash
docker ps
docker ps -a  # Include stopped containers
```

### Stop/Start Containers

```bash
# Stop running container
docker stop <container-id>

# Start stopped container
docker start <container-id>

# Remove container
docker rm <container-id>
```

### View Logs

```bash
# View container logs
docker logs <container-id>

# Follow logs in real-time
docker logs -f <container-id>
```

## Tips & Tricks

### Finding the Right Container

```bash
# What container name will be used?
echo $(autotag pytorch)

# Search for available versions
jetson-containers list pytorch
```

### Passing Environment Variables

```bash
jetson-containers run -e MY_VAR=value $(autotag pytorch)
```

### Using Multiple GPUs

```bash
jetson-containers run --gpus all $(autotag pytorch)
```

### Network Access

```bash
# Container already uses host networking by default
# But you can also expose specific ports
jetson-containers run -p 8080:8080 $(autotag jupyterlab)
```

## Troubleshooting

### Container Not Starting

```bash
# Check Docker is running
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker
```

### Out of Space

```bash
# Check disk usage
df -h

# Clean up unused containers
docker system prune

# Clean up including images
docker system prune -a
```

### Wrong Container Version

```bash
# Pull latest version
docker pull dustynv/pytorch:latest

# Or rebuild locally
jetson-containers build pytorch
```

## Useful Commands Reference

| Command | Description |
|---------|-------------|
| `jetson-containers run --name my_custom_container $(autotag <package>)` | Run container with auto-tag |
| `jetson-containers build <package>` | Build container from source |
| `jetson-containers list` | List all available packages |
| `autotag <package>` | Find compatible container |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers |
| `docker stop <id>` | Stop a container |
| `docker logs <id>` | View container logs |
| `docker rm <container_name>` | Delete a container|
| `docker exec -it <id> /bin/bash` | Enter running container |

## Finding More Containers

Check these directories for available packages:
- `packages/pytorch/` - PyTorch variants
- `packages/llm/` - Large Language Models
- `packages/vlm/` - Vision Language Models
- `packages/code/` - Development tools
- `packages/robots/` - Robotics packages

Each package has its own README with specific usage instructions.

## Getting Help

```bash
# Help for jetson-containers
jetson-containers --help

# Help for specific command
jetson-containers build --help
jetson-containers run --help
```

**Need more details?** See the main README.md in the jetson-containers directory for full documentation.

