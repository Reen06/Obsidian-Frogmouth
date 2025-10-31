# 🔍 **Debugging and Profiling - Complete Guide**

Isaac Lab provides comprehensive debugging and profiling tools to help you identify performance bottlenecks, debug issues, and optimize your training.

## 📊 **TensorBoard Integration**

### **Start TensorBoard**
```bash
# Start TensorBoard for all experiments
tensorboard --logdir logs/ --port 6006

# Start TensorBoard for specific experiment
tensorboard --logdir logs/rsl_rl/my_experiment/2024-01-08_22-31-59/ --port 6006

# Start TensorBoard for multiple experiments
tensorboard --logdir logs/rsl_rl/experiment1:logs/rsl_rl/experiment2 --port 6006
```

### **TensorBoard Features**
- **Training Metrics**: Loss curves, rewards, episode lengths
- **Hyperparameters**: Learning rates, network architectures
- **System Metrics**: GPU usage, memory consumption
- **Model Graphs**: Network architecture visualization
- **Images**: Camera sensor data (if enabled)

### **Access TensorBoard**
Open your browser and navigate to:
```
http://localhost:6006
```

## 🚀 **Performance Profiling**

### **Environment Performance Profiling**
```bash
# Profile environment performance
isaaclab.bat -p scripts/benchmarks/benchmark_non_rl.py \
    --task=Isaac-Ant-Direct-v0 \
    --num_envs=4096 \
    --headless \
    --num_steps=10000

# Profile with specific parameters
isaaclab.bat -p scripts/benchmarks/benchmark_non_rl.py \
    --task=Isaac-Ant-Direct-v0 \
    --num_envs=8192 \
    --headless \
    --num_steps=50000 \
    --profile=True
```

### **RL Training Profiling**
```bash
# Profile RL training performance
isaaclab.bat -p scripts/benchmarks/benchmark_rsl_rl.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --num_envs=4096 \
    --headless \
    --max_iterations=100

# Profile with specific RL library
isaaclab.bat -p scripts/benchmarks/benchmark_rl_games.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rl_games_ppo_cfg \
    --num_envs=4096 \
    --headless
```

### **Camera Performance Profiling**
```bash
# Profile camera performance
isaaclab.bat -p scripts/benchmarks/benchmark_cameras.py \
    --task=Isaac-Cartpole-RGB-Camera-Direct-v0 \
    --num_envs=1024 \
    --headless \
    --height=1080 \
    --width=1920
```

## 🔧 **Common Error Messages and Solutions**

### **CUDA Out of Memory**
```bash
# Error: CUDA out of memory
# Solutions:

# 1. Reduce number of environments
--num_envs=1024

# 2. Reduce batch size
agent.algorithm.num_mini_batches=2

# 3. Use CPU simulation
scene.device=cpu

# 4. Reduce network size
agent.policy.actor_hidden_dims=[256,256]
agent.policy.critic_hidden_dims=[256,256]
```

### **Task Not Found**
```bash
# Error: Task not found
# Solutions:

# 1. Check available tasks
isaaclab.bat -p scripts/environments/list_envs.py

# 2. Verify project installation
pip list | findstr project_name

# 3. Check task registration
python -c "import gymnasium as gym; print([env for env in gym.envs.registry.keys() if 'YourTask' in env])"

# 4. Reinstall project
pip uninstall project_name
pip install -e source/project_name
```

### **Import Errors**
```bash
# Error: ImportError: No module named 'isaaclab'
# Solutions:

# 1. Reinstall Isaac Lab
isaaclab.bat -i

# 2. Check Python path
python -c "import sys; print(sys.path)"

# 3. Verify installation
python -c "import isaaclab; print(isaaclab.__version__)"

# 4. Check for conflicting packages
pip list | findstr "torch gym isaac"
```

### **Camera Tasks Not Working**
```bash
# Error: Camera tasks not working
# Solutions:

# 1. Add --enable_cameras flag
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Cartpole-RGB-Camera-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --enable_cameras

# 2. Check camera configuration
# Verify camera is properly configured in scene

# 3. Test with simple camera task
isaaclab.bat -p scripts/environments/random_agent.py \
    --task=Isaac-Cartpole-RGB-Camera-Direct-v0 \
    --enable_cameras
```

### **Performance Issues**
```bash
# Error: Slow training performance
# Solutions:

# 1. Use headless mode
--headless

# 2. Disable fabric if having issues
--disable_fabric

# 3. Reduce number of environments
--num_envs=2048

# 4. Use CPU simulation
scene.device=cpu
```

### **Checkpoint Loading Errors**
```bash
# Error: Checkpoint loading failed
# Solutions:

# 1. Verify checkpoint exists
ls logs/rsl_rl/experiment_name/timestamp/

# 2. Use correct agent name
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Isaac-Ant-v0 \
    --agent=rsl_rl_ppo_cfg \
    --checkpoint=path/to/model.pt

# 3. Check checkpoint compatibility
# Ensure checkpoint matches current Isaac Lab version
```

## 📈 **Training Monitoring**

### **Check Training Logs**
```bash
# View training output
tail -f logs/rsl_rl/experiment_name/timestamp/training.log

# Windows PowerShell
Get-Content logs/rsl_rl/experiment_name/timestamp/training.log -Wait

# Check specific log sections
grep "ERROR" logs/rsl_rl/experiment_name/timestamp/training.log
grep "WARNING" logs/rsl_rl/experiment_name/timestamp/training.log
```

### **Check Configuration Files**
```bash
# View agent configuration
cat logs/rsl_rl/experiment_name/timestamp/params/agent.yaml

# View environment configuration
cat logs/rsl_rl/experiment_name/timestamp/params/env.yaml

# Windows PowerShell
Get-Content logs/rsl_rl/experiment_name/timestamp/params/agent.yaml
Get-Content logs/rsl_rl/experiment_name/timestamp/params/env.yaml
```

### **Monitor System Resources**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor CPU usage
top -p $(pgrep python)

# Monitor memory usage
free -h
```

## 🔍 **Debugging Techniques**

### **Environment Debugging**
```python
# Add debugging to environment
class MyTaskEnv(DirectRLEnv):
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards with debugging."""
        
        # Compute rewards
        reward = self._compute_reward()
        
        # Debug: Log reward components
        if self.cfg.debug:
            print(f"Reward: {reward.mean():.4f}")
            print(f"Joint pos: {self.joint_pos.mean():.4f}")
            print(f"Joint vel: {self.joint_vel.mean():.4f}")
        
        return reward
```

### **Action Debugging**
```python
# Debug actions
def _apply_action(self) -> None:
    """Apply actions with debugging."""
    
    # Debug: Log action statistics
    if self.cfg.debug:
        print(f"Action mean: {self.actions.mean():.4f}")
        print(f"Action std: {self.actions.std():.4f}")
        print(f"Action min: {self.actions.min():.4f}")
        print(f"Action max: {self.actions.max():.4f}")
    
    # Apply actions
    self.robot.set_joint_effort_target(
        self.actions * self.cfg.action_scale,
        joint_ids=self._joint_indices
    )
```

### **Observation Debugging**
```python
# Debug observations
def _get_observations(self) -> dict:
    """Get observations with debugging."""
    
    obs = self._compute_observations()
    
    # Debug: Log observation statistics
    if self.cfg.debug:
        print(f"Observation shape: {obs.shape}")
        print(f"Observation mean: {obs.mean():.4f}")
        print(f"Observation std: {obs.std():.4f}")
        print(f"Observation min: {obs.min():.4f}")
        print(f"Observation max: {obs.max():.4f}")
    
    return {"policy": obs}
```

## 🎯 **Performance Optimization**

### **Memory Optimization**
```python
# Optimize memory usage
@configclass
class OptimizedEnvCfg(DirectRLEnvCfg):
    """Optimized environment configuration."""
    
    # Reduce memory usage
    scene: MyTaskSceneCfg = MyTaskSceneCfg(
        num_envs=2048,  # Reduce if memory limited
        env_spacing=1.0,  # Reduce spacing
        replicate_physics=True,  # Use GPU physics
        clone_in_fabric=True,  # Use fabric cloning
    )
    
    # Optimize simulation
    sim = SimulationCfg(
        dt=1/120,  # Standard timestep
        substeps=1,  # Minimal substeps
        up_axis="z",
        gravity=(0.0, 0.0, -9.81),
    )
```

### **GPU Optimization**
```python
# Optimize GPU usage
@configclass
class GPTOptimizedCfg:
    """GPU-optimized configuration."""
    
    # Use GPU simulation
    device: str = "cuda"
    
    # Optimize batch sizes
    num_mini_batches: int = 4
    num_learning_epochs: int = 5
    
    # Optimize network size
    actor_hidden_dims: list = [512, 512, 256]
    critic_hidden_dims: list = [512, 512, 256]
```

### **CPU Optimization**
```python
# Optimize CPU usage
@configclass
class CPUOptimizedCfg:
    """CPU-optimized configuration."""
    
    # Use CPU simulation
    device: str = "cpu"
    
    # Reduce batch sizes
    num_mini_batches: int = 2
    num_learning_epochs: int = 3
    
    # Reduce network size
    actor_hidden_dims: list = [256, 256]
    critic_hidden_dims: list = [256, 256]
```

## 🛠️ **Debugging Tools**

### **Python Debugger**
```python
# Use Python debugger
import pdb

def _get_rewards(self) -> torch.Tensor:
    """Compute rewards with debugger."""
    
    # Set breakpoint
    pdb.set_trace()
    
    # Compute rewards
    reward = self._compute_reward()
    
    return reward
```

### **Logging**
```python
# Use Python logging
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _get_rewards(self) -> torch.Tensor:
    """Compute rewards with logging."""
    
    # Log reward computation
    logger.debug(f"Computing rewards for {self.num_envs} environments")
    
    reward = self._compute_reward()
    
    # Log reward statistics
    logger.info(f"Reward mean: {reward.mean():.4f}, std: {reward.std():.4f}")
    
    return reward
```

### **Assertions**
```python
# Use assertions for debugging
def _get_observations(self) -> dict:
    """Get observations with assertions."""
    
    obs = self._compute_observations()
    
    # Assert observation properties
    assert obs.shape[0] == self.num_envs, "Observation batch size mismatch"
    assert not torch.isnan(obs).any(), "NaN values in observations"
    assert not torch.isinf(obs).any(), "Inf values in observations"
    
    return {"policy": obs}
```

## 📊 **Profiling Results Analysis**

### **Performance Metrics**
- **FPS**: Frames per second
- **Throughput**: Steps per second
- **Memory Usage**: GPU/CPU memory consumption
- **Training Time**: Time per iteration
- **Convergence**: Learning curves

### **Bottleneck Identification**
```python
# Identify performance bottlenecks
def profile_environment():
    """Profile environment performance."""
    
    # Profile different components
    import time
    
    # Profile observation computation
    start_time = time.time()
    obs = env._get_observations()
    obs_time = time.time() - start_time
    
    # Profile reward computation
    start_time = time.time()
    reward = env._get_rewards()
    reward_time = time.time() - start_time
    
    # Profile action application
    start_time = time.time()
    env._apply_action()
    action_time = time.time() - start_time
    
    print(f"Observation time: {obs_time:.4f}s")
    print(f"Reward time: {reward_time:.4f}s")
    print(f"Action time: {action_time:.4f}s")
```

## 💡 **Debugging Best Practices**

### **1. Start Simple**
- Begin with basic environments
- Use simple reward functions
- Test with random actions first

### **2. Incremental Debugging**
- Add one component at a time
- Test each component individually
- Use assertions and logging

### **3. Performance Monitoring**
- Monitor training metrics
- Check system resources
- Use profiling tools

### **4. Error Handling**
- Use try-catch blocks
- Log error details
- Provide meaningful error messages

### **5. Reproducibility**
- Use fixed seeds
- Log all hyperparameters
- Save configuration files

## 🚀 **Quick Debugging Commands**

```bash
# Check available tasks
isaaclab.bat -p scripts/environments/list_envs.py

# Test with random actions
isaaclab.bat -p scripts/environments/random_agent.py --task=<TASK>

# Test with zero actions
isaaclab.bat -p scripts/environments/zero_agent.py --task=<TASK>

# Profile performance
isaaclab.bat -p scripts/benchmarks/benchmark_non_rl.py --task=<TASK>

# Start TensorBoard
tensorboard --logdir logs/

# Check GPU usage
nvidia-smi

# Monitor training
tail -f logs/rsl_rl/experiment_name/training.log
```

These debugging and profiling tools provide everything you need to identify issues, optimize performance, and ensure robust training!
