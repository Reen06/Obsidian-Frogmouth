# 🧠 **Complete Guide to Isaac Lab RL Libraries**

Based on the Isaac Lab Complete Reference, here's everything you need to know about the four RL libraries and when to use each one:

## 🚀 **1. RSL-RL (Robotics Systems Lab - Reinforcement Learning)**

### **📋 Overview:**
- **Full Name**: Robotics Systems Lab Reinforcement Learning Library
- **Version**: 3.0.1
- **Primary Focus**: Robotics-specific RL with advanced features
- **Performance**: ~1X baseline performance

### **🎯 Key Features:**
- **Advanced Robotics Features**:
  - Symmetry-based learning for locomotion
  - Curiosity-driven exploration (RND - Random Network Distillation)
  - Recurrent actor-critic networks
  - Policy distillation
  - Action clipping
  - Observation normalization

- **Algorithms**: PPO, AMP (single-agent), IPPO, MAPPO (multi-agent)
- **Multi-Agent**: ❌ No support
- **Distributed**: ❌ No support
- **Vectorized**: ✅ Yes
- **ML Frameworks**: PyTorch only
- **Observation Spaces**: Box spaces only

### **🔧 Configuration Features:**
```python
# Advanced RSL-RL features
class RslRlPpoActorCriticCfg:
    init_noise_std: float = 1.0
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [256, 256]
    critic_hidden_dims: list[int] = [256, 256]
    activation: str = "elu"

class RslRlPpoAlgorithmCfg:
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    symmetry: RslRlSymmetryCfg = None  # Symmetry learning
    rnd: RslRlRndCfg = None  # Curiosity exploration
```

### **🎮 When to Use RSL-RL:**
- **Robotics locomotion tasks** (quadrupeds, bipeds)
- **Tasks requiring symmetry** (walking, running)
- **Exploration-heavy environments**
- **Single-agent robotics applications**
- **When you need advanced robotics-specific features**
- **Research requiring cutting-edge RL techniques**

### **📝 Example Usage:**
```cmd
# Train with symmetry learning
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Velocity-Flat-Anymal-C-v0 --agent=rsl_rl_ppo_cfg

# Train with curiosity exploration
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096
```

---

## 🎮 **2. RL-Games**

### **📋 Overview:**
- **Full Name**: RL-Games (NVIDIA's RL library)
- **Version**: Custom Isaac Sim branch (Python 3.11 compatible)
- **Primary Focus**: High-performance, distributed RL
- **Performance**: ~1X baseline performance

### **🎯 Key Features:**
- **High Performance**:
  - Direct GPU buffer operations
  - Optimized for Isaac Sim
  - Asymmetric actor-critic support
  - Population-based training (PBT)

- **Algorithms**: PPO, AMP (single-agent), IPPO, MAPPO (multi-agent)
- **Multi-Agent**: ❌ No support
- **Distributed**: ✅ Yes (multi-GPU, multi-node)
- **Vectorized**: ✅ Yes
- **ML Frameworks**: PyTorch only
- **Observation Spaces**: Box and Dict spaces

### **🔧 Configuration Features:**
```python
# RL-Games wrapper features
class RlGamesVecEnvWrapper:
    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        rl_device: str = "cuda:0",
        clip_obs: float = 10.0,
        clip_actions: float = 1.0,
        obs_groups: dict[str, list[str]] = None,  # Dict observations
        concate_obs_group: bool = True,  # Concatenate or keep separate
    ):
```

### **🎮 When to Use RL-Games:**
- **High-performance training** (large-scale experiments)
- **Distributed training** (multiple GPUs/nodes)
- **Asymmetric actor-critic** (privileged information)
- **Population-based training** (hyperparameter optimization)
- **Production robotics applications**
- **When you need maximum performance**

### **📝 Example Usage:**
```cmd
# Distributed training
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task=Isaac-Velocity-Flat-Anymal-C-v0 --agent=rl_games_ppo_cfg --distributed

# With Wandb logging
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task=Isaac-Velocity-Flat-Anymal-C-v0 --agent=rl_games_ppo_cfg --wandb-project-name=my_project
```

---

## 🔬 **3. SKRL (Scikit-Reinforcement Learning)**

### **📋 Overview:**
- **Full Name**: Scikit-Reinforcement Learning
- **Version**: 1.4.3+
- **Primary Focus**: Research-friendly, multi-framework RL
- **Performance**: ~1X baseline performance

### **🎯 Key Features:**
- **Multi-Framework Support**:
  - PyTorch backend
  - JAX backend
  - JAX-NumPy backend
  - Easy framework switching

- **Advanced Features**:
  - Multi-agent support
  - Distributed training
  - Vectorized training
  - Composite observation spaces
  - Custom network architectures

- **Algorithms**: PPO, AMP (single-agent), IPPO, MAPPO (multi-agent)
- **Multi-Agent**: ✅ Yes (only library with multi-agent support)
- **Distributed**: ✅ Yes
- **Vectorized**: ✅ Yes
- **ML Frameworks**: PyTorch, JAX
- **Observation Spaces**: Box, Dict, and composite spaces

### **🔧 Configuration Features:**
```python
# SKRL wrapper features
def SkrlVecEnvWrapper(
    env: ManagerBasedRLEnv | DirectRLEnv | DirectMARLEnv,
    ml_framework: Literal["torch", "jax", "jax-numpy"] = "torch",
    wrapper: Literal["auto", "isaaclab", "isaaclab-single-agent", "isaaclab-multi-agent"] = "isaaclab",
):
```

### **🎮 When to Use SKRL:**
- **Multi-agent environments**
- **Research requiring JAX** (faster compilation, functional programming)
- **Complex observation spaces** (composite spaces)
- **When you need framework flexibility**
- **Academic research projects**
- **Custom network architectures**

### **📝 Example Usage:**
```cmd
# PyTorch backend
isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task=Isaac-Cartpole-Direct-v0 --agent=skrl_ppo_cfg

# Multi-agent training
isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task=Isaac-MultiAgent-Task-v0 --agent=skrl_ppo_cfg
```

---

## 🛡️ **4. Stable Baselines3 (SB3)**

### **📋 Overview:**
- **Full Name**: Stable Baselines3
- **Version**: 2.6+
- **Primary Focus**: Easy-to-use, well-tested RL
- **Performance**: ~1.6X baseline performance (slower than others)

### **🎯 Key Features:**
- **User-Friendly**:
  - Simple API
  - Well-documented
  - Stable implementations
  - Easy debugging

- **Algorithms**: PPO (primary), SAC, TD3, DQN, A2C, DDPG
- **Multi-Agent**: ❌ No support
- **Distributed**: ❌ No support
- **Vectorized**: ❌ No support
- **ML Frameworks**: PyTorch only
- **Observation Spaces**: Box spaces only

### **🔧 Configuration Features:**
```python
# SB3 wrapper features (optimized for speed)
class Sb3VecEnvWrapper(VecEnv):
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, fast_variant: bool = True):
        # 4x faster than original implementation
        # Uses numpy buffers for speed
        # Only logs episode info by default
```

### **🎮 When to Use Stable Baselines3:**
- **Learning RL concepts** (educational purposes)
- **Quick prototyping**
- **When you need maximum stability**
- **Small-scale experiments**
- **When performance is not critical**
- **Debugging and testing**

### **📝 Example Usage:**
```cmd
# Simple training
isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task=Isaac-Cartpole-Direct-v0 --agent=sb3_ppo_cfg

# With progress tracking
isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task=Isaac-Cartpole-Direct-v0 --agent=sb3_ppo_cfg --log_interval=100000
```

---

## 📊 **Complete Comparison Table**

| Feature | RSL-RL | RL-Games | SKRL | SB3 |
|---------|--------|----------|------|-----|
| **Performance** | ~1X | ~1X | ~1X | ~1.6X |
| **Multi-Agent** | ❌ | ❌ | ✅ | ❌ |
| **Distributed** | ❌ | ✅ | ✅ | ❌ |
| **Vectorized** | ✅ | ✅ | ✅ | ❌ |
| **ML Frameworks** | PyTorch | PyTorch | PyTorch, JAX | PyTorch |
| **Observation Spaces** | Box | Box, Dict | Box, Dict, Composite | Box |
| **Algorithms** | PPO, AMP | PPO, AMP | PPO, AMP, IPPO, MAPPO | PPO (primary) |
| **Robotics Features** | ✅ Advanced | ✅ High-perf | ✅ Research | ❌ Basic |
| **Ease of Use** | Medium | Medium | Medium | Easy |

**Note**: Performance benchmarks based on Isaac-Humanoid-v0 training on RTX PRO 6000 GPU with 4096 environments for 65.5M steps. SB3 is approximately 1.6X slower than other libraries, not 30X slower as sometimes claimed.

**Note**: While SB3 supports many algorithms (SAC, TD3, DQN, etc.), Isaac Lab officially supports PPO as the primary algorithm. Other SB3 algorithms may work but are not officially tested or supported.

---

## 🎯 **Decision Matrix: When to Choose Each Library**

### **Choose RSL-RL When:**
- ✅ **Robotics locomotion** (quadrupeds, bipeds)
- ✅ **Need symmetry learning**
- ✅ **Want curiosity exploration**
- ✅ **Single-agent robotics**
- ✅ **Research requiring advanced features**

### **Choose RL-Games When:**
- ✅ **High-performance training**
- ✅ **Distributed training** (multi-GPU/node)
- ✅ **Production applications**
- ✅ **Asymmetric actor-critic**
- ✅ **Population-based training**
- ✅ **Maximum performance needed**

### **Choose SKRL When:**
- ✅ **Multi-agent environments**
- ✅ **JAX backend needed**
- ✅ **Complex observation spaces**
- ✅ **Research flexibility**
- ✅ **Custom architectures**
- ✅ **Academic projects**

### **Choose SB3 When:**
- ✅ **Learning RL concepts**
- ✅ **Quick prototyping**
- ✅ **Maximum stability needed**
- ✅ **Small-scale experiments**
- ✅ **Performance not critical**
- ✅ **Easy debugging**

---

## 🚀 **Performance Recommendations**

### **For Maximum Performance:**
1. **RL-Games** (distributed training)
2. **RSL-RL** (robotics-optimized)
3. **SKRL** (PyTorch backend)

### **For Learning/Prototyping:**
1. **SB3** (easiest to use)
2. **SKRL** (good documentation)
3. **RSL-RL** (robotics-focused)

### **For Research:**
1. **SKRL** (multi-agent, JAX)
2. **RSL-RL** (advanced features)
3. **RL-Games** (high-performance)

### **For Production:**
1. **RL-Games** (distributed, stable)
2. **RSL-RL** (robotics-optimized)
3. **SKRL** (flexible)

---

## 💡 **Quick Start Recommendations**

### **Beginner (Learning RL):**
```cmd
isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task=Isaac-Cartpole-Direct-v0 --agent=sb3_ppo_cfg
```

### **Robotics Locomotion:**
```cmd
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Velocity-Flat-Anymal-C-v0 --agent=rsl_rl_ppo_cfg
```

### **High-Performance Training:**
```cmd
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task=Isaac-Velocity-Flat-Anymal-C-v0 --agent=rl_games_ppo_cfg --distributed
```

### **Multi-Agent Research:**
```cmd
isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task=Isaac-MultiAgent-Task-v0 --agent=skrl_ppo_cfg
```

Each library has its strengths and is optimized for different use cases. Choose based on your specific needs: performance, features, ease of use, or research requirements!