# 🚀 **Advanced Features - Complete Guide**

Isaac Lab provides advanced features for sophisticated robotics applications, including imitation learning, sim2sim transfer, population-based training, and custom Hydra configurations.

## 🎭 **Imitation Learning (Isaac Lab Mimic)**

Isaac Lab supports imitation learning through the Mimic extension, allowing you to learn from demonstrations.

### **Installation**
```bash
# Install mimic extension
isaaclab.bat -i mimic
```

### **Generate Demonstration Dataset**
```bash
# Generate demonstration dataset
isaaclab.bat -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task=Isaac-Reach-Franka-v0 \
    --num_demos=100 \
    --demo_path=demos/reach_franka.hdf5

# Generate with specific parameters
isaaclab.bat -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task=Isaac-Reach-Franka-v0 \
    --num_demos=200 \
    --demo_length=1000 \
    --demo_path=demos/reach_franka.hdf5 \
    --expert_policy=pretrained \
    --noise_level=0.1
```

### **Train with Imitation Learning**
```bash
# Train with imitation learning
isaaclab.bat -p scripts/imitation_learning/isaaclab_mimic/train.py \
    --task=Isaac-Reach-Franka-v0 \
    --demo_path=demos/reach_franka.hdf5 \
    --agent=mimic_ppo_cfg

# Train with specific parameters
isaaclab.bat -p scripts/imitation_learning/isaaclab_mimic/train.py \
    --task=Isaac-Reach-Franka-v0 \
    --demo_path=demos/reach_franka.hdf5 \
    --agent=mimic_ppo_cfg \
    --num_envs=4096 \
    --max_iterations=2000 \
    --imitation_weight=0.5
```

### **Imitation Learning Configuration**
```python
# Imitation learning configuration
@configclass
class ImitationLearningCfg:
    """Configuration for imitation learning."""
    
    # Dataset parameters
    demo_path: str = "demos/reach_franka.hdf5"
    num_demos: int = 100
    demo_length: int = 1000
    
    # Training parameters
    imitation_weight: float = 0.5
    bc_weight: float = 1.0
    gail_weight: float = 0.1
    
    # Expert policy
    expert_policy: str = "pretrained"  # "pretrained", "human", "scripted"
    noise_level: float = 0.1
    
    # Learning parameters
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 10
```

## 🔄 **Sim2Sim Transfer**

Transfer trained models between different simulation environments or parameters.

### **Basic Sim2Sim Transfer**
```bash
# Train source environment
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Flat-Anymal-C-v0 \
    --agent=rsl_rl_ppo_cfg \
    --num_envs=4096 \
    --max_iterations=1500

# Transfer to target environment
isaaclab.bat -p scripts/sim2sim_transfer/rsl_rl_transfer.py \
    --source_task=Isaac-Velocity-Flat-Anymal-C-v0 \
    --target_task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --source_checkpoint=logs/rsl_rl/anymal_flat/model_1500.pt \
    --target_checkpoint=logs/rsl_rl/anymal_rough/model_transferred.pt
```

### **Advanced Sim2Sim Transfer**
```bash
# Transfer with fine-tuning
isaaclab.bat -p scripts/sim2sim_transfer/rsl_rl_transfer.py \
    --source_task=Isaac-Velocity-Flat-Anymal-C-v0 \
    --target_task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --source_checkpoint=logs/rsl_rl/anymal_flat/model_1500.pt \
    --fine_tune=True \
    --fine_tune_iterations=500 \
    --learning_rate=1e-4
```

### **Sim2Sim Transfer Configuration**
```python
# Sim2Sim transfer configuration
@configclass
class Sim2SimTransferCfg:
    """Configuration for sim2sim transfer."""
    
    # Source and target tasks
    source_task: str = "Isaac-Velocity-Flat-Anymal-C-v0"
    target_task: str = "Isaac-Velocity-Rough-Anymal-C-v0"
    
    # Checkpoint paths
    source_checkpoint: str = "logs/rsl_rl/anymal_flat/model_1500.pt"
    target_checkpoint: str = "logs/rsl_rl/anymal_rough/model_transferred.pt"
    
    # Transfer parameters
    fine_tune: bool = True
    fine_tune_iterations: int = 500
    learning_rate: float = 1e-4
    
    # Transfer strategy
    transfer_strategy: str = "full"  # "full", "partial", "feature_only"
    freeze_layers: list = []  # Layers to freeze during transfer
```

## 🧬 **Population-Based Training (RL-Games)**

Train multiple agents with different hyperparameters simultaneously.

### **Enable Population-Based Training**
```bash
# Enable population-based training
isaaclab.bat -p scripts/reinforcement_learning/rl_games/train.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rl_games_ppo_cfg \
    --pbt=True \
    --pbt_population_size=8 \
    --pbt_num_generations=10

# PBT with specific hyperparameter ranges
isaaclab.bat -p scripts/reinforcement_learning/rl_games/train.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rl_games_ppo_cfg \
    --pbt=True \
    --pbt_population_size=16 \
    --pbt_lr_range=[1e-4, 1e-2] \
    --pbt_entropy_range=[0.001, 0.1]
```

### **Population-Based Training Configuration**
```python
# Population-based training configuration
@configclass
class PopulationBasedTrainingCfg:
    """Configuration for population-based training."""
    
    # PBT parameters
    pbt_enabled: bool = True
    population_size: int = 8
    num_generations: int = 10
    
    # Hyperparameter ranges
    lr_range: tuple = (1e-4, 1e-2)
    entropy_range: tuple = (0.001, 0.1)
    clip_range: tuple = (0.1, 0.3)
    
    # Selection strategy
    selection_strategy: str = "truncation"  # "truncation", "tournament", "random"
    truncation_threshold: float = 0.2
    
    # Mutation strategy
    mutation_strategy: str = "perturb"  # "perturb", "resample", "crossover"
    mutation_rate: float = 0.2
    perturbation_factor: float = 1.2
```

## ⚙️ **Custom Hydra Configuration**

Advanced Hydra configuration management for complex experiments.

### **Custom Config Directory**
```bash
# Use custom config directory
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --config-dir=my_configs \
    --config-name=custom_train

# Override multiple parameters
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    agent.algorithm.learning_rate=0.001 \
    agent.policy.actor_hidden_dims=[512,512,256] \
    agent.algorithm.entropy_coef=0.01 \
    scene.num_envs=2048
```

### **Custom Output Directory**
```bash
# Custom output directory with timestamp
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    hydra.run.dir=./experiments/\${now:%Y-%m-%d_%H-%M-%S}

# Custom output directory with experiment name
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    hydra.run.dir=./experiments/my_experiment_\${now:%Y-%m-%d_%H-%M-%S}
```

### **Hydra Configuration Files**
```yaml
# my_configs/custom_train.yaml
defaults:
  - base_config
  - _self_

# Experiment configuration
experiment:
  name: "my_custom_experiment"
  description: "Custom experiment with advanced features"
  
# Training configuration
training:
  num_envs: 4096
  max_iterations: 2000
  save_interval: 100
  
# Hyperparameter sweeps
sweep:
  learning_rate: [1e-4, 1e-3, 1e-2]
  entropy_coef: [0.001, 0.01, 0.1]
  clip_param: [0.1, 0.2, 0.3]
```

## 🎯 **Multi-Agent Training (SKRL)**

Train multiple agents in the same environment using SKRL's multi-agent support.

### **Multi-Agent Configuration**
```yaml
# Multi-agent configuration
seed: 42

# Multi-agent models
models:
  separate: True  # Separate networks for each agent
  
  # Policy networks for each agent
  policy_agent_1:
    class: GaussianMixin
    network:
      - name: net
        input: STATES
        layers: [256, 256, 128]
        activations: elu
    output: ACTIONS
  
  policy_agent_2:
    class: GaussianMixin
    network:
      - name: net
        input: STATES
        layers: [256, 256, 128]
        activations: elu
    output: ACTIONS

# Multi-agent memory
memory:
  class: RandomMemory
  memory_size: -1

# Multi-agent configuration
agent:
  class: IPPO  # Independent PPO
  rollouts: 16
  learning_epochs: 5
  mini_batches: 4
  
  # Learning parameters
  learning_rate: 1.0e-3
  gamma: 0.99
  lam: 0.95
  
  # PPO parameters
  clip_param: 0.2
  value_loss_coef: 1.0
  entropy_coef: 0.01
```

### **Multi-Agent Training**
```bash
# Train multi-agent environment
isaaclab.bat -p scripts/reinforcement_learning/skrl/train.py \
    --task=Isaac-MultiAgent-Task-v0 \
    --agent=skrl_ippo_cfg \
    --algorithm=IPPO \
    --ml_framework=torch

# Train with MAPPO (Multi-Agent PPO)
isaaclab.bat -p scripts/reinforcement_learning/skrl/train.py \
    --task=Isaac-MultiAgent-Task-v0 \
    --agent=skrl_mappo_cfg \
    --algorithm=MAPPO \
    --ml_framework=torch
```

## 🔬 **Advanced RL Features**

### **Symmetry Learning (RSL-RL)**
```python
# Symmetry learning configuration
@configclass
class SymmetryLearningCfg:
    """Configuration for symmetry learning."""
    
    enabled: bool = True
    symmetry_axis: str = "x"  # Mirror along x-axis
    symmetry_groups: list = ["left_leg", "right_leg"]
    
    # Symmetry parameters
    symmetry_weight: float = 0.1
    symmetry_loss_type: str = "mse"  # "mse", "l1", "cosine"
    
    # Training parameters
    symmetry_update_freq: int = 10
    symmetry_batch_size: int = 256
```

### **Curiosity Exploration (RSL-RL)**
```python
# Curiosity exploration configuration
@configclass
class CuriosityExplorationCfg:
    """Configuration for curiosity-driven exploration."""
    
    enabled: bool = True
    curiosity_weight: float = 0.1
    
    # RND (Random Network Distillation) parameters
    rnd_hidden_dims: list = [256, 256]
    rnd_learning_rate: float = 1e-3
    rnd_update_freq: int = 1
    
    # Intrinsic reward parameters
    intrinsic_reward_scale: float = 1.0
    intrinsic_reward_decay: float = 0.99
```

### **Recurrent Networks**
```python
# Recurrent network configuration
@configclass
class RecurrentNetworkCfg:
    """Configuration for recurrent networks."""
    
    use_rnn: bool = True
    rnn_type: str = "lstm"  # "lstm", "gru", "rnn"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1
    
    # RNN parameters
    rnn_dropout: float = 0.1
    rnn_bidirectional: bool = False
    
    # Training parameters
    rnn_learning_rate: float = 1e-3
    rnn_grad_clip: float = 1.0
```

## 🎮 **Advanced Training Commands**

### **Distributed Training (RL-Games)**
```bash
# Distributed training on multiple GPUs
isaaclab.bat -p scripts/reinforcement_learning/rl_games/train.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rl_games_ppo_cfg \
    --distributed \
    --num_gpus=4 \
    --num_envs=16384

# Distributed training with specific configuration
isaaclab.bat -p scripts/reinforcement_learning/rl_games/train.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rl_games_ppo_cfg \
    --distributed \
    --distributed_backend=nccl \
    --num_gpus=8 \
    --num_envs=32768
```

### **Hyperparameter Sweeps**
```bash
# Hyperparameter sweep with Hydra
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --multirun \
    agent.algorithm.learning_rate=1e-4,1e-3,1e-2 \
    agent.algorithm.entropy_coef=0.001,0.01,0.1 \
    agent.policy.actor_hidden_dims=[256,256],[512,512],[1024,1024]
```

### **Resume Training with Different Parameters**
```bash
# Resume training with modified parameters
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Ant-Direct-v0 \
    --agent=rsl_rl_ppo_cfg \
    --resume \
    --load_run=2024-01-15_10-30-45 \
    --load_checkpoint=1500 \
    agent.algorithm.learning_rate=1e-4 \
    agent.algorithm.entropy_coef=0.005
```

## 💡 **Advanced Features Best Practices**

### **1. Imitation Learning**
- Record diverse demonstrations
- Use appropriate noise levels
- Balance imitation and RL losses
- Validate demonstrations quality

### **2. Sim2Sim Transfer**
- Start with similar environments
- Use fine-tuning for adaptation
- Monitor transfer performance
- Consider domain adaptation techniques

### **3. Population-Based Training**
- Use appropriate population sizes
- Define meaningful hyperparameter ranges
- Monitor population diversity
- Use appropriate selection strategies

### **4. Multi-Agent Training**
- Design appropriate communication
- Use independent or shared networks
- Consider credit assignment
- Monitor individual agent performance

### **5. Advanced RL Features**
- Start with basic features
- Gradually add complexity
- Monitor training stability
- Use appropriate hyperparameters

## 🚀 **Quick Reference Commands**

```bash
# Imitation learning
isaaclab.bat -p scripts/imitation_learning/isaaclab_mimic/train.py --task=<TASK> --demo_path=<DEMO>

# Sim2Sim transfer
isaaclab.bat -p scripts/sim2sim_transfer/rsl_rl_transfer.py --source_task=<SOURCE> --target_task=<TARGET>

# Population-based training
isaaclab.bat -p scripts/reinforcement_learning/rl_games/train.py --task=<TASK> --pbt=True

# Multi-agent training
isaaclab.bat -p scripts/reinforcement_learning/skrl/train.py --task=<TASK> --algorithm=IPPO

# Distributed training
isaaclab.bat -p scripts/reinforcement_learning/rl_games/train.py --task=<TASK> --distributed

# Hyperparameter sweeps
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK> --multirun
```

These advanced features provide powerful capabilities for sophisticated robotics research and applications!
