# 🎮 **Isaac Lab Training vs Playing - Complete Guide**

Understanding the difference between `train.py` and `play.py` scripts across all RL libraries:

## 🎯 **Big Picture (applies to all RL libraries)**

### **`train.py`**
- Spawns many parallel envs (big `--num_envs`) for throughput
- Optimizes a policy (gradients on), logs metrics, and **writes checkpoints**
- Often headless for speed (`--headless`), minimal rendering
- Typical knobs: `--task`, `--agent`, `--num_envs`, `--seed`, `--max_iterations`, logging dirs, distributed flags

### **`play.py`**
- **Loads a checkpoint** and runs the policy **without learning** (no gradients)
- Usually runs **few envs** (even 1) so you can watch it, record video, or benchmark
- More viewing/recording options (windowed, FPS, cameras, video writer)
- Typical knobs: `--task`, `--agent`, `--checkpoint`, `--episodes/--steps`, `--video*`, rendering flags, deterministic vs stochastic action

### **Rule of thumb:**
- Train with **large `--num_envs`**, **headless**, **no video**
- Play with **small `--num_envs` (1–16)**, **GUI on** (unless you're batch-validating), **video/FPS** if you want

---

## 🚀 **1. RSL-RL**

### **train.py**
- PPO baseline tightly integrated with Isaac
- Emphasis on speed: big `--num_envs`, headless, frequent checkpointing
- Flags you'll see: `--task`, `--agent`, `--num_envs`, `--seed`, `--max_iterations`, maybe `--logdir`/`--run_dir`

### **play.py**
- Loads a saved `.pt` checkpoint and **just executes**
- Defaults to **deterministic** actions (eval mode), small `--num_envs` (often 1)
- Extras: simple video record / FPS toggles

### **Common gotchas**
- Trying to "play" without `--checkpoint` → error
- Using huge `--num_envs` while rendering → slideshow

### **Examples**
```cmd
:: Train (fast)
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Ant-v0 --agent=rsl_rl_ppo_cfg --num_envs=4096 --headless

:: Play (view one policy)
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Ant-v0 --agent=rsl_rl_ppo_cfg --checkpoint=logs\rsl_rl\ant\model_1500.pt
```

---

## 🎮 **2. RL-Games**

### **train.py**
- High-throughput PPO (legacy Isaac Gym favorite)
- Configs often live in JSON/YAML-like files; the script reads config + launches
- Supports distributed training and Wandb logging

### **play.py**
- Loads the checkpoint from rl_games format and runs eval
- Usually expects the **same config** (or compatible) used in training

### **Common gotchas**
- Wrong config path during play (policy loads but obs mapping is off)
- Trying to render too many envs

### **Examples**
```cmd
:: Train
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task=Isaac-Ant-v0 --agent=rl_games_ppo_cfg --num_envs=4096 --headless

:: Play
isaaclab.bat -p scripts\reinforcement_learning\rl_games\play.py --task=Isaac-Ant-v0 --agent=rl_games_ppo_cfg --checkpoint=logs\rl_games\ant\model.pth
```

---

## 🔬 **3. SKRL**

### **train.py**
- Very configurable (PPO, IPPO, MAPPO, AMP)
- Has nice QoL flags: `--video`, `--video_interval`, `--ml_framework`, `--distributed`
- App/Kit flags baked in (e.g., `--rendering_mode`, `--kit_args`, `--experience`)

### **play.py**
- Same App/Kit surface, but **no learning**
- Focus on **evaluation**: deterministic policy, episode count, optional video capture, can run headless or GUI
- Often supports **`--checkpoint`** (required), and **`--num_envs`** (keep small for visuals)

### **Common gotchas**
- Forgetting `--enable_cameras` when recording from camera sensors
- Passing training-only flags (like optimizer stuff) to play — they'll be ignored

### **Examples**
```cmd
:: Train (PPO, headless)
isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task=Isaac-Ant-v0 --agent=skrl_ppo_cfg --num_envs=4096 --headless

:: Play (record short clip every N steps)
isaaclab.bat -p scripts\reinforcement_learning\skrl\play.py --task=Isaac-Ant-v0 --agent=skrl_ppo_cfg --checkpoint=logs\skrl\ant\agent_1500.pt --video --video_length=600
```

---

## 🛡️ **4. SB3 (Stable Baselines3)**

### **train.py**
- Uses SB3's trainers (PPO/SAC/TD3/etc.)
- Checkpoints usually saved as SB3 `.zip` or Torch `.pth`, depending on wrapper
- Great if you already know SB3's hyperparams + callbacks

### **play.py**
- Loads the SB3 model and runs rollout(s)
- More "Gym-like" evaluation feel; easy to add `VecVideoRecorder` equivalents
- Typically one env with GUI

### **Common gotchas**
- Model/type mismatch (e.g., trained PPO but trying to load as SAC)
- SB3 policies expect the same obs/action shapes as training — don't switch tasks

### **Examples**
```cmd
:: Train with SB3 PPO
isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task=Isaac-Ant-v0 --agent=sb3_ppo_cfg --num_envs=1024 --headless

:: Play SB3 policy
isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py --task=Isaac-Ant-v0 --agent=sb3_ppo_cfg --checkpoint=logs\sb3\ant\model.zip
```

---

## 🎯 **Quick "when to use which play.py"**

- **Show a demo/video** → use the same folder's `play.py` that you trained with (formats match)
- **Batch evaluate** (no GUI) → `--headless`, small `--num_envs`, write logs/CSV
- **Debug visuals** → drop `--headless`, add `--/app/renderer/showFPS=true`, and keep `--num_envs=1`

---

## 📋 **Copy-paste templates you can tweak**

### **RSL-RL – view trained policy with FPS**
```cmd
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py ^
  --task=Isaac-Ant-v0 ^
  --agent=rsl_rl_ppo_cfg ^
  --checkpoint=logs\rsl_rl\ant\model_1500.pt ^
  --/app/renderer/showFPS=true
```

### **RL-Games – headless eval (fast)**
```cmd
isaaclab.bat -p scripts\reinforcement_learning\rl_games\play.py ^
  --task=Isaac-Ant-v0 ^
  --agent=rl_games_ppo_cfg ^
  --checkpoint=logs\rl_games\ant\model.pth ^
  --headless --num_envs=16
```

### **SKRL – one-episode visual check**
```cmd
isaaclab.bat -p scripts\reinforcement_learning\skrl\play.py ^
  --task=Isaac-Ant-v0 ^
  --agent=skrl_ppo_cfg ^
  --checkpoint=logs\skrl\ant\agent_1500.pt ^
  --/app/renderer/showFPS=true
```

### **SB3 – one-episode visual check**
```cmd
isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py ^
  --task=Isaac-Ant-v0 ^
  --agent=sb3_ppo_cfg ^
  --checkpoint=logs\sb3\ant\model.zip ^
  --/app/renderer/showFPS=true
```

---

## 💡 **Key Differences Summary**

| Aspect | Training | Playing |
|--------|----------|---------|
| **Purpose** | Learn policy | Execute policy |
| **Environments** | Many (2048-4096) | Few (1-16) |
| **GUI** | Headless (fast) | GUI (visual) |
| **Checkpoint** | Creates | Loads |
| **Gradients** | On | Off |
| **Video** | Optional | Common |
| **Performance** | Maximize throughput | Maximize visualization |

The key is to train fast with many environments and play slow with few environments for visualization!