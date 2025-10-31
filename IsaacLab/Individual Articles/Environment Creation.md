
## Isaac Lab Environment Creation Workflow: `isaaclab.bat --new`

### 1. **Command Execution Flow**

When you run `isaaclab.bat --new`, here's what happens internally:

**Step 1: Batch Script Processing**
- The `isaaclab.bat` script detects the `--new` argument (lines 590-608)
- It extracts the Python executable path using `:extract_python_exe`
- Installs template dependencies: `pip install -q -r tools\template\requirements.txt`
- Launches the template generator: `python tools\template\cli.py`

**Step 2: Template Generator Initialization**
- The CLI handler (`tools/template/cli.py`) starts an interactive session
- Determines if Isaac Lab is pip-installed vs. source-installed
- If pip-installed, forces external project creation (internal tasks not supported)

### 2. **User Input Collection Process**

The CLI collects these inputs through interactive prompts:

**Project Type Selection:**
- **External** (recommended): Creates isolated project outside Isaac Lab repo
- **Internal**: Creates task within Isaac Lab repo (only if source-installed)

### 2. **User Input Collection Process**

The CLI collects these inputs through interactive prompts:

**Project Type Selection:**
- **External** (recommended): Creates isolated project outside Isaac Lab repo
- **Internal**: Creates task within Isaac Lab repo (only if source-installed)

**Project Configuration:**
- **Project Path**: Validates it's outside Isaac Lab directory
- **Project Name**: Must be valid Python identifier
- **Workflow Selection**: Shows feature comparison table and prompts for:
  - Direct | single-agent
  - Direct | multi-agent  
  - Manager-based | single-agent

**RL Library Selection:**
- Displays feature comparison table showing:
  - ML frameworks (PyTorch, JAX)
  - Performance characteristics
  - Algorithm support
  - Multi-agent capabilities
  - Distributed training support
- Prompts for RL libraries: `rl_games`, `rsl_rl`, `skrl`, `sb3`
- For each library, prompts for specific algorithms (PPO, AMP, IPPO, MAPPO)

**Note**: Only these 4 algorithms are supported across all libraries in the template generator

### 3. **File Generation and Copying Logic**

The generator (`tools/template/generator.py`) creates the project structure:

**External Project Structure:**
```
project_name/
├── .dockerignore, .flake8, .gitattributes, .gitignore, .pre-commit-config.yaml
├── README.md (generated from template)
├── scripts/
│   ├── [rl_library_name]/ (copied from Isaac Lab's scripts/reinforcement_learning/)
│   ├── list_envs.py (modified to import your extension)
│   ├── zero_agent.py, random_agent.py
├── source/
│   └── [project_name]/
│       ├── config/extension.toml
│       ├── docs/CHANGELOG.rst
│       ├── setup.py, pyproject.toml
│       └── [project_name]/
│           ├── tasks/
│           │   ├── [workflow_name]/
│           │   │   └── [task_name]/
│           │   │       ├── __init__.py
│           │   │       ├── agents/ (RL config files)
│           │   │       ├── [task]_env.py, [task]_env_cfg.py
│           │   │       └── mdp/ (for manager-based workflows)
│           │   └── __init__.py
│           └── ui_extension_example.py
└── .vscode/ (IDE configuration)
```

### 4. **Workflow Differences: Direct vs Manager-based**

**Direct Workflow:**
- **Single-agent**: Inherits from `DirectRLEnv`, uses `DirectRLEnvCfg`
- **Multi-agent**: Inherits from `DirectMARLEnv`, uses `DirectMARLEnvCfg`
- **File Structure**: Creates `[task]_env.py` and `[task]_env_cfg.py`
- **Features**: Supports multi-agent, fundamental/composite spaces
- **Entry Point**: Direct class reference in gym registration

**Manager-based Workflow:**
- **Single-agent only**: Uses `ManagerBasedRLEnv`, `ManagerBasedRLEnvCfg`
- **File Structure**: Creates `[task]_env_cfg.py` + `mdp/` folder with reward/observation functions
- **Features**: Limited to single-agent, Box spaces only
- **Entry Point**: Uses `isaaclab.envs:ManagerBasedRLEnv` with config reference
- **Architecture**: Uses manager pattern (ActionManager, ObservationManager, EventManager, etc.)

### 5. **RL Library Integration**

**Configuration Files Generated:**
- **rl_games**: YAML config files (`rl_games_ppo_cfg.yaml`)
- **rsl_rl**: Python config classes (`rsl_rl_ppo_cfg.py`)
- **skrl**: YAML config files (`skrl_ppo_cfg.yaml`)
- **sb3**: YAML config files (`sb3_ppo_cfg.yaml`)

**Integration Points:**
- Each RL library gets its own script folder copied from `scripts/reinforcement_learning/`
- Scripts contain placeholder: `# PLACEHOLDER: Extension template (do not remove this comment)`
- Generator replaces this with: `import {project_name}.tasks  # noqa: F401`
- Training scripts use Hydra configuration system to load both environment and agent configs

### 6. **Extension System Integration**

**Extension Metadata:**
- `extension.toml` defines package metadata, dependencies, and module structure
- `setup.py` uses this metadata for pip installation
- Extension gets registered as Python module in Isaac Lab's extension system

**Task Registration:**
- Tasks register with Gymnasium using `gym.register()`
- Task IDs follow pattern: `Template-[TaskName]-[Workflow]-v0`
- Entry points reference either direct classes or manager-based environment with config

**Runtime Integration:**
- Extension gets loaded when imported: `import {project_name}.tasks`
- Isaac Lab's extension system discovers and loads the module
- Training scripts can reference tasks by their registered IDs
- Hydra configuration system resolves task and agent configurations

### 7. **File Linking and Dependencies**

**Dependency Chain:**
1. **Extension Installation**: `pip install -e source/{project_name}` makes module discoverable
2. **Task Import**: `import {project_name}.tasks` triggers gym registration
3. **Training Scripts**: Use `--task` argument to reference registered environment
4. **Configuration Resolution**: Hydra loads both env_cfg and agent_cfg based on task/agent names

**Cross-References:**
- Environment configs reference asset configurations from `isaaclab_assets`
- Agent configs reference Isaac Lab RL framework classes
- Training scripts import both environment and agent configurations
- Extension metadata links to Isaac Lab core dependencies

