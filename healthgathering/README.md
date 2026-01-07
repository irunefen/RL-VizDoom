# Health Gathering: Training and Visualization

Train reinforcement learning agents on VizDoom Health Gathering.

**Available environments:**
- `VizdoomHealthGathering-v0` - Basic scenario
- `VizdoomHealthGatheringSupreme-v0` - Advanced scenario

**Documentation:** https://vizdoom.farama.org/environments/default/#health-gathering-and-health-gathering-supreme

---

## Main Arguments

### Basic Configuration
- `--algo` (str): Algorithm to use (`ppo` or `dqn`)
- `--scenario` (str): Scenario (`gathering`, `supreme`, or `curriculum` for progressive training)
- `--total-timesteps` (int): Total training timesteps
- `--curriculum-split` (float): Fraction of time in easy environment before switching to hard
- `--n-envs` (int): Number of parallel environments
- `--seed` (int): Random seed for reproducibility
- `--resume` (path): Path to .zip model to continue training

### PPO Hyperparameters
- `--learning-rate` (float): Learning rate
- `--n-steps` (int): Steps per update
- `--batch-size` (int): Batch size
- `--n-epochs` (int): Optimization epochs
- `--gamma` (float): Discount factor
- `--gae-lambda` (float): Lambda for GAE
- `--clip-range` (float): PPO clipping range
- `--ent-coef` (float): Entropy coefficient (exploration)

### DQN Hyperparameters
- `--buffer-size` (int): Replay buffer size
- `--learning-starts` (int): Steps before starting training
- `--train-freq` (int): Update frequency
- `--target-update-interval` (int): Target network update frequency
- `--exploration-fraction` (float): Fraction of time dedicated to exploration
- `--exploration-final-eps` (float): Final epsilon value

### Reward Shaping
- `--no-shaping` (flag): Disable reward shaping
- `--health-gain-reward` (float): Reward for gaining health
- `--health-loss-penalty` (float): Penalty for losing health
- `--poison-potion-penalty` (float): Extra penalty for poison potions
- `--floor-damage-penalty` (float): Penalty for floor damage
- `--damage-threshold` (float): Threshold to distinguish damage types
- `--fast-heal-alpha` (float): Bonus for healing quickly
- `--fast-heal-tau` (float): Temporal decay constant
- `--full-health-bonus` (float): Bonus when reaching 100 HP
- `--full-health-cooldown` (int): Cooldown between full health bonuses

### Penalties
- `--no-stuck-penalty` (flag): Disable stuck penalty
- `--stuck-threshold` (float): Movement threshold to detect stuck
- `--stuck-patience` (int): Consecutive frames before applying penalty
- `--stuck-penalty` (float): Penalty when stuck
- `--no-step-penalty` (flag): Disable step penalty
- `--step-penalty` (float): Penalty per step
- `--no-step-bonus` (flag): Disable movement bonus (supreme)

### Logging
- `--run-name` (str): Custom name for output folder
- `--outdir` (path): Root output directory (default: `runs/`)
- `--tensorboard` (flag): Enable TensorBoard logging
- `--eval-freq` (int): Evaluation frequency
- `--eval-episodes` (int): Episodes per evaluation
- `--checkpoint-freq` (int): Checkpoint save frequency

### Visualization
- `--model-path` (path): Path to trained model (required)
- `--episodes` (int): Number of episodes to show
- `--max-steps` (int): Maximum steps per episode
- `--fps` (float): Visualization speed

---

## Training Examples

### Basic training on Gathering
```bash
conda run -n vizdoom python healthgathering/train_healthgathering.py \
  --algo ppo \
  --scenario gathering \
  --total-timesteps 3000000 \
  --n-envs 8 \
  --tensorboard
```

### Training on Supreme
```bash
conda run -n vizdoom python healthgathering/train_healthgathering.py \
  --algo ppo \
  --scenario supreme \
  --total-timesteps 8000000 \
  --n-envs 8 \
  --tensorboard
```

### Curriculum learning (Gathering → Supreme)
```bash
conda run -n vizdoom python healthgathering/train_healthgathering.py \
  --algo ppo \
  --scenario curriculum \
  --total-timesteps 10000000 \
  --curriculum-split 0.5 \
  --n-envs 8 \
  --tensorboard
```

---

## Visualization Examples

### Basic visualization
```bash
conda run -n vizdoom python healthgathering/visualize_healthgathering.py \
  --scenario gathering \
  --algo ppo \
  --model-path runs/ppo_healthgathering/ppo_healthgathering_final.zip
```

### Visualization with specific configuration
```bash
conda run -n vizdoom python healthgathering/visualize_healthgathering.py \
  --scenario supreme \
  --algo ppo \
  --model-path runs/ppo_supreme/best_model.zip \
  --stuck-threshold 2.5 \
  --stuck-penalty 0.20 \
  --no-step-penalty \
  --episodes 10 \
  --fps 30
```

---

## Outputs

Models are saved in:
```
runs/
├── ppo_healthgathering/
│   ├── checkpoints/gathering/          # Periodic checkpoints
│   ├── eval/gathering/best_model/      # Best model
│   └── ppo_healthgathering_final.zip   # Final model
└── tb/                                 # TensorBoard logs
```

---

## TensorBoard

```bash
conda run -n vizdoom tensorboard --logdir=runs/tb
```

Open in your browser: http://localhost:6006


