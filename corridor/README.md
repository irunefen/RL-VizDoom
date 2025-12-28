# Deadly Corridor: PPO / DQN Training

## Training

```bash
# PPO: 
conda run -n vizdoom python corridor/train_corridor.py \
  --algo ppo \
  --total-timesteps 3000000 \
  --n-envs 8 \
  --doom-skill 5 \
  --tensorboard

# DQN:
conda run -n vizdoom python corridor/train_corridor.py \
  --algo dqn \
  --total-timesteps 3000000 \
  --n-envs 8 \
  --doom-skill 5 \
  --tensorboard

# Shaping adjustments (optional)
# - Penalize losing health
# - Penalize excessive shooting
conda run -n vizdoom python corridor/train_corridor.py \
  --algo ppo \
  --total-timesteps 1000000 \
  --n-envs 32 \
  --doom-skill 5 \
  --health-loss-penalty 1.0 \
  --shoot-penalty 0.01 \
  --tensorboard
```

Main outputs:
- Models and buffers: `runs/{ppo|dqn}_corridor/checkpoints/`
- Best model (according to eval): `runs/{ppo|dqn}_corridor/eval/best_model/`
- Final model: `runs/{ppo|dqn}_corridor/{ppo|dqn}_corridor_final.zip`
- TensorBoard logs: `runs/tb/`

## View progress in TensorBoard

```bash
conda run -n vizdoom tensorboard --logdir=runs/tb
```

Then open in your browser: http://localhost:6006

## Visualize trained agent

```bash
# View the best model playing 5 episodes
conda run -n vizdoom python corridor/visualize_corridor.py \
  --algo ppo \
  --model-path runs/ppo_corridor/eval/best_model/best_model.zip \
  --episodes 5 \
  --deterministic

```

