# Defend The Center: PPO/ DQN

## Training

```bash
# PPO:
conda run -n vizdoom python defend_the_center/train_defend_center.py \
  --algo ppo \
  --total-timesteps 3000000 \
  --n-envs 8 \
  --doom-skill 3 \
  --tensorboard

# DQN:
conda run -n vizdoom python defend_the_center/train_defend_center.py \
  --algo dqn \
  --total-timesteps 3000000 \
  --n-envs 8 \
  --doom-skill 3 \
  --tensorboard

# Shaping adjustments (optional)
# - Penalize excessive shooting
conda run -n vizdoom python defend_the_center/train_defend_center.py \
  --algo ppo \
  --total-timesteps 3000000 \
  --n-envs 8 \
  --doom-skill 3 \
  --ammo-waste-penalty 0.01 \
  --tensorboard
```

Main outputs:
- Models and buffers: `runs/{ppo|dqn}_defend_center/checkpoints/`
- Best model: `runs/{ppo|dqn}_defend_center/eval/best_model/`
- Final model: `runs/{ppo|dqn}_defend_center/{ppo|dqn}_defend_center_final.zip`
- Tensorboard logs: `runs/tb/`

## View progress in TensorBoard

```bash
conda run -n vizdoom tensorboard --logdir=runs/tb
```

Then open in your browser: http://localhost:60066


## Visualize trained agent

```bash
# View the best model playing 5 episodes
conda run -n vizdoom python defend_the_center/visualize_defend_center.py \
  --algo ppo \
  --model-path runs/ppo_defend_center/eval/best_model/best_model.zip \
  --episodes 5 \
  --deterministic

```