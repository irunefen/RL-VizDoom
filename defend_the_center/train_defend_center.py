#!/usr/bin/env python

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from vizdoom import gymnasium_wrapper  
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (DummyVecEnv,VecFrameStack,VecTransposeImage,)
import cv2

EnvBuilder = Callable[[], gym.Env]

#Return the discrete action index that triggers ATTACK
def _infer_attack_action_index(env: gym.Env) -> int | None:
    try:
        if not isinstance(env.action_space, spaces.Discrete):
            return None

        unwrapped = env.unwrapped
        game = getattr(unwrapped, "game", None)
        build_action = getattr(unwrapped, "_VizdoomEnv__build_env_action", None)
        if game is None or build_action is None:
            return None

        buttons = list(game.get_available_buttons())
        attack_button_pos = None
        for idx, btn in enumerate(buttons):
            if "ATTACK" in str(btn):
                attack_button_pos = idx
                break
        if attack_button_pos is None:
            return None

        for action in range(int(env.action_space.n)):
            vec = np.asarray(build_action(action)).astype(int)
            if vec.ndim == 1 and vec.shape[0] > attack_button_pos and int(vec[attack_button_pos]) == 1:
                return int(action)
    except Exception:
        return None
    return None

#Penalize excessive ammo usage so the agent learns to aim better
class AmmoRewardShaping(gym.Wrapper):
    def __init__(self, env: gym.Env, ammo_waste_penalty: float = 0.01):
        super().__init__(env)
        self.ammo_waste_penalty = float(ammo_waste_penalty)
        self._prev_ammo: float | None = None
        self._attack_action_index: int | None = _infer_attack_action_index(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_ammo = float(obs["gamevariables"][1]) if isinstance(obs, dict) else None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self._attack_action_index is not None:
            try:
                if int(action) == int(self._attack_action_index):
                    reward -= self.ammo_waste_penalty
            except Exception:
                pass

        return obs, reward, terminated, truncated, info

#Convert Dict(screen, gamevariables) -> (84, 84, 3) uint8 image
class DefendCenterObs84(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)

    def observation(self, obs):

        if not isinstance(obs, dict):
            return obs

        screen = obs["screen"]
        health = float(obs["gamevariables"][0])
        ammo = float(obs["gamevariables"][1])

        # Screen -> grayscale 84x84
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.uint8)

        # Health -> constant plane (0-255)
        health01 = np.clip(health / 100.0, 0.0, 1.0)
        health_plane = np.full((84, 84), int(round(255.0 * health01)), dtype=np.uint8)

        # Ammo -> constant plane 
        ammo01 = np.clip(ammo / 50.0, 0.0, 1.0)
        ammo_plane = np.full((84, 84), int(round(255.0 * ammo01)), dtype=np.uint8)

        return np.stack([gray, health_plane, ammo_plane], axis=-1)


def make_env(
    seed: int,
    doom_skill: int,
    ammo_waste_penalty: float,
) -> EnvBuilder:
    def _init() -> gym.Env:
        env = gym.make("VizdoomDefendCenter-v0")
        env.reset(seed=seed)
        
        game = getattr(env.unwrapped, "game", None)
        if game is not None:
            game.set_doom_skill(doom_skill)

        env = AmmoRewardShaping(env, ammo_waste_penalty=ammo_waste_penalty)
        env = Monitor(env)
        env = DefendCenterObs84(env)
        return env

    return _init

def build_vec_env(
    seed: int,
    doom_skill: int,
    n_envs: int,
    ammo_waste_penalty: float,
) -> DummyVecEnv:
    builders = [
        make_env(
            seed + idx,
            doom_skill,
            ammo_waste_penalty,
        )
        for idx in range(n_envs)
    ]
    vec_env: DummyVecEnv = DummyVecEnv(builders)
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")
    return vec_env

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/DQN on Defend The Center")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"], help="Algorithm to train")
    parser.add_argument("--total-timesteps", type=int, default=3_000_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=256, help="Rollout steps per environment")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--doom-skill", type=int, default=3)
    parser.add_argument("--ammo-waste-penalty", type=float, default=0.01, help="Penalty per shooting action")

    #DQN hyperparameters
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.1)
    parser.add_argument("--exploration-final-eps", type=float, default=0.02)

    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--outdir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", type=str, default="defend_center")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    run_dir = Path(args.outdir) / f"{args.algo}_{args.run_name}"
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    tb_dir = Path(args.outdir) / "tb"
    for path in (checkpoint_dir, eval_dir, tb_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_env = build_vec_env(
        args.seed,
        args.doom_skill,
        args.n_envs,
        args.ammo_waste_penalty,
    )
    eval_env = build_vec_env(
        args.seed + 10_000,
        args.doom_skill,
        1,
        args.ammo_waste_penalty,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix=f"{args.algo}_{args.run_name}",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_dir / "best_model"),
        log_path=str(eval_dir),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    if args.algo == "ppo":
        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.03,
            verbose=1,
            tensorboard_log=str(tb_dir) if args.tensorboard else None,
        )
    else:
        model = DQN(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_final_eps=args.exploration_final_eps,
            gamma=0.99,
            verbose=1,
            tensorboard_log=str(tb_dir) if args.tensorboard else None,
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        tb_log_name=f"{args.algo}_{args.run_name}",
    )

    model.save(run_dir / f"{args.algo}_{args.run_name}_final")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
