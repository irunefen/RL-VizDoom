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

#Penalize health loss and (optionally) shooting to stabilize learning.
class HealthRewardShaping(gym.Wrapper):

    def __init__(self, env: gym.Env, health_loss_penalty: float = 1.0, shoot_penalty: float = 0.01):
        super().__init__(env)
        self.health_loss_penalty = float(health_loss_penalty)
        self.shoot_penalty = float(shoot_penalty)
        self._prev_health: float | None = None
        self._attack_action_index: int | None = _infer_attack_action_index(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_health = float(obs["gamevariables"][0]) if isinstance(obs, dict) else None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, dict) and self._prev_health is not None:
            current_health = float(obs["gamevariables"][0])
            dmg = max(0.0, self._prev_health - current_health)
            if dmg > 0.0:
                reward -= self.health_loss_penalty * dmg
                info["health_loss"] = dmg
            self._prev_health = current_health

        #Small shooting penalty to discourage constant shooting
        #We infer which discrete action maps to ATTACK to avoid hard- oding an index
        if self._attack_action_index is not None:
            try:
                if int(action) == int(self._attack_action_index):
                    reward -= self.shoot_penalty
            except Exception:
                pass

        return obs, reward, terminated, truncated, info

#Add reward shaping to try to avoid the strategy of running to the armor pickup
#To do so we add reward when KILLCOUNT increases and
#Penalize finishing an episode with too few kills
class CombatRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        kill_reward: float = 5.0,
        min_kills: int = 1,
        no_kill_finish_penalty: float = 50.0,
    ):
        super().__init__(env)
        self.kill_reward = float(kill_reward)
        self.min_kills = int(min_kills)
        self.no_kill_finish_penalty = float(no_kill_finish_penalty)
        self._prev_kills: int | None = None
        self._episode_kills: int = 0
        self._kills_supported: bool = True

        self._game = getattr(env.unwrapped, "game", None)
        try:
            from vizdoom import GameVariable

            self._kill_var = GameVariable.KILLCOUNT
        except Exception:
            self._kill_var = None

    def _get_kills(self) -> int | None:
        if self._game is None or self._kill_var is None:
            return None
        try:
            return int(self._game.get_game_variable(self._kill_var))
        except Exception:
            return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        kills = self._get_kills()
        if kills is None:
            self._kills_supported = False
        else:
            self._kills_supported = True
        self._prev_kills = kills
        self._episode_kills = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not self._kills_supported:
            return obs, reward, terminated, truncated, info

        kills = self._get_kills()
        if kills is not None and self._prev_kills is not None:
            delta = max(0, kills - self._prev_kills)
            if delta > 0:
                reward += self.kill_reward * float(delta)
                self._episode_kills += int(delta)
                info["kill_delta"] = int(delta)
        if kills is not None:
            self._prev_kills = kills

        done = bool(terminated) or bool(truncated)
        if done:
            info["episode_kills"] = int(self._episode_kills)
            #If the episode ends and we did not reach the minimum kills we apply a penalty
            if self.min_kills > 0 and self._episode_kills < self.min_kills:
                reward -= self.no_kill_finish_penalty
                info["no_kill_finish_penalty"] = float(self.no_kill_finish_penalty)
            
            #Death penalty: if the agent died (terminated, not just timed out)
            if terminated:
                reward -= 800.0
                info["death_penalty"] = 800.0

        return obs, reward, terminated, truncated, info

#Convert Dict(screen, gamevariables) -> (84, 84, 2) uint8 image
class CorridorObs84(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 255, shape=(84, 84, 2), dtype=np.uint8)

    def observation(self, obs):
        if not isinstance(obs, dict):
            return obs

        screen = obs["screen"]
        health = float(obs["gamevariables"][0])

        # Screen -> grayscale 84x84
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.uint8)

        # Health -> constant plane
        health01 = np.clip(health / 100.0, 0.0, 1.0)
        health_plane = np.full((84, 84), int(round(255.0 * health01)), dtype=np.uint8)

        return np.stack([gray, health_plane], axis=-1)


def make_env(
    seed: int,
    doom_skill: int,
    health_loss_penalty: float,
    shoot_penalty: float,
    kill_reward: float,
    min_kills: int,
    no_kill_finish_penalty: float,
) -> EnvBuilder:

    def _init() -> gym.Env:
        env = gym.make("VizdoomCorridor-v0")
        env.reset(seed=seed)
        
        #Force the requested difficulty (1-5)
        game = getattr(env.unwrapped, "game", None)
        if game is not None:
            game.set_doom_skill(doom_skill)

        env = CombatRewardShaping(
            env,
            kill_reward=kill_reward,
            min_kills=min_kills,
            no_kill_finish_penalty=no_kill_finish_penalty,)

        #Penalize damage taken and constant shooting
        env = HealthRewardShaping(env, health_loss_penalty=health_loss_penalty, shoot_penalty=shoot_penalty)
        env = Monitor(env)
        env = CorridorObs84(env)
        return env

    return _init

def build_vec_env(
    seed: int,
    doom_skill: int,
    n_envs: int,
    health_loss_penalty: float,
    shoot_penalty: float,
    kill_reward: float,
    min_kills: int,
    no_kill_finish_penalty: float,
) -> DummyVecEnv:
    builders = [
        make_env(
            seed + idx,
            doom_skill,
            health_loss_penalty,
            shoot_penalty,
            kill_reward,
            min_kills,
            no_kill_finish_penalty,
        )
        for idx in range(n_envs)
    ]
    vec_env: DummyVecEnv = DummyVecEnv(builders)
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")
    return vec_env

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/DQN on Deadly Corridor")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"], help="Algorithm to train")
    parser.add_argument("--total-timesteps", type=int, default=3_000_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--doom-skill", type=int, default=5)
    parser.add_argument("--health-loss-penalty", type=float, default=3.0, help="Penalty per health point lost")
    parser.add_argument("--shoot-penalty", type=float, default=0.1, help="Per-step penalty when shooting")

    #Combat shaping (enabled by default in the wrapper)
    parser.add_argument("--kill-reward", type=float, default=120.0, help="Reward per kill")
    parser.add_argument("--min-kills", type=int, default=1, help="Minimum kills per episode to avoid the end penalty")
    parser.add_argument("--no-kill-finish-penalty", type=float, default=450.0, help="Penalty when finishing with < min-kills")

    # DQN hyperparameters (ignored by PPO)
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
    parser.add_argument("--run-name", type=str, default="corridor")
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    run_dir = Path(args.outdir) / f"{args.algo}_{args.run_name}"
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    #Shared tensorboard directory to compare algorithms.
    tb_dir = Path(args.outdir) / "tb"
    for path in (checkpoint_dir, eval_dir, tb_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_env = build_vec_env(
        args.seed,
        args.doom_skill,
        args.n_envs,
        args.health_loss_penalty,
        args.shoot_penalty,
        args.kill_reward,
        args.min_kills,
        args.no_kill_finish_penalty,)
    
    eval_env = build_vec_env(
        args.seed + 10_000,
        args.doom_skill,
        1,
        args.health_loss_penalty,
        args.shoot_penalty,
        args.kill_reward,
        args.min_kills,
        args.no_kill_finish_penalty,)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix=f"{args.algo}_{args.run_name}",
        save_replay_buffer=True,
        save_vecnormalize=False,)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_dir / "best_model"),
        log_path=str(eval_dir),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,)

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
            tensorboard_log=str(tb_dir) if args.tensorboard else None,)
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
        tb_log_name=f"{args.algo}_{args.run_name}",)

    model.save(run_dir / f"{args.algo}_{args.run_name}_final")
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
