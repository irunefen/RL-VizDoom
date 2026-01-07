from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Literal

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

from vizdoom import gymnasium_wrapper  
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

EnvBuilder = Callable[[], gym.Env]

# Convert observations to (84,84,2) with health plane
class HealthGatheringObs84(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 255, shape=(84, 84, 2), dtype=np.uint8)

    def observation(self, obs):
        if not isinstance(obs, dict):
            return obs

        screen = obs["screen"] 
        if screen.ndim == 3 and screen.shape[-1] == 3:
            gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        else:
            gray = screen.squeeze()

        gray84 = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA).astype(np.uint8)

        gv = obs.get("gamevariables", None)
        if gv is None or len(gv) == 0:
            health01 = 0.0
        else:
            health = float(gv[0])
            health01 = float(np.clip(health / 100.0, 0.0, 1.0))

        health_plane = np.full((84, 84), int(round(255.0 * health01)), dtype=np.uint8)

        stacked = np.stack([gray84, health_plane], axis=-1) 
        return stacked

# Stuck frame penalty wrapper using frame difference
class StuckFramePenalty(gym.Wrapper):
    def __init__(self, env: gym.Env, threshold: float = 1.5, patience: int = 10, penalty: float = 0.05):
        super().__init__(env)
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.penalty = float(penalty)

        self._prev_gray: np.ndarray | None = None
        self._stuck_count: int = 0
        self._spin_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_gray = self._extract_gray84(obs)
        self._stuck_count = 0
        self._spin_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        cur_gray = self._extract_gray84(obs)

        if self._prev_gray is not None and cur_gray is not None:
            prev = self._prev_gray.astype(np.int16)
            cur = cur_gray.astype(np.int16)

            # 1) diff global 
            diff = float(np.mean(np.abs(cur - prev)))

            # 2) Central diff
            c0, c1 = 32, 52  # [32:52) => 20 px
            center_diff = float(np.mean(np.abs(cur[c0:c1, c0:c1] - prev[c0:c1, c0:c1])))

            # 3) Edge diff
            edge_w = 8
            left_diff = float(np.mean(np.abs(cur[:, :edge_w] - prev[:, :edge_w])))
            right_diff = float(np.mean(np.abs(cur[:, -edge_w:] - prev[:, -edge_w:])))
            edge_diff = 0.5 * (left_diff + right_diff)

            # ---- STUCK  ----
            if diff < self.threshold:
                self._stuck_count += 1
            else:
                self._stuck_count = 0

            if self._stuck_count >= self.patience:
                reward -= self.penalty
                info["shaping/stuck_penalty"] = float(self.penalty)

            # ---- ANTI-SPIN  ----
            # Penalize spinning in place: high edge diff but low center diff
            spinning = (diff > (self.threshold * 3.0)) and (edge_diff > center_diff * 1.10)

            if spinning:
                self._spin_count += 1
            else:
                self._spin_count = 0

            spin_patience = 6
            spin_penalty = self.penalty * 2 

            if self._spin_count >= spin_patience:
                reward -= spin_penalty
                info["shaping/spin_penalty"] = float(spin_penalty)

            info["shaping/frame_diff"] = diff
            info["shaping/center_diff"] = center_diff
            info["shaping/edge_diff"] = edge_diff
            info["shaping/stuck_count"] = int(self._stuck_count)
            info["shaping/spin_count"] = int(getattr(self, "_spin_count", 0))
            info["shaping/spinning"] = bool(spinning)

        self._prev_gray = cur_gray
        return obs, reward, terminated, truncated, info


    @staticmethod
    def _extract_gray84(obs) -> np.ndarray | None:
        try:
            if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[:2] == (84, 84):
                return obs[:, :, 0]
        except Exception:
            return None
        return None
    
# Health delta reward shaping (with advanced features)
class HealthDeltaRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        health_gain_reward: float = 0.2,
        health_loss_penalty: float = 0.05,
        poison_potion_penalty: float = 0.1,  # penalty for poison potions (larger damage)
        floor_damage_penalty: float = 0.02,  # penalty for floor damage (smaller continuous)
        damage_threshold: float = 3.0,       # threshold to distinguish floor vs poison
        fast_heal_alpha: float = 0.5,        
        fast_heal_tau: float = 200.0,        
        full_health_bonus: float = 1.0,      # bonus when reaching 100 HP
        full_health_cooldown: int = 200,     # steps cooldown to avoid farming
        full_health_value: float = 100.0,   
    ):
        super().__init__(env)
        self.health_gain_reward = float(health_gain_reward)
        self.health_loss_penalty = float(health_loss_penalty)
        self.poison_potion_penalty = float(poison_potion_penalty)
        self.floor_damage_penalty = float(floor_damage_penalty)
        self.damage_threshold = float(damage_threshold)

        self.fast_heal_alpha = float(fast_heal_alpha)
        self.fast_heal_tau = float(fast_heal_tau)
        self.full_health_bonus = float(full_health_bonus)
        self.full_health_cooldown = int(full_health_cooldown)
        self.full_health_value = float(full_health_value)

        self._prev_health: float | None = None
        self._steps_since_heal: int = 0
        self._was_full: bool = False
        self._full_cd: int = 0
        self._poison_count: int = 0
        self._floor_damage_count: int = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_health = self._extract_health(obs)
        self._steps_since_heal = 0
        self._was_full = False
        self._full_cd = 0
        self._poison_count = 0
        self._floor_damage_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        cur = self._extract_health(obs)

        # tick counters
        self._steps_since_heal += 1
        if self._full_cd > 0:
            self._full_cd -= 1

        if self._prev_health is not None and cur is not None:
            delta = cur - self._prev_health

            # base delta shaping
            if delta > 0:
                reward += self.health_gain_reward * delta

                # fast-heal bonus: more reward if heal happened soon
                bonus_fast = self.fast_heal_alpha * float(np.exp(-self._steps_since_heal / self.fast_heal_tau))
                reward += bonus_fast
                info["shaping/fast_heal_bonus"] = float(bonus_fast)
                self._steps_since_heal = 0

            elif delta < 0:
                damage = -delta
                if damage > self.damage_threshold:
                    # Poisson potion damage
                    penalty = self.poison_potion_penalty * damage
                    reward -= penalty
                    self._poison_count += 1
                    info["shaping/poison_potion_damage"] = float(damage)
                    info["shaping/poison_potion_penalty"] = float(penalty)
                    info["shaping/poison_count"] = self._poison_count
                else:
                    # Floor damage
                    penalty = self.floor_damage_penalty * damage
                    reward -= penalty
                    self._floor_damage_count += 1
                    info["shaping/floor_damage"] = float(damage)
                    info["shaping/floor_damage_penalty"] = float(penalty)
                    info["shaping/floor_damage_count"] = self._floor_damage_count
                
                if self.health_loss_penalty > 0:
                    reward -= self.health_loss_penalty * damage

            info["shaping/health_delta"] = float(delta)

            # full health bonus 
            is_full = (cur >= self.full_health_value)
            if is_full and (not self._was_full) and self._full_cd == 0:
                reward += self.full_health_bonus
                info["shaping/full_health_bonus"] = float(self.full_health_bonus)
                self._full_cd = self.full_health_cooldown

            self._was_full = bool(is_full)

        self._prev_health = cur
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _extract_health(obs) -> float | None:
        try:
            if isinstance(obs, dict) and "gamevariables" in obs and len(obs["gamevariables"]) > 0:
                return float(obs["gamevariables"][0])
        except Exception:
            return None
        return None

# Step penalty wrapper (to encourage exploration)
class StepPenalty(gym.Wrapper):
    def __init__(self, env: gym.Env, penalty: float = 0.02):
        super().__init__(env)
        self.penalty = float(penalty)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.penalty
        info["shaping/step_penalty"] = self.penalty
        return obs, reward, terminated, truncated, info
    
# Movement bonus wrapper (to encourage movement in supreme)
class MovementBonus(gym.Wrapper):
    def __init__(self, env, threshold=2.0, bonus=0.02):
        super().__init__(env)
        self.threshold = threshold
        self.bonus = bonus
        self.prev_frame = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_frame = obs[..., 0].astype("float32")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        cur = obs[..., 0].astype("float32")
        diff = abs(cur - self.prev_frame).mean()

        if diff > self.threshold:
            reward += self.bonus
            info["shaping/move_bonus"] = self.bonus

        self.prev_frame = cur
        return obs, reward, terminated, truncated, info

def make_env(
    env_id: str,
    seed: int,
    idx: int,
    stuck_penalty_enabled: bool,
    stuck_threshold: float,
    stuck_patience: int,
    stuck_penalty: float,
    shaping: bool,
    health_gain_reward: float,
    health_loss_penalty: float,    
    poison_potion_penalty: float,
    floor_damage_penalty: float,
    damage_threshold: float,    
    step_penalty_enabled: bool,
    step_penalty: float,
    fast_heal_alpha: float,
    fast_heal_tau: float,
    full_health_bonus: float,
    full_health_cooldown: int,
    full_health_value: float,
    step_bonus:bool,
) -> EnvBuilder:
    def _init():
        env = gym.make(env_id)

        # health delta shaping 
        if shaping:
            env = HealthDeltaRewardShaping(
                env,
                health_gain_reward=health_gain_reward,
                health_loss_penalty=health_loss_penalty,
                poison_potion_penalty=poison_potion_penalty,
                floor_damage_penalty=floor_damage_penalty,
                damage_threshold=damage_threshold,
                fast_heal_alpha=fast_heal_alpha,
                fast_heal_tau=fast_heal_tau,
                full_health_bonus=full_health_bonus,
                full_health_cooldown=full_health_cooldown,
                full_health_value=full_health_value,
            )

        env = HealthGatheringObs84(env)

        # Stuck frame penalty 
        if stuck_penalty_enabled:
            env = StuckFramePenalty(
                env,
                threshold=stuck_threshold,
                patience=stuck_patience,
                penalty=stuck_penalty,
            )

        # Step penalty
        if step_penalty_enabled and step_penalty > 0:
            env = StepPenalty(env, penalty=step_penalty)
        
        # Supreme movement bonus
        if step_bonus:
            env = MovementBonus(env, threshold=2.0, bonus=1) 

        env = Monitor(env)
        env.reset(seed=seed + idx)
        return env

    return _init

def build_vec_env(
    env_id: str,
    n_envs: int,
    seed: int,
    stuck_penalty_enabled: bool,
    stuck_threshold: float,
    stuck_patience: int,
    stuck_penalty: float,
    shaping: bool,
    health_gain_reward: float,
    health_loss_penalty: float,
    poison_potion_penalty: float,
    floor_damage_penalty: float,
    damage_threshold: float,
    step_penalty_enabled: bool,
    step_penalty: float,
    fast_heal_alpha: float,
    fast_heal_tau: float,
    full_health_bonus: float,
    full_health_cooldown: int,
    full_health_value: float,
    step_bonus: bool,
):
    env_fns = [
        make_env(
            env_id, seed, i,
            stuck_penalty_enabled, stuck_threshold, stuck_patience, stuck_penalty,
            shaping, health_gain_reward, health_loss_penalty,
            poison_potion_penalty, floor_damage_penalty, damage_threshold,
            step_penalty_enabled, step_penalty,
            fast_heal_alpha, fast_heal_tau,
            full_health_bonus, full_health_cooldown, full_health_value, step_bonus
        )
        for i in range(n_envs)
    ]
    vec = DummyVecEnv(env_fns)
    vec = VecFrameStack(vec, n_stack=4)
    vec = VecTransposeImage(vec)
    return vec


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO/DQN on HealthGathering (and Supreme).")

    p.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--scenario", choices=["gathering", "supreme", "curriculum"], default="gathering",
                   help="Which scenario to train. 'curriculum' trains gathering then supreme continuing from the same weights.")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--curriculum-split", type=float, default=0.6,
                   help="If scenario=curriculum, fraction of total timesteps spent on the easier env before switching.")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=2)

    # PPO hyperparameters
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.01)

    # DQN hyperparameters
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--target-update-interval", type=int, default=10_000)
    p.add_argument("--exploration-fraction", type=float, default=0.2)
    p.add_argument("--exploration-final-eps", type=float, default=0.05)

    # shaping
    p.add_argument("--no-shaping", action="store_true", help="Disable health-delta shaping.")
    p.add_argument("--health-gain-reward", type=float, default=0.2, help="Multiplier for +delta_health.")
    p.add_argument("--health-loss-penalty", type=float, default=0.05, help="Multiplier for -delta_health (general).")
    p.add_argument("--poison-potion-penalty", type=float, default=0.1, help="Extra penalty for poison potion damage (large damage).")
    p.add_argument("--floor-damage-penalty", type=float, default=0.02, help="Extra penalty for floor damage (small continuous).")
    p.add_argument("--damage-threshold", type=float, default=10.0, help="Threshold to distinguish floor damage from poison potion.")

    # advanced shaping
    p.add_argument("--fast-heal-alpha", type=float, default=0.5,
                help="Extra bonus when healing soon after the previous heal.")
    p.add_argument("--fast-heal-tau", type=float, default=200.0,
                help="Decay steps for fast-heal bonus (bigger = slower decay).")
    p.add_argument("--full-health-bonus", type=float, default=1.0,
                help="One-time bonus when reaching full health (100).")
    p.add_argument("--full-health-cooldown", type=int, default=200,
                help="Cooldown steps between full-health bonuses.")


    # logging / output
    p.add_argument("--run-name", type=str, default=None, help="Overrides the default run name.")
    p.add_argument("--outdir", type=Path, default=Path("runs"), help="Root output folder (default: runs/)")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)

    # customize options
    # stuck penalty (anti-wall / anti-stuck)
    p.add_argument("--no-stuck-penalty", action="store_true", help="Disable stuck (frame-diff) penalty.")
    p.add_argument("--stuck-threshold", type=float, default=1.5,
                   help="If mean(|frame_t - frame_{t-1}|) < threshold, counts as stuck step.")
    p.add_argument("--stuck-patience", type=int, default=4,
                   help="Consecutive stuck steps before penalty starts applying.")
    p.add_argument("--stuck-penalty", type=float, default=0.05,
                   help="Penalty subtracted each step while stuck.")
    
    # step penalty (exploration)
    p.add_argument("--no-step-penalty", action="store_true", help="Disable per-step penalty (exploration).")
    p.add_argument("--step-penalty", type=float, default=0.02,
                   help="Penalty subtracted each step to discourage passive survival.")
    
    # supreme movement bonus
    p.add_argument("--no-step-bonus", action="store_true", help="Enable movement bonus in supreme scenario.")
    
    # resume
    p.add_argument("--resume", type=Path, default=None, help="Path to an existing .zip model to continue training from.")

    return p.parse_args()


def default_env_id(which: Literal["gathering", "supreme"]) -> str:
    if which == "gathering":
        return "VizdoomHealthGathering-v0"
    return "VizdoomHealthGatheringSupreme-v0"


def main():
    args = parse_args()
    set_random_seed(args.seed)

    shaping = not args.no_shaping

    scenario_tag = args.scenario
    if args.run_name is None:
        base_run_name = {
            "gathering": "healthgathering",
            "supreme": "healthgathering_supreme",
            "curriculum": "health_curriculum",
        }[scenario_tag]
    else:
        base_run_name = args.run_name

    outdir = Path(args.outdir)
    run_dir = outdir / f"{args.algo}_{base_run_name}"
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    tb_log = str(outdir / "tb") if args.tensorboard else None

    def _make_train(env_id: str, stage_name: str):
        if stage_name == "supreme":
            stage_step_penalty = min(args.step_penalty, 0.01)  # o 0.0
        else:
            stage_step_penalty = args.step_penalty

        return build_vec_env(
            env_id=env_id,
            n_envs=args.n_envs,
            seed=args.seed,
            stuck_penalty_enabled=not args.no_stuck_penalty,
            stuck_threshold=args.stuck_threshold,
            stuck_patience=args.stuck_patience,
            stuck_penalty=args.stuck_penalty,
            shaping=shaping,
            health_gain_reward=args.health_gain_reward,
            health_loss_penalty=args.health_loss_penalty,
            poison_potion_penalty=args.poison_potion_penalty,
            floor_damage_penalty=args.floor_damage_penalty,
            damage_threshold=args.damage_threshold,
            step_penalty_enabled=not args.no_step_penalty,
            step_penalty=stage_step_penalty,
            fast_heal_alpha=args.fast_heal_alpha,
            fast_heal_tau=args.fast_heal_tau,
            full_health_bonus=args.full_health_bonus,
            full_health_cooldown=args.full_health_cooldown,
            full_health_value=100.0,
            step_bonus=not args.no_step_bonus,
        )

    def _make_eval(env_id: str):
        env = DummyVecEnv([make_env(
            env_id,
            args.seed + 10_000,
            0,
            False,  # no stuck penalty in eval
            args.stuck_threshold,
            args.stuck_patience,
            args.stuck_penalty,
            shaping, 
            args.health_gain_reward,
            args.health_loss_penalty,
            args.poison_potion_penalty,
            args.floor_damage_penalty,
            args.damage_threshold,
            False,   # no step penalty in eval
            0.0,
            args.fast_heal_alpha,
            args.fast_heal_tau,
            args.full_health_bonus,
            args.full_health_cooldown,
            100.0,
            not args.no_step_bonus,
        )])
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
        return env

    # Curriculum plan
    if args.scenario == "curriculum":
        first_env = default_env_id("gathering")
        second_env = default_env_id("supreme")
        t_first = int(round(args.total_timesteps * float(np.clip(args.curriculum_split, 0.0, 1.0))))
        t_second = int(args.total_timesteps) - t_first
        plan = [("gathering", first_env, t_first), ("supreme", second_env, t_second)]
    else:
        env_id = default_env_id("gathering" if args.scenario == "gathering" else "supreme")
        plan = [(args.scenario, env_id, int(args.total_timesteps))]

    model = None
    last_stage_path = None

    for stage_name, env_id, stage_steps in plan:
        if stage_steps <= 0:
            continue

        print(f"\n=== Stage: {stage_name} | env={env_id} | steps={stage_steps} ===")

        train_env = _make_train(env_id, stage_name)
        eval_env = _make_eval(env_id)

        # callbacks 
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.n_envs),
            save_path=str(ckpt_dir / stage_name),
            name_prefix=f"{args.algo}_{stage_name}",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(eval_dir / stage_name / "best_model"),
            log_path=str(eval_dir / stage_name / "logs"),
            eval_freq=max(1, args.eval_freq // args.n_envs),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )

        if model is None:
            # resume
            if args.resume is not None:
                if not args.resume.exists():
                    raise FileNotFoundError(f"Resume model not found: {args.resume}")
                print(f"Resuming from: {args.resume}")
                if args.algo == "ppo":
                    model = PPO.load(args.resume, env=train_env)

                    # FORCE LR when resuming 
                    if args.algo == "ppo":
                        new_lr = float(args.learning_rate)
                        for pg in model.policy.optimizer.param_groups:
                            pg["lr"] = new_lr
                        # also override the schedule used for logging / updates
                        model.learning_rate = new_lr
                        model.lr_schedule = lambda _: new_lr
                        print(f"[resume] Forced PPO lr = {new_lr}")
                else:
                    model = DQN.load(args.resume, env=train_env)
            else:
                if args.algo == "ppo":
                    model = PPO(
                        policy="CnnPolicy",
                        env=train_env,
                        learning_rate=args.learning_rate,
                        n_steps=args.n_steps,
                        batch_size=args.batch_size,
                        n_epochs=args.n_epochs,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                        clip_range=args.clip_range,
                        ent_coef=args.ent_coef,
                        verbose=1,
                        tensorboard_log=tb_log,
                    )
                else:
                    model = DQN(
                        policy="CnnPolicy",
                        env=train_env,
                        learning_rate=args.learning_rate,
                        buffer_size=args.buffer_size,
                        learning_starts=args.learning_starts,
                        train_freq=args.train_freq,
                        target_update_interval=args.target_update_interval,
                        exploration_fraction=args.exploration_fraction,
                        exploration_final_eps=args.exploration_final_eps,
                        gamma=args.gamma,
                        verbose=1,
                        tensorboard_log=tb_log,
                    )
        else:
            # curriculum: set new env
            model.set_env(train_env)
            if stage_name == "supreme" and args.algo == "ppo":
                new_lr = min(args.learning_rate, 1e-4)
                for pg in model.policy.optimizer.param_groups:
                    pg["lr"] = new_lr
                print(f"[stage=supreme] Set PPO optimizer lr = {new_lr}")

        model.learn(
            total_timesteps=stage_steps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
            tb_log_name=f"{args.algo}_{base_run_name}",
            reset_num_timesteps=False,
        )

        # save stage checkpoint
        last_stage_path = run_dir / f"{args.algo}_{base_run_name}_{stage_name}_final"
        model.save(last_stage_path)

        train_env.close()
        eval_env.close()
    # final alias
    if last_stage_path is not None:
        final_path = run_dir / f"{args.algo}_{base_run_name}_final"
        try:
            # save again under final alias for convenience
            model.save(final_path)
        except Exception:
            pass

    print(f"\nDone. Outputs in: {run_dir}")


if __name__ == "__main__":
    main()