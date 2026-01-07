from __future__ import annotations

import argparse
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from vizdoom import gymnasium_wrapper
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
import secrets

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

        return np.stack([gray84, health_plane], axis=-1)

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

            # ---- STUCK ----
            if diff < self.threshold:
                self._stuck_count += 1
            else:
                self._stuck_count = 0

            if self._stuck_count >= self.patience:
                reward -= self.penalty

            # ---- ANTI-SPIN  ----
            # Penalize spinning in place: high edge diff but low center diff
            spinning = (diff > (self.threshold * 3.0)) and (edge_diff > center_diff * 1.10)

            if spinning:
                self._spin_count += 1
            else:
                self._spin_count = 0

            spin_patience = 3
            spin_penalty = self.penalty * 2  

            if self._spin_count >= spin_patience:
                reward -= spin_penalty

            info["shaping/stuck_count"] = int(self._stuck_count)
            info["shaping/spin_count"] = int(getattr(self, "_spin_count", 0))
            info["shaping/spinning"] = bool(spinning)
            info["shaping/stuck_penalty"] = float(self.penalty)
            info["shaping/spin_penalty"] = float(spin_penalty)

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

# Health delta reward shaping wrapper
class HealthDeltaRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        health_gain_reward: float = 0.2,
        health_loss_penalty: float = 0.05,
        poison_potion_penalty: float = 0.1,
        floor_damage_penalty: float = 0.02,
        damage_threshold: float = 3.0,
        fast_heal_alpha: float = 0.5,
        fast_heal_tau: float = 200.0,
        full_health_bonus: float = 1.0,
        full_health_cooldown: int = 200,
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

                # fast-heal bonus (more reward if heal happened soon)
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

# Step penalty wrapper
class StepPenalty(gym.Wrapper):
    def __init__(self, env: gym.Env, penalty: float = 0.02):
        super().__init__(env)
        self.penalty = float(penalty)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.penalty
        info["shaping/step_penalty"] = float(self.penalty)
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


def parse_args():
    p = argparse.ArgumentParser(description="Visualize a trained agent on HealthGathering/Supreme with same wrappers as training.")
    p.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--scenario", choices=["gathering", "supreme"], default="gathering")
    p.add_argument("--model-path", type=Path, required=True, help="Path to .zip (SB3) model")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--fps", type=float, default=35.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)

    # shaping (match train script)
    p.add_argument("--no-shaping", action="store_true", help="Disable health-delta shaping.")
    p.add_argument("--health-gain-reward", type=float, default=0.2)
    p.add_argument("--health-loss-penalty", type=float, default=0.05)
    p.add_argument("--poison-potion-penalty", type=float, default=0.1, help="Extra penalty for poison potion damage.")
    p.add_argument("--floor-damage-penalty", type=float, default=0.02, help="Extra penalty for floor damage.")
    p.add_argument("--damage-threshold", type=float, default=3.0, help="Threshold to distinguish floor damage from poison.")

    # advanced shaping
    p.add_argument("--fast-heal-alpha", type=float, default=0.5,
                   help="Extra bonus when healing soon after the previous heal.")
    p.add_argument("--fast-heal-tau", type=float, default=200.0,
                   help="Decay steps for fast-heal bonus (bigger = slower decay).")
    p.add_argument("--full-health-bonus", type=float, default=1.0,
                   help="One-time bonus when reaching full health (100).")
    p.add_argument("--full-health-cooldown", type=int, default=200,
                   help="Cooldown steps between full-health bonuses to avoid farming.")

    # stuck penalty 
    p.add_argument("--no-stuck-penalty", action="store_true")
    p.add_argument("--stuck-threshold", type=float, default=1.5)
    p.add_argument("--stuck-patience", type=int, default=10)
    p.add_argument("--stuck-penalty", type=float, default=0.05)

    # step penalty
    p.add_argument("--no-step-penalty", action="store_true")
    p.add_argument("--step-penalty", type=float, default=0.02)

    # supreme movement bonus
    p.add_argument("--no-step-bonus", action="store_true", help="Enable movement bonus in supreme scenario.")

    return p.parse_args()


def env_id(scenario: str) -> str:
    return "VizdoomHealthGathering-v0" if scenario == "gathering" else "VizdoomHealthGatheringSupreme-v0"


def main():
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    shaping = not args.no_shaping

    # Load model
    model = PPO.load(args.model_path) if args.algo == "ppo" else DQN.load(args.model_path)
    model.learning_rate = args.learning_rate

    # Render env 
    def _make():
        e = gym.make(env_id(args.scenario), render_mode="rgb_array")

        # Health shaping primero
        if shaping:
            e = HealthDeltaRewardShaping(
                e,
                health_gain_reward=args.health_gain_reward,
                health_loss_penalty=args.health_loss_penalty,
                poison_potion_penalty=args.poison_potion_penalty,
                floor_damage_penalty=args.floor_damage_penalty,
                damage_threshold=args.damage_threshold,
                fast_heal_alpha=args.fast_heal_alpha,
                fast_heal_tau=args.fast_heal_tau,
                full_health_bonus=args.full_health_bonus,
                full_health_cooldown=args.full_health_cooldown,
                full_health_value=100.0,
            )

        e = HealthGatheringObs84(e)

        # stuck/spin penalty
        if not args.no_stuck_penalty:
            e = StuckFramePenalty(
                e,
                threshold=args.stuck_threshold,
                patience=args.stuck_patience,
                penalty=args.stuck_penalty,
            )

        #  step penalty 
        if (not args.no_step_penalty) and args.step_penalty > 0:
            e = StepPenalty(e, penalty=args.step_penalty)

        # Supreme movement bonus
        if not args.no_step_bonus:
            e = MovementBonus(e, threshold=2.0, bonus=1)

        return e

    env = DummyVecEnv([_make])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    wait_ms = max(1, int(round(1000.0 / float(args.fps))))
    panel_width = 420 # info panel

    for ep in range(args.episodes):
        seed_ep = secrets.randbits(31) 
        try:
            env.seed(seed_ep)
        except Exception:
            try:
                env.env_method("reset", seed=seed_ep)
            except Exception:
                pass
        obs = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0

        while not done and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)

            done = bool(dones[0])
            if done:
                info0 = infos[0] if infos and isinstance(infos, (list, tuple)) else {}
                print("DONE info:", {k: info0.get(k) for k in ["episode", "TimeLimit.truncated", "final_info", "shaping/health_delta"] if k in info0})
            ep_reward += float(reward[0])
            steps += 1

            frame = env.get_attr("render")[0]()
            if frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                info0 = infos[0] if infos and isinstance(infos, (list, tuple)) else {}

                # ---- RIGHT PANEL  ----
                H, W, _ = bgr.shape
                panel = np.zeros((H, panel_width, 3), dtype=np.uint8)

                y = 30
                line_h = 15

                def draw(txt):
                    nonlocal y
                    cv2.putText(panel, txt, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                    y += line_h

                draw("=== SHAPING INFO ===")

                for k in (
                    "shaping/frame_diff",
                    "shaping/center_diff",
                    "shaping/edge_diff",
                    "shaping/stuck_count",
                    "shaping/stuck_penalty",
                    "shaping/spinning",
                    "shaping/spin_count",
                    "shaping/spin_penalty",
                    "shaping/health_delta",
                    "shaping/step_penalty",
                    "shaping/fast_heal_bonus",
                    "shaping/full_health_bonus",
                    "shaping/poison_potion_damage",
                    "shaping/poison_potion_penalty",
                    "shaping/poison_count",
                    "shaping/floor_damage",
                    "shaping/floor_damage_penalty",
                    "shaping/floor_damage_count",
                    "shaping/move_bonus",
                ):
                    if k in info0:
                        draw(f"{k}: {info0[k]:.4f}" if isinstance(info0[k], float) else f"{k}: {info0[k]}")

                draw("")
                draw("=== GAME ===")

                obs0 = obs[0] 
                health_plane = obs0[-1]

                # HEALTH 
                health01 = float(health_plane[0, 0]) / 255.0
                health = int(round(health01 * 100))
                draw(f"HEALTH: {health}")

                draw(f"STEP: {steps}")
                draw(f"EP REWARD: {ep_reward:.2f}")

                combined = np.hstack([bgr, panel])

                cv2.imshow("HealthGathering", combined)

                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    done = True
                    break

        print(f"Episode {ep+1}/{args.episodes}: steps={steps} reward={ep_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
