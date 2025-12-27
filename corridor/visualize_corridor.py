#!/usr/bin/env python

#Visualize an already trained agent
from __future__ import annotations
import argparse
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from vizdoom import gymnasium_wrapper  
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
import cv2

class CorridorObs84(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 255, shape=(84, 84, 2), dtype=np.uint8)

    def observation(self, obs):
        if not isinstance(obs, dict):
            return obs

        screen = obs["screen"]
        health = float(obs["gamevariables"][0])

        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.uint8)

        health01 = np.clip(health / 100.0, 0.0, 1.0)
        health_plane = np.full((84, 84), int(round(255.0 * health01)), dtype=np.uint8)
        return np.stack([gray, health_plane], axis=-1)

#Create the enviroment as on the training
def make_eval_env(render: bool = True, doom_skill: int = 5):
    def _make() -> gym.Env:
        render_mode = "human" if render else None
        env = gym.make("VizdoomCorridor-v0", render_mode=render_mode)

        game = getattr(env.unwrapped, "game", None)
        if game is not None:
            game.set_doom_skill(doom_skill)

        env = CorridorObs84(env)
        return env

    vec_env = DummyVecEnv([_make])
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")
    return vec_env

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained agent on Deadly Corridor")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"], help="Model algorithm")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the .zip model (e.g. runs/ppo_corridor/eval/best_model/best_model.zip)",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic policy (no exploration)")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering (print metrics only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not args.model_path.exists():
        print(f"Error: model not found at {args.model_path}")
        return
    
    print(f"Loading model from: {args.model_path}")
    if args.algo == "ppo":
        model = PPO.load(args.model_path)
    else:
        model = DQN.load(args.model_path)

    env = make_eval_env(render=not args.no_render)

    try:
        model_obs_space = getattr(model, "observation_space", None)
        env_obs_space = getattr(env, "observation_space", None)
        if isinstance(model_obs_space, spaces.Box) and isinstance(env_obs_space, spaces.Box):
            if model_obs_space.shape != env_obs_space.shape:
                env.close()
                return
    except Exception:
        pass
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(args.episodes):
        obs = env.reset()  
        episode_reward = 0.0
        episode_length = 0

        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += float(reward[0])
            episode_length += 1
            if bool(done[0]):
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(
            f"Episode {episode + 1}/{args.episodes}: "
            f"Reward = {episode_reward:.2f}, "
            f"Length = {episode_length}"
        )
    
    env.close()
    
    print("\n" + "*" * 50)
    print(f"Average over {args.episodes} episodes:")
    print(f"  Mean reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"  Mean length: {sum(episode_lengths) / len(episode_lengths):.1f}")
    print(f"  Best reward: {max(episode_rewards):.2f}")
    print("*" * 50)


if __name__ == "__main__":
    main()
