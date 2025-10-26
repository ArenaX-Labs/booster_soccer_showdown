import sys, os
import argparse
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import glfw
from stable_baselines3 import TD3, SAC
from huggingface_hub import hf_hub_download

# Make repo root importable without absolute paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

import sai_mujoco  # noqa: F401  # registers envs
from booster_control.t1_utils import LowerT1JoyStick

# ---------- Command→Action wrapper ----------
class CommandActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.lower_control = LowerT1JoyStick(self.base_env)
        # RL policy outputs 3-dim command (vx, vy, yaw_rate) in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def action(self, command):
        # NOTE: relies on base_env private getters; works but is brittle.
        observation = self.base_env._get_obs()
        info = self.base_env._get_info()
        ctrl, _ = self.lower_control.get_actions(command, observation, info)
        return ctrl


def resolve_model_filename(env_id: str) -> str:
    """
    Pick the model filename based on env name.
    - If 'goalie' in env_id (case-insensitive) -> goalie model
    - Else if 'kick' in env_id -> kicker model
    - Else raise a helpful error
    """
    env_lc = env_id.lower()
    if "goalie" in env_lc:
        return "models/td3_goalie_penalty_kick.zip"
    if "kick" in env_lc:
        # Adjust if your repo uses a different filename for the kicker model
        return "models/sac_kick_to_target.zip"
    raise ValueError(
        f"Could not resolve a model for env '{env_id}'. "
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="LowerT1GoaliePenaltyKick-v0",
        help="Gym env ID (e.g., LowerT1GoaliePenaltyKick-v0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device for SB3 (e.g., 'cpu', 'mps', 'cuda')",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of rollout episodes",
    )
    args = parser.parse_args()

    # Resolve model path from env name
    try:
        model_relpath = resolve_model_filename(args.env)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Try to fetch the model from Hugging Face
    try:
        model_file = hf_hub_download(
            repo_id="SaiResearch/booster_soccer_models",
            filename=model_relpath,
            repo_type="model",
        )
    except Exception as e:
        print(
            f"[ERROR] Model file '{model_relpath}' not found in SaiResearch/booster_soccer_models.\n"
            f"Details: {e}"
        )
        sys.exit(1)

    # Build env
    base_env = gym.make(args.env, render_mode="human")
    env = CommandActionWrapper(base_env)

    viewer = getattr(env.base_env.mujoco_renderer, "viewer", None)
    window = getattr(viewer, "window", None) if viewer is not None else None


    if "KickToTarget" in args.env:
        model = SAC.load(model_file, device=args.device)
    else:
        model = TD3.load(model_file, device=args.device)

    # ---- Rollout ----
    for ep in range(args.episodes):
        obs, info = env.reset(seed=42 + ep)
        terminated = truncated = False
        ep_return = 0.0
        print(f"[Episode {ep+1}] Running. Press ESC to stop.")

        viewer = getattr(env.base_env.mujoco_renderer, "viewer", None)
        window = getattr(viewer, "window", None) if viewer is not None else None


        while not (terminated or truncated):

            if window is not None and glfw.get_current_context() is not None:
                # Stop if user hit ESC inside the MuJoCo window
                if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    print("\n[INFO] ESC pressed — stopping and closing.")
                    env.close()
                    sys.exit(0)

                # Stop if user clicked the window close button (red X)
                if glfw.window_should_close(window):
                    print("\n[INFO] Window closed — stopping and exiting.")
                    env.close()
                    sys.exit(0)
                    
            if "KickToTarget" in args.env:
                obs = env.lower_control.get_obs(np.zeros(3), obs, info)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)

        print(f"[Episode {ep+1}] return = {ep_return:.3f}")

    env.close()
    print("[INFO] Environment closed. Exiting.")


if __name__ == "__main__":
    main()
