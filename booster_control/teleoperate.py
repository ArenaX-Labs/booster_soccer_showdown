"""
Teleoperate T1 robot in a gymnasium environment using a keyboard.
"""

import time
import argparse
import sai_mujoco  # noqa: F401
import gymnasium as gym
from se3_keyboard import Se3Keyboard
from t1_utils import LowerT1JoyStick

def teleop(env_name: str = "LowerT1GoaliePenaltyKick-v0"):

    env = gym.make(env_name, render_mode="human")
    lower_t1_robot = LowerT1JoyStick(env.unwrapped)

    # Initialize the T1 SE3 keyboard controller with the viewer
    keyboard_controller = Se3Keyboard(renderer=env.unwrapped.mujoco_renderer)

    # Set the reset environment callback
    keyboard_controller.set_reset_env_callback(env.reset)

    # Print keyboard control instructions
    print("\nKeyboard Controls:")
    print(keyboard_controller)

    # Main teleoperation loop
    episode_count = 0
    while True:
        # Reset environment for new episode
        terminated = truncated = False
        observation, info = env.reset()
        episode_count += 1

        print(f"\nStarting episode {episode_count}")

        # Episode loop
        while not (terminated or truncated):
            # Get keyboard input and apply it directly to the environment
            command = keyboard_controller.advance()
            ctrl = lower_t1_robot.get_actions(command, observation, info)
            observation, reward, terminated, truncated, info = env.step(ctrl)
            
            if terminated or truncated:
                break

        # Print episode result
        if info.get("success", False):
            print(f"Episode {episode_count} completed successfully!")
        else:
            print(f"Episode {episode_count} completed without success")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Teleoperate T1 robot in a gymnasium environment.")
    parser.add_argument(
        "--env", type=str, default="LowerT1GoaliePenaltyKick-v0", help="The environment to teleoperate."
    )

    args = parser.parse_args()

    teleop(args.env)