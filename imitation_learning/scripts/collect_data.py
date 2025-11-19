"""
Teleoperate T1 robot in a gymnasium environment using a keyboard.
"""
import os
import sys

# Make repo root importable without absolute paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
sys.path.append(repo_root)

import time
import argparse
import sai_mujoco  # noqa: F401
import gymnasium as gym
import numpy as np
from booster_control.se3_keyboard import Se3Keyboard, Se3Keyboard_Pynput
from booster_control.t1_utils import LowerT1JoyStick
from imitation_learning.scripts.preprocessor import Preprocessor

def get_task_one_hot(env_name):

    if "GoaliePenaltyKick" in env_name:
        task_one_hot = np.array([1.0, 0.0, 0.0])
    elif "ObstaclePenaltyKick" in env_name:
        task_one_hot = np.array([0.0, 1.0, 0.0])
    elif "KickToTarget" in env_name:
        task_one_hot = np.array([0.0, 0.0, 1.0])

    return task_one_hot

def teleop(env_name: str = "LowerT1GoaliePenaltyKick-v0", pos_sensitivity:float = 0.1, rot_sensitivity:float = 1.5, dataset_directory = "./data.npz", renderer="mjviewer"):

    env = gym.make(env_name, render_mode="human", renderer=renderer)
    lower_t1_robot = LowerT1JoyStick(env.unwrapped)
    preprocessor = Preprocessor()

    # Initialize the T1 SE3 keyboard controller with the viewer
    if renderer == "mjviewer":
        keyboard_controller = Se3Keyboard_Pynput(
            renderer=env.unwrapped.mujoco_renderer,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
        )
    else:
        keyboard_controller = Se3Keyboard(
            renderer=env.unwrapped.mujoco_renderer,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
        )

    # Set the reset environment callback
    keyboard_controller.set_reset_env_callback(env.reset)

    # Print keyboard control instructions
    print("\nKeyboard Controls:")
    print(keyboard_controller)

    dataset = {
        "observations" : [],
        "actions" : [],
        "done": []
    }

    # Main teleoperation loop
    episode_count = 0
    task_one_hot = get_task_one_hot(env_name)
    while True:
        # Reset environment for new episode
        terminated = truncated = False
        observation, info = env.reset()
        episode_count += 1

        episode = {
            "observations" : [],
            "actions" : [],
            "done": []
        }

        print(f"\nStarting episode {episode_count}")
        # Episode loop  
        while not (terminated or truncated):

            preprocessed_observation = preprocessor.modify_state(observation.copy(), info.copy(), task_one_hot)
            # Get keyboard input and apply it directly to the environment
            if keyboard_controller.should_quit():
                print("\n[INFO] ESC pressed — exiting teleop.")
                np.savez(dataset_directory, observations=dataset["observations"], actions=dataset["actions"], done = dataset["done"])
                env.close()
                return
            
            command = keyboard_controller.advance()
            ctrl, actions = lower_t1_robot.get_actions(command, observation, info)

            episode["observations"].append(preprocessed_observation)
            episode["actions"].append(actions)
            
            observation, reward, terminated, truncated, info = env.step(ctrl)
            episode["done"].append(terminated)

            if terminated or truncated:
                break
        
        dataset["observations"].extend(episode["observations"])
        dataset["actions"].extend(episode["actions"])
        dataset["done"].extend(episode["done"])

        # Print episode result
        if info.get("success", True):
            print(f"Episode {episode_count} completed successfully!")
        else:
            print(f"Episode {episode_count} completed without success")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Teleoperate T1 robot in a gymnasium environment.")
    parser.add_argument("--env", type=str, default="LowerT1GoaliePenaltyKick-v0", help="The environment to teleoperate.")
    parser.add_argument("--pos_sensitivity", type=float, default=0.1, help="SE3 Keyboard position sensitivity.")
    parser.add_argument("--rot_sensitivity", type=float, default=0.5, help="SE3 Keyboard rotation sensitivity.")
    parser.add_argument("--data_set_directory", type=str, default="./data/dataset_kick.npz", help="SE3 Keyboard rotation sensitivity.")
    parser.add_argument("--renderer", type=str, default="mjviewer", help="Which renderer to use.")

    args = parser.parse_args()

    teleop(args.env, args.pos_sensitivity, args.rot_sensitivity, args.data_set_directory, args.renderer)
