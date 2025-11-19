import os
import sys
import random
import argparse
import numpy as np

# Make repo root importable without absolute paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

import gymnasium as gym
import sai_mujoco

import jax
from agents import agents
from utils.buffers import buffers, Dataset
from utils.evaluation import *
from utils.flax_utils import restore_agent
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

def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = {}
    train_dataset = np.load(args.dataset_dir, allow_pickle=True)
    data["observations"] = np.array(train_dataset["observations"], dtype=np.float32)
    data["actions"] = np.array(train_dataset["actions"], dtype=np.float32)
    
    (agent_class, agent_config) = agents[args.agents]

    if agent_config["dataset_class"] != "Dataset":
        buffer_class = buffers[agent_config["dataset_class"]]
        train_dataset = buffer_class(Dataset.create(**data),agent_config)
    else:
        train_dataset = Dataset.create(**data)
        
    example_batch = train_dataset.sample(1)

    agent = agent_class.create(
        args.seed,
        example_batch['observations'],
        example_batch['actions'],
        {},
    )
    agent = restore_agent(agent, args.restore_path, args.restore_epoch)
    agent = supply_rng(agent.get_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    env = gym.make(args.env_name, render_mode="human")
    lower_t1_robot = LowerT1JoyStick(env.unwrapped)
    preprocessor = Preprocessor()
    task_one_hot = get_task_one_hot(args.env_name)
    observation, info = env.reset()
    i = 0
    while True:
        preprocessed_observation = preprocessor.modify_state(observation.copy(), info.copy(), task_one_hot)
        action = agent(observation=preprocessed_observation, temperature=0.0)
        action = np.array(action)
        i += 1
        ctrl = lower_t1_robot.get_torque(observation, action)
        observation, reward, terminated, truncated, info = env.step(ctrl)

        if terminated or truncated:     
            observation, info = env.reset()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_group', type=str, default='Debug', help='Run group.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--agents', type=str, default="bc", help='Agent to load.')

    # Environment
    parser.add_argument('--env_name', type=str, default='LowerT1GoaliePenaltyKick-v0', help='Environment (dataset) name.')
    parser.add_argument('--dataset_dir', type=str, default="./data/data1.npz", help='Dataset directory.')
    parser.add_argument('--eval_episodes', type=int, default=20, help='Number of episodes for each task.')

    # Save / restore
    parser.add_argument('--restore_path', type=str, default='./exp/booster/Debug/`cobot_pick_place_20251021-212615_bc/', help='Save directory.')
    parser.add_argument('--restore_epoch', type=int, default=1000000, help='Epoch checkpoint.')

    args = parser.parse_args()

    main(args)