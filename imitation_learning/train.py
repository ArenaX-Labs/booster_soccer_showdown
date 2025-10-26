import os
import wandb
import time
import tqdm
import random
import argparse
import numpy as np

from agents import agents

import jax
import jax.numpy as jnp
from utils.buffers import buffers, Dataset
from utils.logging import get_exp_name, setup_wandb
from utils.evaluation import *
from utils.flax_utils import save_agent

def sanitize_metrics(metrics):
    sanitized = {}
    for k, v in metrics.items():
        if isinstance(v, (jnp.ndarray, float, int)):
            sanitized[k] = float(v)
        else:
            sanitized[k] = v
    return sanitized

def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    exp_name = get_exp_name(args.env_name, args.agents)
    setup_wandb(project='booster', group=args.run_group, name=exp_name)

    args.save_dir = os.path.join(args.save_dir, wandb.run.project, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    data = {}
    train_dataset = np.load(args.dataset_dir, allow_pickle=True)
    data["observations"] = np.array(train_dataset["observations"], dtype=np.float32)
    data["actions"] = np.array(train_dataset["actions"], dtype=np.float32)
        
    print(data["observations"].shape)
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

    first_time = time.time()
    last_time = time.time()

    for i in tqdm.tqdm(range(1, args.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        sample_key, key = jax.random.split(key)
        batch = train_dataset.sample(agent_config['batch_size'])

        if args.add_noise:
            obs_noise_key, key = jax.random.split(key)
            noisy_obs = batch['observations'] + args.noise_scale * jax.random.normal(
                obs_noise_key, batch['observations'].shape
            )

            # Replace in batch (JAX PyTree-safe update)
            batch = batch.copy()
            batch['observations'] = noisy_obs

        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % args.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/epoch_time'] = (time.time() - last_time) / args.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()

            train_metrics = sanitize_metrics(train_metrics)
            wandb.log(sanitize_metrics(train_metrics), step=i)

        # Save agent.
        if i % args.save_interval == 0:
            save_agent(agent, args.save_dir, i)

    wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_group', type=str, default='Debug', help='Run group.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--agents', type=str, default="bc", help='Agent to load.')

    # Environment
    parser.add_argument('--env_name', type=str, default='LowerT1GoaliePenaltyKick-v0', help='Environment (dataset) name.')
    parser.add_argument('--dataset_dir', type=str, default="./data/data1.npz", help='Dataset directory.')
    parser.add_argument('--dataset_replace_interval', type=int, default=1000, help='Dataset replace interval.')
    parser.add_argument('--num_datasets', type=int, default=None, help='Number of datasets to use.')

    # Save / restore
    parser.add_argument('--save_dir', type=str, default='exp/', help='Save directory.')
    parser.add_argument('--restore_path', type=str, default=None, help='Restore path.')
    parser.add_argument('--restore_epoch', type=int, default=None, help='Restore epoch.')

    # Training steps and logging
    parser.add_argument('--offline_steps', type=int, default=1000000, help='Number of offline steps.')
    parser.add_argument('--log_interval', type=int, default=5000, help='Logging interval.')
    parser.add_argument('--save_interval', type=int, default=100000, help='Saving interval.')
    parser.add_argument("--add_noise", type=bool, default=False, help="Add noise to observation space?")
    parser.add_argument("--noise_scale", type=int, default=0.01, help="How much noise to add?")

    # Evaluation
    parser.add_argument('--video_episodes', type=int, default=1, help='Number of video episodes for each task.')
    parser.add_argument('--video_frame_skip', type=int, default=3, help='Frame skip for videos.')
    args = parser.parse_args()

    main(args)