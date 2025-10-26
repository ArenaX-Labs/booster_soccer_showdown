
import jax
import flax
import optax
import jax.numpy as jnp

import copy
from utils.networks import GCActor, GCDetActor
from typing import Any
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

BC_CONFIG_DICT = {
    "agent_name": 'bc',  # Agent name.
    "lr": 1e-4,  # Learning rate.
    "batch_size": 1024,  # Batch size.
    "actor_hidden_dims": (256, 256, 256),  # Actor network hidden dimensions.
    "discount": 0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
    "clip_threshold": 100.0,
    "const_std": True,  # Whether to use constant standard deviation for the actor.
    "discrete": False,  # Whether the action space is discrete.
    # Dataset hyperparameters.
    "dataset_class": 'Dataset',  # Dataset class name.
    "gc_negative": True,  # Unused (defined for compatibility with GCDataset).
}

class BCAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def actor_loss(self, batch, grad_params, rng=None):

        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -log_prob.mean()

        actor_info = {
            'actor_loss': actor_loss,
            'bc_log_prob': log_prob.mean(),
        }

        actor_info.update(
            {
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        )

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):

        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):

        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    def get_actions(self, observation, seed= None, temperature= 1.0):

        dist = self.network.select('actor')(observation, temperature=temperature)
        actions = dist.sample(seed= seed)
        actions = jnp.clip(actions, -1.0, 1.0)

        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, cfg):

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)


        _cfg = copy.deepcopy(BC_CONFIG_DICT)
        _cfg.update(cfg if cfg is not None else {})

        action_dim = ex_actions.shape[-1]

        actor_def = GCActor(
            hidden_layers=_cfg['actor_hidden_dims'],
            action_dim=action_dim,
        )

        network_info = dict(
            actor=(actor_def, (ex_observations))
        )

        network = {k: v[0] for k,v in network_info.items()}
        network_args = {k: v[1] for k,v in network_info.items()}

        network_def = ModuleDict(network)
        network_tx = optax.chain(
            optax.clip_by_global_norm(_cfg['clip_threshold']),
            optax.adam(_cfg['lr'])
        )
        # network_tx = optax.adam(learning_rate=_cfg['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**_cfg))
    
class BCMSEAgent(BCAgent):

    @jax.jit
    def actor_loss(self, batch, grad_params, rng=None):

        actions = self.network.select('actor')(batch['observations'], params=grad_params)

        actor_loss = optax.huber_loss(actions, batch['actions']).mean()

        actor_info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, actor_info
    
    def get_actions(self, observation, seed= None):

        actions = self.network.select('actor')(observation)
        return actions
    
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, cfg):

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)


        _cfg = copy.deepcopy(BC_CONFIG_DICT)
        _cfg.update(cfg if cfg is not None else {})

        action_dim = ex_actions.shape[-1]

        actor_def = GCDetActor(
            hidden_layers=_cfg['actor_hidden_dims'],
            action_dim=action_dim,
        )

        network_info = dict(
            actor=(actor_def, (ex_observations))
        )

        network = {k: v[0] for k,v in network_info.items()}
        network_args = {k: v[1] for k,v in network_info.items()}

        network_def = ModuleDict(network)
        network_tx = optax.chain(
            optax.clip_by_global_norm(_cfg['clip_threshold']),
            optax.adam(_cfg['lr'])
        )
        # network_tx = optax.adam(learning_rate=_cfg['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**_cfg))