import jax
import flax
import optax
import jax.numpy as jnp

import copy
from utils.networks import GCActor, GCValue
from typing import Any
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

HIQL_CONFIG_DICT = {
    "agent_name": 'hiql',  # Agent name.
    "lr": 3e-5,  # Learning rate.
    "batch_size": 1024,  # Batch size.
    "actor_hidden_dims": (256, 256),  # Actor network hidden dimensions.
    "value_hidden_dims": (256, 256),  # Value network hidden dimensions.
    "beta": 0.5, # Temperature in AWR.
    "layer_norm": False,  # Whether to use layer normalization.
    "tau": 0.005,
    "expectile_tau": 0.7,  # IQL expectile.
    "subgoal_step": 40, # step k to get subgoal
    "discount": 0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
    "const_std": False,  # Whether to use constant standard deviation for the actor.
    "discrete": False,  # Whether the action space is discrete.
    # Dataset hyperparameters.
    "dataset_class": 'HGCDataset',  # Dataset class name.
    "value_p_curgoal": 0.2,  # Unused (defined for compatibility with GCDataset).
    "value_p_trajgoal": 0.5,  # Unused (defined for compatibility with GCDataset).
    "value_p_randomgoal": 0.3,  # Unused (defined for compatibility with GCDataset).
    "value_geom_sample": True,  # Unused (defined for compatibility with GCDataset).
    "actor_p_curgoal": 0.0,  # Probability of using the current state as the actor goal.
    "actor_p_trajgoal": 1.0,  # Probability of using a future state in the same trajectory as the actor goal.
    "actor_p_randomgoal": 0.0,  # Probability of using a random state as the actor goal.
    "actor_geom_sample": False,  # Whether to use geometric sampling for future actor goals.
    "gc_negative": True,  # Unused (defined for compatibility with GCDataset).
}

class HIQLAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):

        val = jnp.where(adv >= 0, expectile, (1 - expectile))
        return val * (diff**2)

    def value_loss(self, batch, grad_params):
        
        next_v = self.network.select("target_value")(batch["next_observations"], batch["value_goals"])
        
        target_v = batch["rewards"] + self.config["discount"]*batch["masks"]*next_v

        v = self.network.select("target_value")(batch["observations"], batch["value_goals"])

        adv = target_v - v

        v_c = self.network.select("value")(batch["observations"], batch["value_goals"], params=grad_params)

        value_loss = self.expectile_loss(adv, target_v - v_c, self.config["expectile_tau"]).mean()

        value_info = {
            "value_loss" :  value_loss,
            "v_mean" : v_c.mean(),
            "v_max" :  v_c.max(),
            "v_min" :  v_c.min(),
        }

        return value_loss, value_info
        
    def high_actor_loss(self, batch, grad_params, rng=None):

        v = self.network.select("value")(batch["observations"], batch["high_actor_goals"])
        next_v = self.network.select("value")(batch["high_actor_targets"], batch["high_actor_goals"])

        adv = next_v - v
        exp_adv = jnp.minimum(jnp.exp(self.config["beta"]*adv),100)

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['high_actor_targets'])

        actor_loss = -(exp_adv*log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['high_actor_targets']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

        return actor_loss, actor_info
    
    def low_actor_loss(self, batch, grad_params, rng=None):

        v = self.network.select("value")(batch["observations"], batch["low_level_actor_goals"])
        next_v = self.network.select("value")(batch["next_observations"], batch["low_level_actor_goals"])

        adv = next_v - v
        exp_adv = jnp.minimum(jnp.exp(self.config["beta"]*adv),100)

        dist = self.network.select('low_actor')(batch['observations'], batch['low_level_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_adv*log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):

        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        rng, high_actor_rng = jax.random.split(rng)
        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, high_actor_rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        rng, low_actor_rng = jax.random.split(rng)
        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        loss = low_actor_loss + value_loss + high_actor_loss
        info["total_loss"] = loss
        return loss, info
    
    def update_target_network(self, network, function_name):
        new_target_params = jax.tree.map(
            lambda p, tp : p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{function_name}'],
            self.network.params[f'modules_target_{function_name}']
        )

        network.params[f'modules_target_{function_name}'] = new_target_params
    
    @jax.jit
    def update(self, batch):

        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.update_target_network(new_network, "value")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def get_actions(self, observation, goal= None, seed= None, temperature= 1.0):
        
        high_seed, low_seed = jax.random.split(seed)
        subgoal_dist = self.network.select('high_actor')(observation, goal, temperature=temperature)
        sub_goal = subgoal_dist.sample(seed= high_seed)

        dist = self.network.select('low_actor')(observation, sub_goal, temperature=temperature)
        actions = dist.sample(seed= low_seed)
        actions = jnp.clip(actions, -1.0, 1.0)

        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, cfg):

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)


        _cfg = copy.deepcopy(HIQL_CONFIG_DICT)
        _cfg.update(cfg if cfg is not None else {})

        action_dim = ex_actions.shape[-1]
        state_dim = ex_observations.shape[-1]

        high_actor_def = GCActor(
            hidden_layers=_cfg['actor_hidden_dims'],
            action_dim=state_dim,
            const_std = _cfg["const_std"]
        )

        low_actor_def = GCActor(
            hidden_layers=_cfg['actor_hidden_dims'],
            action_dim=action_dim,
            const_std = _cfg["const_std"]
        )

        value_def = GCValue(
            hidden_layers=_cfg["value_hidden_dims"],
            layer_norm=_cfg["layer_norm"]
        )

        network_info = dict(
            high_actor=(high_actor_def, (ex_observations, ex_observations)),
            low_actor=(low_actor_def, (ex_observations, ex_observations)),
            value=(value_def, (ex_observations, ex_observations)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_observations)),
        )

        network = {k: v[0] for k,v in network_info.items()}
        network_args = {k: v[1] for k,v in network_info.items()}

        network_def = ModuleDict(network)
        network_tx = optax.adam(learning_rate=_cfg['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**_cfg))
    