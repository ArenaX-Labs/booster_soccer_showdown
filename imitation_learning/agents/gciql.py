
import jax
import flax
import optax
import jax.numpy as jnp

import copy
from utils.networks import GCActor, GCValue
from typing import Any
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

GCIQL_CONFIG_DICT = {
    "agent_name": 'gciql',  # Agent name.
    "lr": 3e-4,  # Learning rate.
    "batch_size": 1024,  # Batch size.
    "actor_hidden_dims": (512, 512, 512),  # Actor network hidden dimensions.
    "value_hidden_dims": (256, 256),  # Value network hidden dimensions.
    "beta": 0.3, # Temperature in AWR.
    "layer_norm": True,  # Whether to use layer normalization.
    "tau": 0.005,
    "clip_threshold": 100,
    "expectile_tau": 0.9,  # IQL expectile.
    "discount": 0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
    "const_std": True,  # Whether to use constant standard deviation for the actor.
    "discrete": False,  # Whether the action space is discrete.
    # Dataset hyperparameters.
    "dataset_class": 'GCDataset',  # Dataset class name.
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

class GCIQLAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def expectile_loss(self, diff, expecitile):

        val = jnp.where(diff > 0, expecitile, (1 - expecitile))
        return val * (diff**2)

    @jax.jit
    def value_loss(self, batch, grad_params):
        
        target_q = self.network.select("target_critic")(batch["observations"], batch["value_goals"], batch["actions"])
        v = self.network.select("value")(batch["observations"], batch["value_goals"], params=grad_params)

        value_loss = self.expectile_loss(target_q - v, self.config["expectile_tau"]).mean()

        value_info = {
            "value_loss" :  value_loss,
            "v_mean" : v.mean(),
            "v_max" :  v.max(),
            "v_min" :  v.min(),
        }

        return value_loss, value_info
    
    @jax.jit
    def critic_loss(self, batch, grad_params):

        q = self.network.select("critic")(batch["observations"], batch["value_goals"], batch["actions"])
        target_v = self.network.select("value")(batch["observations"], batch["value_goals"], params= grad_params)

        target_q = batch["rewards"] + self.config["discount"]*batch["masks"]*target_v

        critic_loss = ((target_q - q)**2).mean()

        critic_info = {
            "critic_loss" :  critic_loss,
            "q_mean" : q.mean(),
            "q_min" : q.min(),
            "q_max" : q.max()
        }

        return critic_loss, critic_info
    
    @jax.jit
    def actor_loss(self, batch, grad_params, rng=None):

        v = self.network.select("value")(batch["observations"], batch["actor_goals"])
        q = self.network.select("critic")(batch["observations"], batch["actor_goals"], batch["actions"])

        adv = q - v
        exp_adv = jnp.minimum(jnp.exp(self.config["beta"]*adv),100)

        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
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

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss + value_loss + critic_loss
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
        self.update_target_network(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    def get_actions(self, observation, goal= None, seed= None, temperature= 1.0):

        dist = self.network.select('actor')(observation, goal, temperature=temperature)
        actions = dist.sample(seed= seed)
        actions = jnp.clip(actions, -1.0, 1.0)

        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, cfg):

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)


        _cfg = copy.deepcopy(GCIQL_CONFIG_DICT)
        _cfg.update(cfg if cfg is not None else {})

        action_dim = ex_actions.shape[-1]

        actor_def = GCActor(
            hidden_layers=_cfg['actor_hidden_dims'],
            action_dim=action_dim,
        )

        value_def = GCValue(
            hidden_layers=_cfg["value_hidden_dims"],
            layer_norm=_cfg["layer_norm"]
        )

        critic_def = GCValue(
            hidden_layers=_cfg["value_hidden_dims"],
            layer_norm=_cfg["layer_norm"]
        )

        target_critic_def = copy.deepcopy(critic_def)

        network_info = dict(
            actor=(actor_def, (ex_observations, ex_observations)),
            value=(value_def, (ex_observations, ex_observations)),
            critic=(critic_def, (ex_observations, ex_observations, ex_actions)),
            target_critic=(target_critic_def, (ex_observations, ex_observations, ex_actions))
        )

        network = {k: v[0] for k,v in network_info.items()}
        network_args = {k: v[1] for k,v in network_info.items()}

        network_def = ModuleDict(network)
        # network_tx = optax.adam(learning_rate=_cfg['lr'])
        network_tx = optax.chain(
            optax.clip_by_global_norm(_cfg['clip_threshold']),
            optax.adam(_cfg['lr'])
        )
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**_cfg))
    