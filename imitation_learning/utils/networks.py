
import distrax
import jax.numpy as jnp
import flax.linen as nn

from typing import Any, Sequence, Optional

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')

class MLP(nn.Module):

    hidden_layers: Sequence[int]
    activation: Any = nn.relu
    activate_final: bool = False
    kernel_init: Any = default_init()
    bias_init: Any = None
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        
        for i, size in enumerate(self.hidden_layers):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_layers) or self.activate_final:
                x = self.activation(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                
            if i == len(self.hidden_layers) - 2:
                self.sow('intermediates', 'feature', x)
        return x
    
class GCActor(nn.Module):

    hidden_layers: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2.0
    const_std: bool = False

    def setup(self):
        
        self.actor_net = MLP(self.hidden_layers, activate_final= True, layer_norm= True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

        if not self.const_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
    
    def __call__(self, observations, goal = None, temperature = 1.0):
        
        inputs = [observations]
        if goal is not None:
            inputs.append(goal)
        inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)

        if self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = self.log_std_net(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        return distribution
    
class GCTanhGaussianActor(nn.Module):

    hidden_layers: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2.0
    const_std: bool = False

    def setup(self):
        
        self.actor_net = MLP(self.hidden_layers, activate_final= True, layer_norm= True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

        if not self.const_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
    
    def __call__(self, observations, goal = None, temperature = 1.0):
        
        inputs = [observations]
        if goal is not None:
            inputs.append(goal)
        inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)

        if self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = self.log_std_net(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.Normal(loc=means, scale=jnp.exp(log_stds) * temperature)
        return distribution
    
class GCLaplaceActor(nn.Module):

    hidden_layers: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2.0
    const_std: bool = False

    def setup(self):
        
        self.actor_net = MLP(self.hidden_layers, activate_final= True, layer_norm= True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

        if not self.const_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
    
    def __call__(self, observations, goal = None, temperature = 1.0):
        
        inputs = [observations]
        if goal is not None:
            inputs.append(goal)
        inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)

        if self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = self.log_std_net(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.Laplace(loc=means, scale=jnp.exp(log_stds) * temperature)
        return distribution
    
class GCDetActor(nn.Module):

    hidden_layers: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2

    def setup(self):
        
        self.actor_net = MLP(self.hidden_layers, activate_final= True, layer_norm= True)
        self.action_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
    
    def __call__(self, observations, goal = None, temperature = 1.0):
        
        inputs = [observations]
        if goal is not None:
            inputs.append(goal)
        inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.action_net(outputs)

        return nn.tanh(means) 
       
class GCValue(nn.Module):

    hidden_layers: Sequence[int]
    layer_norm: bool = True

    def setup(self):

        self.value_net = MLP((*self.hidden_layers,1), activate_final=False, layer_norm=self.layer_norm)
    
    def __call__(self, observations, goals = None, actions = None):
        
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        
        if actions is not None:
            inputs.append(actions)
        
        inputs = jnp.concatenate(inputs, axis = -1)
        value_output = self.value_net(inputs)
        return value_output
