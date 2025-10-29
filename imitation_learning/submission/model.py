from torch import nn
import torch
import os
import numpy as np

def add_weight_noise(model, std=0.01):
    """Add Gaussian noise to all parameters in-place."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn_like(param) * std
            param.add_(noise)
    return model

class BoosterModel(nn.Module):
    """
    Works with your big obs from `modify_state(...)`.
    Only uses the first 30 columns:
      0:12  -> qpos
      12:24 -> qvel
      24:27 -> projected gravity
      27:30 -> base angular velocity
    Everything after 30 is ignored by the gait low-level.

    Internals:
      • phase = [cos(2π*gp), sin(2π*gp)] * (gf > 1e-8)
      • gp <- fmod(gp + dt * gf, 1.0), gf = average(cfg["commands"]["gait_frequency"])
      • Branch-free decimation (trace-safe)
      • PD torque + per-joint clamp
    """
    def __init__(self, model_path):
        super().__init__()

        # constants
        self.register_buffer("default_dof_pos", torch.tensor(
            [-0.2, 0.0, 0.0, 0.4, -0.25, 0.0, -0.2, 0.0, 0.0, 0.4, -0.25, 0.0], dtype=torch.float32))
        self.register_buffer("dof_stiffness", torch.tensor(
            [200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 200.0, 200.0, 200.0, 200.0, 50.0, 50.0], dtype=torch.float32))
        self.register_buffer("dof_damping", torch.tensor(
            [5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0], dtype=torch.float32))
        self.register_buffer("ctrl_min", torch.tensor(
            [-45, -45, -30, -65, -24, -15, -45, -45, -30, -65, -24, -15], dtype=torch.float32))
        self.register_buffer("ctrl_max", torch.tensor(
            [45, 45, 30, 65, 24, 15, 45, 45, 30, 65, 24, 15], dtype=torch.float32))

        self.device = self.default_dof_pos.device
        self.model = torch.jit.load(model_path)
        self.model.eval()

        add_weight_noise(self.model, std=0.01)

    @torch.no_grad()
    def forward(self, obs):
        """
        obs: (N, M>=30) tensor/ndarray from modify_state(...)
        command: (N,3) [lin_x, lin_y, yaw], defaults to zeros if None
        returns: ctrl (N,12)
        """

        obs = torch.tensor(obs, dtype=torch.float32)
        device = self.device
        obs = obs.to(device=device, dtype=torch.float32)
        N, _ = obs.shape

        qpos = obs[:,:12]
        qvel = obs[:,12:24]
        # branch-free decimation

        actions = self.model(obs)[0].clamp(-1.0, 1.0)  # (N,12)
        targets = self.default_dof_pos.expand(N, -1) + actions

        # PD control + clamp
        ctrl = self.dof_stiffness * (targets - qpos) - self.dof_damping * qvel
        ctrl = torch.minimum(torch.maximum(ctrl, self.ctrl_min.expand_as(ctrl)),
                             self.ctrl_max.expand_as(ctrl))
        return ctrl