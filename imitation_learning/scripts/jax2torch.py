import pickle
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F

class TorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, activate_final=True, layer_norm=True):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.activate_final = activate_final
        self.layer_norm = layer_norm

        last_dim = input_dim
        for h in hidden_layers:
            self.hidden_layers.append(nn.Linear(last_dim, h))
            if layer_norm:
                self.layer_norms.append(nn.LayerNorm(h, eps=1e-6))
            last_dim = h

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if i + 1 < len(self.hidden_layers) or self.activate_final:
                x = F.relu(x)
                if self.layer_norm:
                    x = self.layer_norms[i](x)
        return x


class TorchGCActor(nn.Module):
    def __init__(self, obs_dim, hidden_layers, action_dim):
        super().__init__()
        self.actor_net = TorchMLP(obs_dim, hidden_layers, activate_final=True, layer_norm=True)
        last_hidden = hidden_layers[-1]
        self.mean_net = nn.Linear(last_hidden, action_dim)
        self.log_std_net = nn.Linear(last_hidden, action_dim)
    
    def forward(self, obs, temperature=1.0):
        feat = self.actor_net(obs)
        mean = self.mean_net(feat)
        log_std = self.log_std_net(feat)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        std = torch.exp(log_std) * temperature
        return mean, std

def load_dense(torch_layer, jax_layer):
    torch_layer.weight.data = torch.tensor(jax_layer["kernel"]).T
    torch_layer.bias.data = torch.tensor(jax_layer["bias"])

def load_layernorm(torch_ln, jax_ln):
    torch_ln.weight.data = torch.tensor(jax_ln["scale"])
    torch_ln.bias.data = torch.tensor(jax_ln["bias"])

def convert(pkl_file, output_path="./converted_model.pt"):

    # ======================================================
    # Load JAX parameters dynamically
    # ======================================================
    with open(pkl_file, "rb") as f:
        jax_params = pickle.load(f)

    params = jax_params["agent"]["network"]["params"]["modules_actor"]

    # ------------------------------------------------------
    # Infer network structure dynamically
    # ------------------------------------------------------
    actor_net = params["actor_net"]

    # Detect Dense layers and their sizes
    dense_layers = [k for k in actor_net.keys() if "Dense" in k]
    dense_layers.sort(key=lambda x: int(x.split("_")[-1]))  # ensure order

    hidden_sizes = [actor_net[d]["bias"].shape[0] for d in dense_layers]
    input_dim = actor_net[dense_layers[0]]["kernel"].shape[0]
    action_dim = params["mean_net"]["bias"].shape[0]

    print(f"Detected MLP structure: input_dim={input_dim}, hidden_layers={hidden_sizes}, action_dim={action_dim}")

    # ======================================================
    # Initialize PyTorch model dynamically
    # ======================================================
    torch_model = TorchGCActor(
        obs_dim=input_dim,
        hidden_layers=hidden_sizes,
        action_dim=action_dim,
    )

    # ======================================================
    # Load weights dynamically
    # ======================================================
    # Load actor_net layers
    for i, dname in enumerate(dense_layers):
        load_dense(torch_model.actor_net.hidden_layers[i], actor_net[dname])

    # Load LayerNorms if they exist
    ln_layers = [k for k in actor_net.keys() if "LayerNorm" in k]
    ln_layers.sort(key=lambda x: int(x.split("_")[-1]))

    for i, lname in enumerate(ln_layers):
        load_layernorm(torch_model.actor_net.layer_norms[i], actor_net[lname])

    # Load output heads
    load_dense(torch_model.mean_net, params["mean_net"])
    load_dense(torch_model.log_std_net, params["log_std_net"])

    print("✅ Successfully loaded all weights dynamically!")

    # ======================================================
    # Test forward pass
    # ======================================================
    obs = torch.randn(1, input_dim)
    mean, std = torch_model(obs)
    print("mean shape:", mean.shape, "std shape:", std.shape)

    # ======================================================
    # 7. Save TorchScript model
    # ======================================================
    scripted_model = torch.jit.trace(torch_model, (torch.randn(1, input_dim),))
    torch.jit.save(scripted_model, output_path)

    print("✅ TorchScript model saved")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--pkl', type=str, required=True, help='Checkpoint pkl file to convert to pytorch')
    parser.add_argument('--out', type=str, required=True, help='Where to the save the output torch model')
    args = parser.parse_args()

    convert(args.pkl, args.out)
