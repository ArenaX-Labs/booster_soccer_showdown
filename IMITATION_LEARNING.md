# BOOSTER SOCCER SHOWDOWN (Imitation Learning)

An **imitation learning pipeline** designed for training robust agents to mimic expert demonstrations in the **Booster Soccer Showdown** environments. This repository supports data collection, preprocessing, model training, conversion between frameworks (JAX ↔ PyTorch), and submission-ready model packaging.

To make it easy to train models, the IL models are trained to output joint positions which are then converted to torque using a PD controller before feeding it to the simulator.

---

## ⚙️ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🎮 Data Collection

You can collect teleoperation or scripted demonstration data using:

```bash
python imitation_learning/scripts/collect_data.py \
  --env LowerT1KickToTarget-v0 \
  --data_set_directory path/to/data.npz \
  --renderer mjviewer
```

This script records trajectories in `.npz` format containing observations and actions, rewards.

Data collection automatically includes preprocessing to ensure consistent observation spaces across all environments. This is done through the built-in `Preprocessor` class in the `imitation_learning/scripts/preprocessor.py` script, which augments each observation with projected gravity and base angular velocity derived from robot state information. It can be modified according to the requirement of the user.

---

## 🚀 Training

Train an imitation learning agent (e.g., BC, IQL, HIQL) end-to-end:

```bash
python imitation_learning/train.py \
  --agents bc \
  --dataset_dir path/to/data.npz \
```

Supported agents:

* `bc` — Behavioral Cloning
* `iql` — Implicit Q-Learning
* `gbc` — Goal-Conditioned BC - experimental
* `hiql` — Hierarchical Imitation Q-Learning - experimental
* `gqicl` — Goal-Conditioned IQL - experimental

The checkpoints are saved in the `./exp` folder by default.

---

## 🧪 Evaluation

Test your trained policy in the simulator:

```bash
python imitation_learning/test.py \
  --restore_path path/to/checkpoints \
  --restore_epoch 1000000 \
  --dataset_dir path/to/data.npz \
```

---

## 🧩 Model Conversion (JAX → PyTorch)

If your model was trained in JAX/Flax, convert it to PyTorch for submission on SAI:

```bash
python imitation_learning/scripts/jax2torch.py \
  --pkl path/to/checkpoint.pkl \
  --out path/to/model.pt
```

---

## 📦 Submission

To submit the converted model on SAI:

```bash
python imitation_learning/submission/submit_sai.py 
```

---
