# BOOSTER SOCCER SHOWDOWN

![Booster Soccer Showdown Banner](resources/comp.png)  

A fast, extensible robotics soccer competition focused on **generalization across environments**. All the models and datasets are hosted on [Hugging Face](https://huggingface.co/SaiResearch) and the competition is live on [SAI](https://competesai.com/competitions/cmp_xnSCxcJXQclQ).

---

## Compatibility & Requirements

* **Python**: 3.10+
* **Pip**: compatible with environments using Python ≥ 3.10
* **OS**: macOS (Apple Silicon), Linux, and Windows

> Tip: Use a Python 3.10+ environment created via `pyenv`, `conda`, or `uv` for the smoothest experience.

---

## Installation

1. **Clone the repo**

```bash
git clone https://github.com/ArenaX-Labs/booster_soccer_showdown.git
cd booster_soccer_showdown
```

2. **Create & activate a Python 3.10+ environment**

```bash
# any env manager is fine; here are a few options
# --- venv ---
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# --- conda ---
# conda create -n booster-ssl python=3.11 -y && conda activate booster-ssl
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Teleoperation

Booster Soccer Showdown supports keyboard teleop out of the box.

```bash
python booster_control/teleoperate.py \
  --env LowerT1GoaliePenaltyKick-v0 
```

**Default bindings (example):**

* `W/S`: move forward/backward
* `A/D`: move left/right
* `Q/E`: rotate left/right
* `L`: reset commands
* `P`: reset environment

---

⚠️ **Note for macOS and Windows users**
Because different renderers are used on macOS and Windows, you may need to adjust the **position** and **rotation** sensitivity for smooth teleoperation.
Run the following command with the sensitivity flags set explicitly:

```bash
python booster_control/teleoperate.py \
  --env LowerT1GoaliePenaltyKick-v0 \
  --pos_sensitivity 1.5 \
  --rot_sensitivity 1.5
```

(Tune `--pos_sensitivity` and `--rot_sensitivity` as needed for your setup.)

---

## Mimic

The `mimic/` tools let you replay and analyze motions with MuJoCo.

### What’s inside

* `mimic/forward_kinematics.py` — computes derived robot signals (end-effector poses, COM, contacts, velocities, etc.) from motion data using MuJoCo kinematics.
* `mimic/visualize_data.py` — replays a motion in the MuJoCo viewer at a chosen FPS.
* **Model**: the Booster T1 MuJoCo XML is in `mimic/assets/booster_t1/`.

> Motion files are hosted on [Hugging Face](https://huggingface.co/datasets/SaiResearch/booster_dataset) and can be downloaded to use with these scripts. 

---

### 1) Compute kinematics from a motion file

```bash
python mimic/forward_kinematics.py \
  --robot booster_t1 \
  --npz goal_kick.npz \
  --out out/example_motion_fk.npz
```

**Args (common):**

* `--robot` : robot to be loaded (choices: booster_t1 or booster_lower_t1)
* `--npz` : Name of the motion file (`.npz`).
* `--out` : output `.npz` file with enriched signals.

---

### 2) Visualize a motion in MuJoCo

```bash
# Visualize raw motion
python mimic/visualize_data.py \
  --robot booster_t1 \
  --npz goal_kick.npz \
  --fps 30
```
---

## Training

We provide a minimal reinforcement learning pipeline for training agents with **Deep Deterministic Policy Gradient (DDPG)** in the Booster Soccer Showdown environments in the `training_scripts/` folder. The training stack consists of three scripts:

### 1) `ddpg.py`

Defines the **DDPG_FF model**, including:

* Actor and Critic neural networks with configurable hidden layers and activation functions.
* Target networks and soft-update mechanism for stability.
* Training step implementation (critic loss with MSE, actor loss with policy gradient).
* Utility functions for forward passes, action selection, and backpropagation.

---

### 2) `training.py`

Provides the **training loop** and supporting components:

* **ReplayBuffer** for experience storage and sampling.
* **Exploration noise** injection to encourage policy exploration.
* Iterative training loop that:

  * Interacts with the environment.
  * Stores experiences.
  * Periodically samples minibatches to update actor/critic networks.
* Tracks and logs progress (episode rewards, critic/actor loss) with `tqdm`.

---

### 3) `main.py`

Serves as the **entry point** to run training:

* Initializes the Booster Soccer Showdown environment via the **SAI client**.
* Defines a **Preprocessor** to normalize and concatenate robot state, ball state, and environment info into a training-ready observation vector.
* Instantiates a **DDPG_FF model** with custom architecture.
* Defines an **action function** that rescales raw policy outputs to environment-specific action bounds.
* Calls the training loop, and after training, supports:

  * `sai.watch(...)` for visualizing learned behavior.
  * `sai.benchmark(...)` for local benchmarking.

---

### Example: Run Training

```bash
python training_scripts/main.py
```

This will:

1. Build the environment.
2. Initialize the model.
3. Run the training loop with replay buffer and DDPG updates.
4. Launch visualization and benchmarking after training.


### Example: Test pretrained model

```bash
python training_scripts/test.py --env LowerT1KickToTarget-v0
```
