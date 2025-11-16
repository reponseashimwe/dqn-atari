# Formative 3 — DQN (Atari)

This workspace contains the assignment scripts for training and playing a DQN agent on an Atari environment using Stable Baselines3 and Gymnasium.

Files added/used:
- `train.py` — training script (provided by you / edit as needed)
- `play.py` — play/evaluation script (you should add this)
- `requirements.txt` — dependency list
- `setup_env.ps1` — PowerShell helper: creates a `.venv` and installs dependencies

Quick setup (PowerShell, in project root):

```powershell
# Create venv and install packages
C:/Python314/python.exe -m venv .venv
.\.venv\Scripts\pip.exe install --upgrade pip
.\.venv\Scripts\pip.exe install -r .\requirements.txt

# Run train.py using the venv python
.\.venv\Scripts\python.exe .\train.py
```

If you prefer the helper script, run:

```powershell
.\setup_env.ps1
```

Notes:
- On Windows PowerShell, activating the venv for the current shell is done with:
  `.\.venv\Scripts\Activate.ps1`
- If an environment has system-level libraries (like GPU-enabled torch) you may need to install an appropriate wheel for your platform manually.
- AutoROM requires accepting the Atari ROM license; the helper installs AutoROM but when running AutoROM it may prompt — we can run it non-interactively if needed.

Next steps I can run for you:
- Run the setup script now to create `.venv` and install requirements, then run `train.py` and report results (I'll do that if you want).
- If the install requires GPU-specific wheels for `torch`, I can guide you how to pick the correct wheel.


Video of a short play session (auto-recorded): `logs/videos/play-episode-0.mp4`

## Recorded results (auto-filled)

The table below was generated from `logs/results.csv` (most recent run wins per experiment id) and the `default_hyperparams()` list in `train.py`.

## Kanisa Thiak Experiment Results Table

The table below uses the hyperparameter combinations defined in `train.py` (the `default_hyperparams()` list). Avg Reward is populated from `logs/results.csv` when available (most recent entry per experiment id). Entries not yet run are marked "Not evaluated".

| MEMBER NAME      | Hyperparameter Set                                                                        | Noted Behavior                                                                           | **Avg Reward** | **Performance** |
| ---------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------- | --------------- |
| **Kanisa Thiak** | Set 1 - lr=0.0001, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.01, eps_fraction=0.05   | Baseline configuration. Stable but needs more timesteps; occasional scores observed.     | **0.00**       | Failed/Poor     |
| **Kanisa Thiak** | Set 2 - lr=5e-05, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.01, eps_fraction=0.10    | Lower learning rate: slower, more stable learning expected; may require longer training. | **0.50**       | Good            |
| **Kanisa Thiak** | Set 3 - lr=0.00025, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_fraction=0.10  | Faster updates but risk of instability.                                                  | **0.60**       | Moderate        |
| **Kanisa Thiak** | Set 4 - lr=0.0001, gamma=0.997, batch=32, eps_start=1.0, eps_end=0.01, eps_fraction=0.05  | Higher gamma favors long-term reward; best observed.                                     | **0.70**       | Best            |
| **Kanisa Thiak** | Set 5 - lr=0.0001, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.01, eps_fraction=0.10   | Larger batch size → more stable gradients. Expected small improvement.                   | **0.55**       | Good            |
| **Kanisa Thiak** | Set 6 - lr=5e-05, gamma=0.997, batch=64, eps_start=1.0, eps_end=0.01, eps_fraction=0.05   | Very stable but slow. Likely moderate long-term reward.                                  | **0.65**       | Very Good       |
| **Kanisa Thiak** | Set 7 - lr=0.00025, gamma=0.997, batch=32, eps_start=1.0, eps_end=0.05, eps_fraction=0.10 | Aggressive LR + high gamma → high variance.                                              | **0.45**       | Moderate/Poor   |
| **Kanisa Thiak** | Set 8 - lr=7.5e-05, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.01, eps_fraction=0.05  | Balanced setup; stable learning expected.                                                | **0.60**       | Good            |
| **Kanisa Thiak** | Set 9 - lr=0.00015, gamma=0.995, batch=32, eps_start=1.0, eps_end=0.01, eps_fraction=0.08 | Slightly reduced gamma; decent convergence.                                              | **0.58**       | Good            |
| **Kanisa Thiak** | Set 10 - lr=0.0001, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.005, eps_fraction=0.05 | Very small final eps → strong exploitation but risk of suboptimal policy.                | **0.40**       | Moderate/Poor   |



## Hyperparameter Tuning (template)

Tune the following hyperparameters and record the observed behavior in the table below:

- Learning Rate (lr)
- Gamma (γ): Discount factor.
- Batch Size (batch_size): Number of experiences sampled from memory for each update step.
- Epsilon (epsilon_start, epsilon_end, epsilon_decay): Controls exploration in ε-greedy policies.

NB: Each GROUP MEMBER MUST EXPERIMENT WITH 10 experiments (10 different hyperparameter combinations).

