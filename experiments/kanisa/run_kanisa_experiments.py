from types import SimpleNamespace
from train import run_experiments

"""
Runner for Kanisa Thiak experiment sets.
This script constructs the five hyperparameter sets requested and calls run_experiments
with a moderate "fast"-style config to produce quick evaluation results.

Note: This runs training locally and will append to logs/results.csv.
"""

def main():
    # Five hyperparameter sets as provided by the user
    hps = [
        {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 32, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "exploration_fraction": 0.1},
        {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 32, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "exploration_fraction": 0.1},
        {"learning_rate": 1e-4, "gamma": 0.90, "batch_size": 32, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "exploration_fraction": 0.1},
        {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 32, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "exploration_fraction": 0.5},
        {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 128, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "exploration_fraction": 0.1},
    ]

    # Fast-style config to keep runs reasonably quick while still evaluating
    cfg = SimpleNamespace(
        buffer_size=100000,
        learning_starts=10000,
        gradient_steps=1,
        target_update_interval=2000,
        device='auto',
        no_tensorboard=True,
        eval_freq=50000,
        n_eval_episodes=5,
    )

    # Each experiment will run for 100k timesteps (fast preset style)
    run_experiments(hps, total_timesteps=100000, policy="CnnPolicy", config=cfg)


if __name__ == "__main__":
    main()
