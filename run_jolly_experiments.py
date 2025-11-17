from types import SimpleNamespace
from train import run_experiments

"""
Runner for Jolly Umulisa experiment sets.

This script defines five hyperparameter sets and calls run_experiments
from train.py to train and evaluate them.

Results will be appended to logs/results.csv using the same pipeline
as the other group scripts.
"""

def main():
    # Five hyperparameter sets (from Jolly's notebook)
    hps = [
        # Set 1 - Baseline (Best documented in notebook)
        {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.10,
        },
        # Set 2 - Higher learning rate
        {
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.10,
        },
        # Set 3 - Lower gamma (focus on short-term rewards)
        {
            "learning_rate": 1e-4,
            "gamma": 0.90,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.10,
        },
        # Set 4 - Extended exploration (slower epsilon decay)
        {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.50,
        },
        # Set 5 - Larger batch size
        {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 128,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.10,
        },
    ]

    # Configuration similar to Kanisa’s "fast" config
    cfg = SimpleNamespace(
        buffer_size=100000,
        learning_starts=10000,
        gradient_steps=1,
        target_update_interval=2000,
        device="auto",
        no_tensorboard=True,
        eval_freq=50000,
        n_eval_episodes=5,
    )

    total_timesteps = 100_000  # each experiment runs for 100k steps

    print(f"Running {len(hps)} experiments for Jolly Umulisa...")
    print(f"Total timesteps per experiment: {total_timesteps}")

    run_experiments(
        hps,
        total_timesteps=total_timesteps,
        policy="CnnPolicy",
        config=cfg,
    )

    print(" Finished running Jolly Umulisa experiments.")
    print("   Check logs/results.csv for recorded average rewards.")


if __name__ == "__main__":
    main()
