"""
Run selected hyperparameter sets (2..10 from default_hyperparams) using train.run_experiments
with a short total_timesteps to obtain quick evaluations and append results to logs/results.csv.
This script calls run_experiments from train.py directly so it uses the same saving/logging.
"""
import argparse
import argparse as _argparse
from train import default_hyperparams, run_experiments

def make_config():
    # mimic the 'fast' preset but shorter for quick evaluation
    cfg = _argparse.Namespace()
    cfg.buffer_size = 100000
    cfg.learning_starts = 10000
    cfg.gradient_steps = 1
    cfg.target_update_interval = 2000
    cfg.eval_freq = 25000
    cfg.n_eval_episodes = 3
    cfg.device = 'auto'
    cfg.no_tensorboard = True
    return cfg

if __name__ == '__main__':
    # Select sets 2..10 (index 1..9)
    all_hps = default_hyperparams()
    selected = all_hps[1:10]

    cfg = make_config()
    total_timesteps = 50000  # short run for quick evaluation

    print(f"Running {len(selected)} experiments (sets 2..10) for {total_timesteps} timesteps each...")
    run_experiments(selected, total_timesteps=total_timesteps, policy='CnnPolicy', config=cfg)
    print("Completed selected experiments.")
