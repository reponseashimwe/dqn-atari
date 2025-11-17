import os
import csv
import argparse
import gymnasium as gym
import ale_py
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env


# Directory for logs and saved models
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def make_env(use_cnn=True):
    """Create Atari environment with proper preprocessing"""
    # Use Stable Baselines3's built-in Atari environment creation
    # This handles all the preprocessing, frame stacking, and image space correctly
    env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=42)
    
    # For CNN policy, the environment should already be properly configured
    # If you need to use MLP policy instead, you can flatten the observations
    if not use_cnn:
        from stable_baselines3.common.vec_env import VecFlatten
        env = VecFlatten(env)
    
    return env


def run_experiments(hyperparams_list, total_timesteps=3000000, policy="CnnPolicy", config=None):  #  FIXED: 3M timesteps
    results_file = os.path.join(LOG_DIR, "results.csv")
    header = [
        "experiment_id", "policy", "learning_rate", "gamma",
        "batch_size", "exploration_initial_eps", "exploration_final_eps",
        "exploration_fraction", "total_timesteps", "mean_reward",
        "std_reward", "model_path",
    ]
    # Only write header if file doesn't exist (so results accumulate across runs)
    if not os.path.exists(results_file):
        with open(results_file, "w", newline="") as f:
            csv.writer(f).writerow(header)

    for i, hp in enumerate(hyperparams_list, start=1):
        print(f"\n=== Experiment {i} / {len(hyperparams_list)} ===")
        print("Hyperparams:", hp)
        print(f"Training for {total_timesteps:,} timesteps...")
        
        try:
            use_cnn = (policy == "CnnPolicy")
            env = make_env(use_cnn=use_cnn)
            eval_env = make_env(use_cnn=use_cnn)

            #  FIXED: Enable proper normalization for CNN
            policy_kwargs = {}
            if use_cnn:
                policy_kwargs = dict(normalize_images=True)  #  FIXED: Enable normalization

            # Resolve training config (from CLI/preset)
            cfg = config or {}
            # buffer, learning start, gradient steps, target update interval
            buffer_size = getattr(cfg, 'buffer_size', None) or 200000
            learning_starts = getattr(cfg, 'learning_starts', None) or 50000
            gradient_steps = getattr(cfg, 'gradient_steps', None) or 4
            target_update_interval = getattr(cfg, 'target_update_interval', None) or 10000
            device = getattr(cfg, 'device', 'auto')
            # tensorboard logging
            tensorboard_log = None if getattr(cfg, 'no_tensorboard', False) else LOG_DIR

            model = DQN(
                policy,
                env,
                # configurable replay buffer for speed/stability
                buffer_size=buffer_size,
                learning_rate=hp.get("learning_rate", 1e-4),
                batch_size=hp.get("batch_size", 32),
                gamma=hp.get("gamma", 0.99),
                exploration_initial_eps=hp.get("exploration_initial_eps", 1.0),
                # Use hyperparam-provided final eps (defaults to a low value for stable policies)
                exploration_final_eps=hp.get("exploration_final_eps", 0.01),
                exploration_fraction=hp.get("exploration_fraction", 0.1),
                # configurable target update and learning params
                target_update_interval=target_update_interval,
                train_freq=4,
                gradient_steps=gradient_steps,
                # Delay learning until buffer has a reasonable amount of experience
                learning_starts=learning_starts,
                verbose=1,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                device=device,
            )

            best_model_dir = os.path.join(LOG_DIR, f"best_exp_{i}")
            os.makedirs(best_model_dir, exist_ok=True)
            #  FIXED: Less frequent but more thorough evaluation
            # Evaluation config
            eval_freq = getattr(cfg, 'eval_freq', None) or 50000
            n_eval_episodes = getattr(cfg, 'n_eval_episodes', None) or 10
            callbacks = []
            if eval_freq and eval_freq > 0:
                callbacks = [
                    EvalCallback(
                        eval_env,
                        best_model_save_path=best_model_dir,
                        log_path=best_model_dir,
                        eval_freq=eval_freq,
                        n_eval_episodes=n_eval_episodes,
                        deterministic=True,
                    )
                ]

            model.learn(total_timesteps=total_timesteps, callback=callbacks)

            # Save per-experiment model
            model_path = os.path.join(LOG_DIR, f"dqn_exp_{i}.zip")
            model.save(model_path)

            # Use a robust evaluation helper that handles VecEnv or Gym envs
            test_env = make_env(use_cnn=use_cnn)
            mean_reward, std_reward = robust_evaluate(model, test_env, n_eval_episodes=10, deterministic=True)

            with open(results_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    i,
                    policy,
                    hp.get("learning_rate", 1e-4),
                    hp.get("gamma", 0.99),
                    hp.get("batch_size", 32),
                    hp.get("exploration_initial_eps", 1.0),
                    hp.get("exploration_final_eps", 0.01),
                    hp.get("exploration_fraction", 0.1),
                    total_timesteps,
                    mean_reward,
                    std_reward,
                    model_path,
                ])

            print(f" Experiment {i} completed successfully!")
            print(f"Final performance: {mean_reward:.2f} ± {std_reward:.2f} reward")
            env.close()
            test_env.close()
            
        except Exception as e:
            print(f" Experiment {i} failed with error: {e}")
            import traceback
            traceback.print_exc()
            # Record failed experiment row so results.csv reflects attempts
            try:
                model_path = locals().get('model_path', '') or ''
                with open(results_file, "a", newline="") as f:
                    csv.writer(f).writerow([
                        i,
                        policy,
                        hp.get("learning_rate", ''),
                        hp.get("gamma", ''),
                        hp.get("batch_size", ''),
                        hp.get("exploration_initial_eps", ''),
                        hp.get("exploration_final_eps", ''),
                        hp.get("exploration_fraction", ''),
                        total_timesteps,
                        "FAILED",
                        "FAILED",
                        model_path,
                    ])
            except Exception:
                # If writing to CSV also fails, continue without blocking further experiments
                pass
            continue


def default_hyperparams():
    """Hyperparameters optimized for Atari environments"""
    # Ten carefully chosen hyperparameter combinations to try.
    # These explore learning rates, gamma, batch sizes and exploration schedules
    return [
        {"learning_rate": 1e-4,  "gamma": 0.99,  "batch_size": 32, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01, "exploration_fraction": 0.05},
        {"learning_rate": 5e-5,  "gamma": 0.99,  "batch_size": 32, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01, "exploration_fraction": 0.10},
        {"learning_rate": 2.5e-4, "gamma": 0.99,  "batch_size": 32, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "exploration_fraction": 0.10},
        {"learning_rate": 1e-4,  "gamma": 0.997, "batch_size": 32, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01, "exploration_fraction": 0.05},
        {"learning_rate": 1e-4,  "gamma": 0.99,  "batch_size": 64, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01, "exploration_fraction": 0.10},
        {"learning_rate": 5e-5,  "gamma": 0.997, "batch_size": 64, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01, "exploration_fraction": 0.05},
        {"learning_rate": 2.5e-4, "gamma": 0.997, "batch_size": 32, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "exploration_fraction": 0.10},
        {"learning_rate": 7.5e-5, "gamma": 0.99,  "batch_size": 64, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01, "exploration_fraction": 0.05},
        {"learning_rate": 1.5e-4, "gamma": 0.995, "batch_size": 32, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01, "exploration_fraction": 0.08},
        {"learning_rate": 1e-4,  "gamma": 0.99,  "batch_size": 32, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.005, "exploration_fraction": 0.05},
    ]


def robust_evaluate(model, env, n_eval_episodes: int = 10, deterministic: bool = True):
    """Evaluate a model on env for n episodes and return mean and std of episode rewards.

    This helper handles both vectorized envs (VecEnv) and regular Gym envs by
    running episodes on a single sub-environment and using model.predict(...).
    """
    # Try to get a single environment to run episodes on
    single_env = env
    try:
        # VecEnv exposes `envs` list (most implementations). Use the first sub-env.
        if hasattr(env, "envs") and len(getattr(env, "envs")) > 0:
            single_env = env.envs[0]
    except Exception:
        single_env = env

    rewards = []
    for ep in range(n_eval_episodes):
        obs = single_env.reset()
        done = False
        ep_rew = 0.0
        # Some envs return (obs, info) from reset
        if isinstance(obs, tuple) or (isinstance(obs, list) and len(obs) == 2):
            obs = obs[0]

        while True:
            # model.predict accepts raw obs; unwrap for vectorized shapes if necessary
            action, _ = model.predict(obs, deterministic=deterministic)
            step_result = single_env.step(action)
            # step may return (obs, rew, done, info) or (obs, rew, terminated, truncated, info)
            if len(step_result) == 4:
                obs, rew, done, info = step_result
                truncated = False
            else:
                obs, rew, terminated, truncated, info = step_result
                done = terminated or truncated

            ep_rew += float(rew)
            if done:
                break

        rewards.append(ep_rew)

    mean = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
    std = float(np.std(rewards)) if len(rewards) > 0 else 0.0
    print(f"robust_evaluate -> rewards: {rewards}, mean: {mean:.2f}, std: {std:.2f}")
    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # CLI: timesteps, presets and tuning knobs for fast iterations
    parser.add_argument("--timesteps", type=int, default=3000000)
    parser.add_argument("--experiments", type=int, default=1)
    parser.add_argument("--policy", choices=["CnnPolicy", "MlpPolicy"], default="CnnPolicy",
                        help="Policy to use: CnnPolicy (recommended for Atari) or MlpPolicy")
    parser.add_argument("--preset", choices=["debug", "fast", "full"], default="debug",
                        help="Preset configuration for speed: debug (very fast), fast (moderate), full (original)")
    parser.add_argument("--buffer_size", type=int, default=None, help="Replay buffer size (overrides preset)")
    parser.add_argument("--learning_starts", type=int, default=None, help="Learning starts (overrides preset)")
    parser.add_argument("--gradient_steps", type=int, default=None, help="Gradient steps per update (overrides preset)")
    parser.add_argument("--target_update_interval", type=int, default=None, help="Target network update interval (overrides preset)")
    parser.add_argument("--eval_freq", type=int, default=None, help="Evaluation frequency (set <=0 to disable)")
    parser.add_argument("--n_eval_episodes", type=int, default=None, help="Number of eval episodes")
    parser.add_argument("--device", type=str, default='auto', help="Device for PyTorch/SB3 (cpu,cuda,auto)")
    parser.add_argument("--no-tensorboard", action='store_true', help="Disable tensorboard logging to speed up I/O")
    args = parser.parse_args()

    # Apply preset defaults (only when user didn't pass explicit values)
    if args.preset == 'debug':
        if args.timesteps == 3000000:
            args.timesteps = 50000
        if args.buffer_size is None:
            args.buffer_size = 50000
        if args.learning_starts is None:
            args.learning_starts = 5000
        if args.gradient_steps is None:
            args.gradient_steps = 1
        if args.target_update_interval is None:
            args.target_update_interval = 1000
        if args.eval_freq is None:
            args.eval_freq = 25000
        if args.n_eval_episodes is None:
            args.n_eval_episodes = 1
    elif args.preset == 'fast':
        if args.timesteps == 3000000:
            args.timesteps = 100000
        if args.buffer_size is None:
            args.buffer_size = 100000
        if args.learning_starts is None:
            args.learning_starts = 10000
        if args.gradient_steps is None:
            args.gradient_steps = 1
        if args.target_update_interval is None:
            args.target_update_interval = 2000
        if args.eval_freq is None:
            args.eval_freq = 50000
        if args.n_eval_episodes is None:
            args.n_eval_episodes = 3
    else:  # full
        # full (original) defaults
        if args.buffer_size is None:
            args.buffer_size = 200000
        if args.learning_starts is None:
            args.learning_starts = 50000
        if args.gradient_steps is None:
            args.gradient_steps = 4
        if args.target_update_interval is None:
            args.target_update_interval = 10000
        if args.eval_freq is None:
            args.eval_freq = 50000
        if args.n_eval_episodes is None:
            args.n_eval_episodes = 10

    # Pass args (config) into run_experiments so it can use the CLI/preset values
    hps = default_hyperparams()[: args.experiments]
    run_experiments(hps, total_timesteps=args.timesteps, policy=args.policy, config=args)