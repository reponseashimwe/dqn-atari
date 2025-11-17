import os
import gymnasium as gym
import ale_py
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv
import argparse
from gymnasium.wrappers import RecordVideo
from train import default_hyperparams
import re
import csv
import datetime
import cv2

def make_play_env(env_id="BreakoutNoFrameskip-v4", render_mode=None):
    """Create a vectorized Atari env with the same preprocessing used during training.

    We use `make_atari_env(n_envs=1)` which applies the standard Atari wrappers
    (frame stacking, preprocessing) and returns a VecEnv. For rendering we call
    the underlying sub-environment's `render()` method.
    """
    # make_atari_env will apply the same preprocessing used during training
    vec_env = make_atari_env(env_id, n_envs=1, seed=42)
    # Wrap with Monitor on the sub-env for episode logging
    try:
        if hasattr(vec_env, "envs") and len(vec_env.envs) > 0:
            vec_env.envs[0] = Monitor(vec_env.envs[0])
    except Exception:
        pass

    return vec_env


def robust_evaluate(model, env, n_eval_episodes: int = 10, deterministic: bool = True):
    """Evaluate a model on env for n episodes and return mean and std of episode rewards.

    Handles both VecEnv and regular Gym envs by running episodes on a single
    sub-environment and using model.predict(...).
    """
    # Try to get a single environment to run episodes on
    single_env = env
    try:
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
            action, _ = model.predict(obs, deterministic=deterministic)
            step_result = single_env.step(action)
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

def load_latest_model(log_dir="logs"):
    """Load the most recent trained model"""
    # Look for the best model from experiments
    best_model_path = None
    
    # Check for experiment models
    for i in range(1, 10):  # Check first 10 experiments
        exp_dir = os.path.join(log_dir, f"best_exp_{i}")
        model_path = os.path.join(exp_dir, "best_model.zip")
        if os.path.exists(model_path):
            best_model_path = model_path
            print(f"Found best model from experiment {i}")
            break
    
    # If no best model found, look for the final trained models
    if best_model_path is None:
        for i in range(1, 10):
            model_path = os.path.join(log_dir, f"dqn_exp_{i}.zip")
            if os.path.exists(model_path):
                best_model_path = model_path
                print(f"Found final model from experiment {i}")
                break
    
    if best_model_path is None:
        raise FileNotFoundError("No trained model found. Please run train.py first.")
    
    print(f"Loading model from: {best_model_path}")
    model = DQN.load(best_model_path)
    return model, best_model_path

def play_episodes(model, env, num_episodes=10, deterministic=True, render=True, auto_press_fire=True):
    """Play episodes using either a VecEnv (recommended) or a single Gym env.

    If `env` is a VecEnv (as returned by `make_atari_env`), we use batched
    observations and actions (num_envs==1). To render, we call the underlying
    sub-environment's `render()` method (env.envs[0].render()).
    """
    total_rewards = []
    total_lengths = []

    is_vec = hasattr(env, "num_envs")

    for ep in range(num_episodes):
        # Reset returns batched obs for VecEnv
        reset_out = env.reset()
        if is_vec:
            if isinstance(reset_out, tuple):
                obs, infos = reset_out
            else:
                obs = reset_out
            # obs has batch dim
            batch_obs = obs
        else:
            if isinstance(reset_out, tuple):
                obs, info = reset_out
            else:
                obs = reset_out
            batch_obs = np.expand_dims(obs, 0)

        ep_reward = 0.0
        ep_len = 0
        done = False

        print(f"\n=== Episode {ep + 1}/{num_episodes} ===")

        # Debug: print observation & action info once at episode start
        try:
            if is_vec and hasattr(env, "envs") and len(env.envs) > 0:
                subenv = env.envs[0]
            else:
                subenv = env

            # Action meanings (useful for Atari to check FIRE/button mapping)
            try:
                action_meanings = subenv.unwrapped.get_action_meanings()
            except Exception:
                try:
                    action_meanings = subenv.get_action_meanings()
                except Exception:
                    action_meanings = None

            print("[debug] is_vec:", is_vec)
            print("[debug] batch_obs.shape:", getattr(batch_obs, 'shape', None))
            try:
                print("[debug] obs min/max:", np.min(batch_obs), np.max(batch_obs))
            except Exception:
                pass
            print("[debug] action_space:", getattr(subenv, 'action_space', None))
            if action_meanings is not None:
                print("[debug] action_meanings:", action_meanings)
        except Exception:
            pass

        # Optional: press FIRE at start to begin the game if necessary (Breakout needs FIRE)
        if auto_press_fire:
            try:
                if action_meanings and 'FIRE' in action_meanings:
                    fire_idx = int(action_meanings.index('FIRE'))
                    # send a few FIRE actions to ensure the game starts
                    for _ in range(4):
                        if is_vec:
                            out = env.step([fire_idx])
                            # consume outputs but don't use them here
                        else:
                            env.step(fire_idx)
                        # small break between presses
            except Exception:
                pass

        while not done:
            # Predict using batched observation
            action, _ = model.predict(batch_obs, deterministic=deterministic)
            # Ensure actions are batched for VecEnv.step
            if is_vec:
                step_action = action
            else:
                # action may be an array like [x]
                step_action = [int(action[0])]

            step_out = env.step(step_action)
            # VecEnv.step returns batched outputs
            if is_vec:
                next_obs, rewards, dones, infos = step_out
                obs = next_obs[0]
                reward = float(rewards[0])
                done = bool(dones[0])
            else:
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = step_out

            ep_reward += float(reward)
            ep_len += 1

            # Update batch_obs for next prediction
            if is_vec:
                batch_obs = np.expand_dims(obs, 0)
            else:
                batch_obs = np.expand_dims(obs, 0)

            # Render via sub-env if requested
            if render:
                try:
                    # Prefer to get an RGB array and display via OpenCV for a reliable window
                    frame = None
                    if is_vec and hasattr(env, "envs") and len(env.envs) > 0:
                        subenv = env.envs[0]
                        try:
                            # try common signatures
                            frame = subenv.render()
                        except TypeError:
                            try:
                                frame = subenv.render(mode='rgb_array')
                            except Exception:
                                frame = None
                    else:
                        try:
                            frame = env.render()
                        except TypeError:
                            try:
                                frame = env.render(mode='rgb_array')
                            except Exception:
                                frame = None

                    if frame is not None:
                        # frame is typically HxWxC RGB; OpenCV expects BGR
                        try:
                            img = frame[:, :, ::-1]
                        except Exception:
                            img = frame
                        cv2.imshow('Agent Play', img)
                        # waitKey is necessary for the window to update; 1 ms is non-blocking
                        cv2.waitKey(1)
                    else:
                        # fallback to env's own render (may open a window)
                        if is_vec and hasattr(env, "envs"):
                            try:
                                env.envs[0].render()
                            except Exception:
                                pass
                        else:
                            try:
                                env.render()
                            except Exception:
                                pass

                except Exception:
                    pass

            if ep_len % 100 == 0:
                print(f"Step {ep_len}, Reward: {ep_reward}")

        total_rewards.append(ep_reward)
        total_lengths.append(ep_len)

        print(f"Episode {ep + 1} finished:")
        print(f"  Total reward: {ep_reward}")
        print(f"  Episode length: {ep_len}")

    # Summary
    print(f"\n=== Performance Summary ===")
    print(f"Average reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"Average episode length: {np.mean(total_lengths):.2f} +/- {np.std(total_lengths):.2f}")
    print(f"Max reward: {np.max(total_rewards)}")
    print(f"Min reward: {np.min(total_rewards)}")

def main():
    parser = argparse.ArgumentParser(description="Play with a trained DQN agent in Atari")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to the trained model (default: auto-detect latest)")
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4",
                       help="Atari environment ID")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to play")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic actions (recommended for evaluation)")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic actions (for exploration)")
    parser.add_argument("--no-render", action="store_true",
                       help="Run without rendering (faster)")
    parser.add_argument("--record", action="store_true",
                       help="Record video of the episodes to logs/videos/")
    
    args = parser.parse_args()
    
    # Set deterministic based on arguments
    deterministic = not args.stochastic
    
    try:
        # Load the trained model and remember the path
        if args.model_path and os.path.exists(args.model_path):
            print(f"Loading specified model: {args.model_path}")
            model = DQN.load(args.model_path)
            model_path = args.model_path
        else:
            model, model_path = load_latest_model()
        
        # Decide whether to render during play
        render_flag = not args.no_render

        # Create a vectorized play env with the same preprocessing used in training
        env = make_play_env(args.env)

        # Optionally wrap the underlying sub-env with RecordVideo to save a video file
        if args.record:
            video_dir = os.path.join("logs", "videos")
            os.makedirs(video_dir, exist_ok=True)
            try:
                if hasattr(env, "envs") and len(env.envs) > 0:
                    # Force recording for every episode by using an episode_trigger that returns True
                    try:
                        env.envs[0] = RecordVideo(
                            env.envs[0],
                            video_folder=video_dir,
                            name_prefix="play",
                            episode_trigger=lambda idx: True,
                        )
                    except TypeError:
                        # Fallback for older/newer gym versions that use `video_callable` name
                        try:
                            env.envs[0] = RecordVideo(
                                env.envs[0],
                                video_folder=video_dir,
                                name_prefix="play",
                                video_callable=lambda idx: True,
                            )
                        except Exception:
                            # If both signatures fail, wrap without forcing and warn
                            env.envs[0] = RecordVideo(env.envs[0], video_folder=video_dir, name_prefix="play")
                else:
                    # Fallback: try to wrap env directly
                    try:
                        env = RecordVideo(env, video_folder=video_dir, name_prefix="play", episode_trigger=lambda idx: True)
                    except TypeError:
                        try:
                            env = RecordVideo(env, video_folder=video_dir, name_prefix="play", video_callable=lambda idx: True)
                        except Exception:
                            env = RecordVideo(env, video_folder=video_dir, name_prefix="play")
            except Exception:
                print("Warning: failed to enable recording; continuing without video.")
        
        print("\n" + "="*50)
        print("Starting Game Play")
        print("="*50)
        print(f"Environment: {args.env}")
        print(f"Model: {type(model).__name__}")
        print(f"Policy: {'Deterministic' if deterministic else 'Stochastic'}")
        print(f"Episodes: {args.episodes}")
        print("="*50)

        # Play the episodes (render if requested)
        play_episodes(model, env,
                      num_episodes=args.episodes,
                      deterministic=deterministic,
                      render=render_flag)

        # Evaluate deterministically for logging (use 10 episodes)
        try:
            # If we loaded model via load_latest_model, we may have model_path variable
            model_path = args.model_path
        except Exception:
            model_path = None

        # model_path should already be set from above (either --model_path or discovered)
        # If not set for any reason, try to discover it without loading a model
        if not model_path:
            try:
                _, discovered_path = load_latest_model()
                model_path = discovered_path
            except Exception:
                model_path = None

        # Perform a robust evaluation (no rendering) and append results to logs/results.csv
        try:
            eval_env = make_play_env(args.env)
            mean_reward, std_reward = robust_evaluate(model, eval_env, n_eval_episodes=10, deterministic=True)
            eval_env.close()

            # Prepare CSV row
            results_file = os.path.join("logs", "results.csv")
            header = [
                "experiment_id", "policy", "learning_rate", "gamma",
                "batch_size", "exploration_initial_eps", "exploration_final_eps",
                "exploration_fraction", "total_timesteps", "mean_reward",
                "std_reward", "model_path",
            ]

            # Try to parse experiment id from model_path
            exp_id = ''
            hp = None
            if model_path:
                m = re.search(r"best_exp_(\d+)", model_path)
                if not m:
                    m = re.search(r"dqn_exp_(\d+)", model_path)
                if m:
                    exp_id = int(m.group(1))
                    try:
                        hps = default_hyperparams()
                        if 1 <= exp_id <= len(hps):
                            hp = hps[exp_id - 1]
                    except Exception:
                        hp = None

            # Ensure results CSV exists with header
            if not os.path.exists(results_file):
                with open(results_file, "w", newline="") as f:
                    csv.writer(f).writerow(header)

            # Compose row, using hyperparams if available
            row = [
                exp_id or '',
                getattr(model, 'policy', '') and getattr(model.policy, '__class__', type('X', (), {})).__name__ or 'DQN',
                hp.get('learning_rate', '') if hp else '',
                hp.get('gamma', '') if hp else '',
                hp.get('batch_size', '') if hp else '',
                hp.get('exploration_initial_eps', '') if hp else '',
                hp.get('exploration_final_eps', '') if hp else '',
                hp.get('exploration_fraction', '') if hp else '',
                '',  # total_timesteps unknown here
                mean_reward,
                std_reward,
                model_path or '',
            ]

            with open(results_file, "a", newline="") as f:
                csv.writer(f).writerow(row)

            print(f"Appended evaluation to {results_file}: mean={mean_reward:.2f} std={std_reward:.2f}")
        except Exception as e:
            print(f"Warning: could not evaluate & append results automatically: {e}")

        env.close()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have run train.py first to train a model.")
    except KeyboardInterrupt:
        print("\nPlay interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()