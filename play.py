# 4. AUTOMATED FINAL EVALUATION & PLAY (WITH VIDEO RECORDING)

import gymnasium as gym
import ale_py
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
import os
import shutil 

# --- Import the video recorder wrapper ---
from gymnasium.wrappers import RecordVideo

# --- Define a folder to save the videos ---
VIDEO_FOLDER = 'videos'
os.makedirs(VIDEO_FOLDER, exist_ok=True)


def make_env_render(model_name):
    """
    Creates the environment and wraps it for video recording.
    """
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    
    # This will record every episode to find the best run
    env = RecordVideo(
        env,
        video_folder=VIDEO_FOLDER,
        name_prefix=f"{model_name}-run",
        episode_trigger=lambda x: True  # Record ALL episodes
    )
    
    env = AtariWrapper(env)
    return env

def play_agent(model_path):
    print(f"Starting Agent Evaluation and Video Generation...")
    print(f"Loading model from: {model_path}")
    
    try:
        model = DQN.load(model_path)
        # Extract a clean name for the video file
        model_name = model_path.split('/')[-1].replace('.zip', '')
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{model_path}'")
        return
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
        return

    # Pass the model_name to the env creator
    eval_env = DummyVecEnv([lambda: make_env_render(model_name)])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    episode_rewards = []
    
    # --- ADDED: Track the best score ---
    best_score = -np.inf
    best_episode_index = 0
    N_EPISODES = 10 # <-- Set number of episodes to run
    
    print("\n" + "="*60)
    print(f"Starting Evaluation: Playing {N_EPISODES} episodes...")
    print(f"(All episodes will be recorded to '{VIDEO_FOLDER}' folder)")
    print("="*60)
    
    for i in range(N_EPISODES):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_array, info = eval_env.step(action)
            
            # This render() call is necessary to trigger the video recorder
            eval_env.render()
            
            total_reward += reward[0]
            done = done_array[0]
        
        episode_rewards.append(total_reward)
        print(f"Visual Episode {i+1} finished with reward: {total_reward:.2f}")
        
        # --- ADDED: Check if this is the new best score ---
        if total_reward > best_score:
            best_score = total_reward
            best_episode_index = i
    
    eval_env.close()
    
    # Calculate and print average
    mean_reward = np.mean(episode_rewards)
    print(f"\n" + "="*60)
    print(f"Evaluation Complete.")
    print(f"--> Average Reward over {len(episode_rewards)} episodes: {mean_reward:.2f}")
    print(f"--> Best run was Episode {best_episode_index + 1} with a score of {best_score:.2f}")

    # Find and copy the best video
    best_video_filename = None
    # Video files are named like '...-run-episode-0.mp4'
    for file in os.listdir(VIDEO_FOLDER):
        if file.endswith(f"-episode-{best_episode_index}.mp4"):
            best_video_filename = file
            break
    
    if best_video_filename:
        source_path = os.path.join(VIDEO_FOLDER, best_video_filename)
        # Clean destination name
        dest_path = f"video.mp4"
        
        # Copy the file to the root directory
        shutil.copy(source_path, dest_path)
        print(f"\nSuccessfully copied best run to: {dest_path}")
    else:
        print(f"\nError: Could not find video file for best episode (index {best_episode_index}).")
        
    print(f"All original videos remain in '{VIDEO_FOLDER}' directory.")
    print(f"="*60)


if __name__ == "__main__":
    # Register the ALE environments
    gym.register_envs(ale_py)
    
    # This is the path to your winning model
    MODEL_PATH = "dqn_model_set_10.zip"
    
    play_agent(MODEL_PATH)