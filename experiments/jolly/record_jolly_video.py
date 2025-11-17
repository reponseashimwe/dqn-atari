import os
import gymnasium as gym
import ale_py  # <-- important: ALE Atari integration
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

# Ensure Atari environments like BreakoutNoFrameskip-v4 are registered
gym.register_envs(ale_py)

# Path to your best model (Jolly - Experiment 5)
MODEL_PATH = "./logs/dqn_exp_5.zip"
VIDEO_DIR = "./videos"

def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    print(f"Loading model from: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH)

    # Create the Atari environment with rgb_array rendering for video
    env_id = "BreakoutNoFrameskip-v4"
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id, render_mode="rgb_array")

    # Wrap with RecordVideo: record every episode
    env = RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        name_prefix="jolly_breakout_dqn",
        episode_trigger=lambda episode_id: True,  # record all episodes
    )

    num_episodes = 3
    print(f"Recording {num_episodes} episodes...")

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Use the trained model to predict actions
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

        print(f"Episode {ep + 1}/{num_episodes} finished with reward: {total_reward}")

    env.close()
    print(f" Video(s) saved in: {os.path.abspath(VIDEO_DIR)}")

if __name__ == "__main__":
    main()
