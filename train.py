import gymnasium as gym
import ale_py
import os
import numpy as np
import gc
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import CheckpointCallback

# --- Configuration ---
ENV_ID = "BreakoutNoFrameskip-v4"
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 4
LOG_DIR = "./dqn_logs_set_10/"
MODEL_SAVE_PATH = "dqn_model.zip" # Final model name

# --- Winning Hyperparameters (Set 10) ---
WINNING_HYPERPARAMS = {
    'name': 'Set_10_Large_Batch_Faster_LR',
    'lr': 2e-4,  # Faster learning rate for quicker convergence
    'gamma': 0.99,
    'batch_size': 256,  # Very large batch for maximum stability
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.1
}

def setup_environment(env_id, n_envs):
    """Creates the vectorized, frame-stacked training environment."""
    # This utility handles all necessary Atari wrappers (grayscale, resize, etc.)
    env = make_atari_env(env_id, n_envs=n_envs, seed=np.random.randint(1000))
    # Stacks 4 consecutive frames so the agent can perceive motion
    env = VecFrameStack(env, n_stack=4)
    return env

def train_agent(hyperparams, total_timesteps):
    """Initializes and trains the DQN agent."""
    
    print(f"\n" + "="*60)
    print(f"STARTING TRAINING: {hyperparams['name']}")
    print(f"Parameters: {hyperparams}")
    print(f"="*60)
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. Setup Environment
    env = setup_environment(ENV_ID, N_ENVS)

    # 2. Setup Callback (to save checkpoints)
    checkpoint_callback = CheckpointCallback(
        save_freq=200000 // N_ENVS, # Save every 200k steps
        save_path=LOG_DIR,
        name_prefix="dqn_chkpt"
    )

    # 3. Define the DQN Model
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=hyperparams['lr'],
        gamma=hyperparams['gamma'],
        batch_size=hyperparams['batch_size'],
        exploration_initial_eps=hyperparams['epsilon_start'],
        exploration_final_eps=hyperparams['epsilon_end'],
        exploration_fraction=hyperparams['epsilon_decay'],
        buffer_size=50000,
        train_freq=4,
        verbose=1, # Set to 1 to see training tables
        tensorboard_log=LOG_DIR
    )

    # 4. Train the Model
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True # Show the progress bar
    )
    
    # 5. Save the Final Model
    model.save(MODEL_SAVE_PATH)
    
    # 6. Cleanup
    env.close()
    del model
    gc.collect()
    
    print(f"\n" + "="*60)
    print(f"Training Complete!")
    print(f"Final model saved to: {MODEL_SAVE_PATH}")
    print(f"Logs saved to: {LOG_DIR}")
    print(f"="*60)

if __name__ == "__main__":
    # Register the ALE environments
    gym.register_envs(ale_py)
    
    # Run the training
    train_agent(WINNING_HYPERPARAMS, TOTAL_TIMESTEPS)