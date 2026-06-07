
import os
import sys
import gc

cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
if os.path.exists(cuda_path):
    os.add_dll_directory(cuda_path)

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from config import ENV_ID, ENV_CONFIG, PPO_CONFIG, TOTAL_TIMESTEPS

def run_training() -> None:
    os.makedirs("./assets/checkpoints", exist_ok=True)
    os.makedirs("./assets/logs", exist_ok=True)

    env = make_vec_env(
        ENV_ID,
        n_envs=4,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": ENV_CONFIG}
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=2500,  
        save_path="./assets/checkpoints/",
        name_prefix="highway_ppo"
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,         
        device="cuda",     
        **PPO_CONFIG
    )
    
    gc.disable()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    gc.enable()
    
    model.save("./assets/checkpoints/highway_ppo_fully_trained")
    print("Training complete.")
    env.close()

if __name__ == "__main__":
    run_training()