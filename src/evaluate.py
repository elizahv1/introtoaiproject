
import os
import gymnasium as gym
import highway_env  
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from config import ENV_ID, ENV_CONFIG

def record_evolution_stage(checkpoint_name: str, video_folder: str) -> None:
    eval_config = ENV_CONFIG.copy()

    eval_config.update({
        "vehicles_count": 35,              
        "initial_lane_vehicles_count": 15, 
        "spawn_probability": 0.6,          
        "scaling": 5.5     #camera zoom            
    })
    
    env = gym.make(ENV_ID, render_mode="rgb_array", config=eval_config)
    
    env = RecordVideo(
        env=env,
        video_folder=video_folder,
        name_prefix=checkpoint_name,
        disable_logger=True
    )
    
    checkpoint_path = f"./assets/checkpoints/{checkpoint_name}"
    obs, _ = env.reset()
    done = False
    truncated = False
    
    total_reward = 0.0
    steps_survived = 0

    if "untrained" in checkpoint_name:
        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps_survived += 1
            
    elif os.path.exists(checkpoint_path + ".zip"):
        model = PPO.load(checkpoint_path, env=env)
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps_survived += 1
    else:
        env.close()
        return
        
    env.close()

    outcome = "COLLISION (Crashed)" if done else "SUCCESS (Survived Full Duration)"
    print(f"------------ METRICS FOR {checkpoint_name.upper()} ------------")
    print(f"  * Final Outcome:   {outcome}")
    print(f"  * Total Reward:    {total_reward:.2f}")
    print(f"  * Steps Survived:  {steps_survived} / {eval_config.get('duration', 40)}")
    print(f"-----------------------------------------------------------\n")

def run_evaluation() -> None:
    os.makedirs("./videos", exist_ok=True)
    
    stages = [
        "highway_ppo_untrained", 
        "highway_ppo_10000_steps",      #untrained , partial trained and fully trained model videos
        "highway_ppo_fully_trained"
    ]
    
    
    for stage in stages:
        record_evolution_stage(stage, "./videos")

if __name__ == "__main__":
    run_evaluation()