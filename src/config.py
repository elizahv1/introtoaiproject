# src/config.py
from typing import Dict, Any

ENV_ID: str = "highway-v0"

ENV_CONFIG: Dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False
    },
    "action": {
        "type": "DiscreteMetaAction"
    },
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "duration": 80,

    "lanes_count": 3,
    "vehicles_count": 40,
    "initial_lane_vehicles_count": 20,
    "spawn_probability": 0.6,

    "offroad_terminal": False,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",

    "collision_reward": -5.0,
    "high_speed_reward": 1.5,
    "lane_change_reward": -0.05,
    "right_lane_reward": 0.6,
    "reward_speed_range": [20, 50]
}

PPO_CONFIG: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": 0.99,
    "tensorboard_log": "./assets/logs/"
}

# --- UPDATED TO 50,000 TOTAL TIMESTEPS ---
TOTAL_TIMESTEPS: int = 50000