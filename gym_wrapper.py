"""
Wrapper to make old Gym environments compatible with Gymnasium
Use this for Pushing2D which uses old Gym API
"""
import gymnasium as gym
import numpy as np

class OldGymWrapper(gym.Env):
    """
    Wraps old gym.Env to work with Gymnasium
    
    Old API: step() returns (obs, reward, done, info)
    New API: step() returns (obs, reward, terminated, truncated, info)
    """
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self, seed=None, options=None):
        """Gymnasium reset() returns (obs, info)"""
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        return obs, {}
    
    def step(self, action):
        """Convert old (obs, reward, done, info) to new (obs, reward, terminated, truncated, info)"""
        obs, reward, done, info = self.env.step(action)
        
        # Split 'done' into 'terminated' and 'truncated'
        # Check if it's a timeout (max steps reached)
        truncated = info.get("done") == "max_steps_reached"
        terminated = done and not truncated
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render() if hasattr(self.env, 'render') else None
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


# Usage in train_all_envs.py:
def get_env_info(env_id):
    """Get environment information"""
    try:
        # Handle custom Pushing2D
        if env_id == "Pushing2D-v1":
            from pusher2d import Pusher2d
            from gym_wrapper import OldGymWrapper  # This file
            env = OldGymWrapper(Pusher2d())
        else:
            env = gym.make(env_id)
        
        env_info = {
            "obs_dim": env.observation_space.shape[0],
            "act_dim": env.action_space.shape[0],
            "act_low": env.action_space.low,
            "act_high": env.action_space.high,
        }
        env.close()
        return env_info
    except Exception as e:
        print(f"❌ Error loading {env_id}: {e}")
        return None


def train_sac(env_id, config, seed=0):
    """Train SAC - updated to handle Pushing2D"""
    # ... (existing code)
    
    # Create environment
    if env_id == "Pushing2D-v1":
        from pusher2d import Pusher2d
        from gym_wrapper import OldGymWrapper
        env = OldGymWrapper(Pusher2d())
        eval_env = OldGymWrapper(Pusher2d())
    else:
        env = gym.make(env_id)
        eval_env = gym.make(env_id)
    
    # ... (rest of training code)