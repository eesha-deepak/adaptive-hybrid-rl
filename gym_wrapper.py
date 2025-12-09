import gymnasium as gym
import numpy as np

class OldGymWrapper(gym.Env):

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
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
    try:
        if env_id == "Pushing2D-v1":
            from pusher2d import Pusher2d
            from gym_wrapper import OldGymWrapper
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
        print(f"Error loading {env_id}: {e}")
        return None


def train_sac(env_id, config, seed=0):

    if env_id == "Pushing2D-v1":
        from pusher2d import Pusher2d
        from gym_wrapper import OldGymWrapper
        env = OldGymWrapper(Pusher2d())
        eval_env = OldGymWrapper(Pusher2d())
    else:
        env = gym.make(env_id)
        eval_env = gym.make(env_id)
