import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from typing import Dict, Optional

from hybrid.planner_pets_improved import ImprovedPETSAgent as PETSAgent
from hybrid.gate import SwitchingGate
from sac import SACAgent

class HybridAgent:
    """
    Adaptive hybrid agent that switches between model-based and model-free:
    """
    
    def __init__(self, 
                 env_info: Dict,
                 # PETS parameters
                 n_models: int = 5,
                 horizon: int = 15,
                 n_candidates: int = 500,
                 n_elite: int = 50,
                 # SAC parameters
                 sac_ckpt: Optional[str] = None,
                 # Gate parameters
                 disagreement_threshold: float = 0.05,
                 calibration_quantile: float = 0.9,
                 hysteresis_factor: float = 0.2,
                 # General
                 device=None):

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.env_info = env_info
        
        self.pets = PETSAgent(
            env_info,
            n_models=n_models,
            horizon=horizon,
            n_candidates=n_candidates,
            n_elite=n_elite,
            device=self.device
        )
        
        self.sac = SACAgent(env_info, device=self.device)
        if sac_ckpt:
            self.load_sac(sac_ckpt)
        
        self.gate = SwitchingGate(
            threshold=disagreement_threshold,
            calibration_quantile=calibration_quantile,
            hysteresis_factor=hysteresis_factor
        )

        self.total_steps = 0
        self.n_model_based = 0
        self.n_model_free = 0
        self.episode_modes = []
        
    def load_sac(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.sac.actor.load_state_dict(checkpoint["actor"])
        print(f"Loaded SAC policy from {ckpt_path}")
        
        if "log_alpha" in checkpoint:
            self.sac.log_alpha.data = checkpoint["log_alpha"].to(self.device)
            print(f"   Loaded alpha = {self.sac.log_alpha.exp().item():.4f}")
    
    def act(self, obs: np.ndarray, mode: str = "auto") -> Dict:
        """
        Select action using hybrid strategy
        """
        disagreement = self._get_disagreement(obs)

        if mode == "auto":
            use_mb = self.gate.should_use_mb(disagreement)
        elif mode == "model-based":
            use_mb = True
        elif mode == "model-free":
            use_mb = False
        elif mode == "mixed":
            return self._act_mixed(obs, disagreement)
        else:
            print("Mode does not exist")
        
        # Execute chosen controller
        if use_mb:
            result = self.pets.act(obs)
            self.n_model_based += 1
            result["mode"] = "MB"
        else:
            result = self.sac.act(obs, deterministic=False)
            self.n_model_free += 1
            result["mode"] = "MF"
        
        result["disagreement"] = disagreement
        self.total_steps += 1
        
        return result
    
    def _get_disagreement(self, obs: np.ndarray) -> float:
      """
      Compute ensemble disagreement for given observation.
      """
      obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
      
      with torch.no_grad():
          sac_action = self.sac.act(obs, deterministic=False)["action"]
      
      action_t = torch.tensor(sac_action, dtype=torch.float32, device=self.device).unsqueeze(0)
      with torch.no_grad():
          disagreement = self.pets.ensemble.get_disagreement(obs_t, action_t)
      
      return disagreement.item()
    
    def _act_mixed(self, obs: np.ndarray, disagreement: float) -> Dict:
        """
        Soft mixing of model-based and model-free actions
        Weight based on model confidence (low disagreement = high MB weight)
        """
        w_mb = max(0.0, 1.0 - disagreement / self.gate.threshold)
        w_mf = 1.0 - w_mb
        
        action_mb = self.pets.act(obs)["action"]
        action_mf = self.sac.act(obs, deterministic=False)["action"]

        action = w_mb * action_mb + w_mf * action_mf

        if w_mb > w_mf:
            self.n_model_based += 1
            mode = "MB"
        else:
            self.n_model_free += 1
            mode = "MF"
        
        self.total_steps += 1
        
        return {
            "action": action,
            "mode": f"{mode}_mixed",
            "disagreement": disagreement,
            "w_mb": w_mb,
            "w_mf": w_mf
        }
    
    def add_data(self, obs, action, next_obs, reward, done):
        """
        Add transition to both agents and update calibration
        """
        self.pets.add_data(obs, action, next_obs, reward)

        self.sac.step({
            "obs": obs,
            "action": action,
            "next_obs": next_obs,
            "reward": reward,
            "done": float(done)
        })

        self._update_calibration(obs, action, next_obs)
    
    def _update_calibration(self, obs, action, next_obs):
        """Track (disagreement, actual_error) for threshold calibration"""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            disagreement = self.pets.ensemble.get_disagreement(obs_t, action_t)
            mean, _ = self.pets.ensemble.forward(obs_t, action_t, model_idx=0)
            pred_next_obs = obs_t + mean

            error = torch.norm(pred_next_obs - next_obs_t, dim=-1).item()

            self.gate.add_calibration_data(disagreement.item(), error)
    
    def train_models(self, n_epochs: int = 100, batch_size: int = 256) -> Dict:
        """
        Train PETS ensemble models on collected data
        """
        return self.pets.train_models(n_epochs, batch_size)
    
    def calibrate_threshold(self) -> Dict:
        return self.gate.calibrate_threshold()
    
    def get_stats(self) -> Dict:
        total_actions = self.n_model_based + self.n_model_free
        
        stats = {
            "total_steps": self.total_steps,
            "n_model_based": self.n_model_based,
            "n_model_free": self.n_model_free,
            "model_based_pct": 100 * self.n_model_based / max(1, total_actions),
            "model_free_pct": 100 * self.n_model_free / max(1, total_actions),
            "buffer_size": self.pets.buffer.size,
        }
        stats.update(self.gate.get_stats())

        coverage = self.gate.get_coverage_stats()
        if coverage:
            stats.update(coverage)
        
        return stats
    
    def reset_episode_tracking(self):
        self.episode_modes = []
    
    def get_episode_stats(self) -> Dict:
        if not self.episode_modes:
            return {}
        
        mb_count = sum(1 for m in self.episode_modes if m == "MB")
        total = len(self.episode_modes)
        
        return {
            "episode_length": total,
            "mb_actions": mb_count,
            "mf_actions": total - mb_count,
            "mb_pct": 100 * mb_count / total if total > 0 else 0
        }