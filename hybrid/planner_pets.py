import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from hybrid.replay_buffer import ReplayBuffer


class EnsembleDynamicsModel(nn.Module):
    """
    Ensemble of probabilistic neural network dynamics models
    """
    
    def __init__(self, obs_dim, act_dim, n_models=5, hidden_size=400, device='cpu'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_models = n_models
        self.device = device
        
        self.max_logvar = nn.Parameter(
            torch.ones(1, obs_dim, device=device) * -3,
            requires_grad=False
        )
        self.min_logvar = nn.Parameter(
            torch.ones(1, obs_dim, device=device) * -7,
            requires_grad=False
        )
        
        self.networks = nn.ModuleList([
            self._create_network(hidden_size) 
            for _ in range(n_models)
        ])
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.to(device)
    
    def _create_network(self, hidden_size):
        return nn.Sequential(
            nn.Linear(self.obs_dim + self.act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * self.obs_dim)
        )
    
    def _get_output(self, raw_output):
        mean = raw_output[..., :self.obs_dim]
        raw_logvar = raw_output[..., self.obs_dim:]
        
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        
        return mean, logvar
    
    def forward(self, obs, action, model_idx=None):
        """
        Predict next state delta
        """
        x = torch.cat([obs, action], dim=-1)
        
        if model_idx is not None:
            output = self.networks[model_idx](x)
            return self._get_output(output)
        else:
            outputs = [self._get_output(net(x)) for net in self.networks]
            return outputs
    
    def sample_prediction(self, obs, action, model_idx):
        """Sample from a specific model's predictive distribution"""
        mean, logvar = self.forward(obs, action, model_idx=model_idx)
        std = torch.exp(0.5 * logvar)
        
        delta = mean + std * torch.randn_like(std)
        next_obs = obs + delta
        
        return next_obs
    
    def get_disagreement(self, obs, action):
        """
        Compute ensemble disagreement
        """
        with torch.no_grad():
            predictions = []
            for i in range(self.n_models):
                mean, _ = self.forward(obs, action, model_idx=i)
                predictions.append(mean)

            predictions = torch.stack(predictions, dim=0)
            disagreement = torch.std(predictions, dim=0).mean(dim=-1)
            
        return disagreement
    
    def train_ensemble(self, inputs, targets, batch_size=128, num_epochs=5):
        """
        Train ensemble with bootstrapped data
        """
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        
        N = inputs.shape[0]
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for net in self.networks:
                idx = np.random.randint(0, N, size=min(batch_size, N))
                x_batch = inputs[idx]
                y_batch = targets[idx]
                
                mean, logvar = self._get_output(net(x_batch))
                
                var = torch.exp(logvar)
                loss = 0.5 * (torch.log(2 * np.pi * var) + (y_batch - mean)**2 / var)
                loss = loss.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            losses.append(np.mean(epoch_losses))
        
        return losses


class PETSAgent:
    def __init__(self, env_info: Dict, 
                 replay_buffer: Optional[ReplayBuffer] = None,
                 n_models=5, horizon=15, 
                 n_candidates=500, n_elite=50, n_iterations=5,
                 num_particles=6, device='cpu'):
        self.device = device
        self.obs_dim = env_info["obs_dim"]
        self.act_dim = env_info["act_dim"]
        self.act_low = env_info["act_low"]
        self.act_high = env_info["act_high"]
        
        self.n_models = n_models
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_elite = n_elite
        self.n_iterations = n_iterations
        self.num_particles = num_particles

        self.ensemble = EnsembleDynamicsModel(
            self.obs_dim, self.act_dim, n_models, device=device
        )

        if replay_buffer is None:
            self.buffer = ReplayBuffer(
                state_dim=self.obs_dim,
                action_dim=self.act_dim,
                max_size=100000,
                save_timestep=False
            )
        else:
            self.buffer = replay_buffer

        self.obs_mean = np.zeros(self.obs_dim)
        self.obs_std = np.ones(self.obs_dim)

        self.mean = np.zeros(horizon * self.act_dim)
        self.var = np.ones(horizon * self.act_dim) * 0.25
        
        print(f"PETS: {n_models} models, horizon={horizon}, candidates={n_candidates}")
    
    def plan(self, obs: np.ndarray) -> np.ndarray:
        """
        Plan action sequence using CEM + TS1
        """
        obs_norm = (obs - self.obs_mean) / (self.obs_std + 1e-6)
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        mean = self.mean.copy()
        var = self.var.copy()
        
        action_bounds_low = np.tile(self.act_low, self.horizon)
        action_bounds_high = np.tile(self.act_high, self.horizon)
        
        for iteration in range(self.n_iterations):
            action_seqs = np.random.randn(self.n_candidates, self.horizon * self.act_dim)
            action_seqs = action_seqs * np.sqrt(var) + mean
            action_seqs = np.clip(action_seqs, action_bounds_low, action_bounds_high)
            
            costs = self._evaluate_action_sequences_ts1(obs_t, action_seqs)
            
            elite_idx = np.argsort(costs)[:self.n_elite]
            elites = action_seqs[elite_idx]
            
            mean = np.mean(elites, axis=0)
            var = np.var(elites, axis=0) + 1e-6
        
        best_action_seq = mean.reshape(self.horizon, self.act_dim)
        self.mean = np.concatenate([
            mean[self.act_dim:],
            np.zeros(self.act_dim)
        ])
        self.var = np.concatenate([
            var[self.act_dim:],
            np.ones(self.act_dim) * 0.25
        ])

        return best_action_seq[0]
    
    def _evaluate_action_sequences_ts1(self, obs, action_seqs):
        """
        Evaluate action sequences using TS1 (Trajectory Sampling)
        """
        n_seqs = action_seqs.shape[0]
        
        obs_particles = obs.repeat(n_seqs * self.num_particles, 1)

        costs = np.zeros((n_seqs, self.num_particles))

        actions = action_seqs.reshape(n_seqs, self.horizon, self.act_dim)
        actions = np.transpose(actions, (1, 0, 2))

        current_obs = obs_particles
        
        for t in range(self.horizon):
            current_actions = actions[t]

            current_actions_particles = torch.tensor(
                np.repeat(current_actions, self.num_particles, axis=0),
                dtype=torch.float32,
                device=self.device
            )

            with torch.no_grad():
                model_indices = np.random.randint(
                    0, self.n_models,
                    size=n_seqs * self.num_particles
                )

                next_obs_list = []
                for i, model_idx in enumerate(model_indices):
                    next_obs = self.ensemble.sample_prediction(
                        current_obs[i:i+1],
                        current_actions_particles[i:i+1],
                        model_idx=model_idx
                    )
                    next_obs_list.append(next_obs)
                
                next_obs = torch.cat(next_obs_list, dim=0)

            step_costs = self._cost_fn(next_obs.cpu().numpy())
            step_costs = step_costs.reshape(n_seqs, self.num_particles)
            costs += step_costs
            
            current_obs = next_obs

        return np.mean(costs, axis=1)
    
    def _cost_fn(self, states):
        states_denorm = states * self.obs_std + self.obs_mean

        if self.obs_dim == 3:  
            cos_th = states_denorm[:, 0]
            sin_th = states_denorm[:, 1]
            thdot = states_denorm[:, 2]

            angle_cost = (cos_th - 1.0)**2 + sin_th**2
            velocity_cost = 0.1 * thdot**2
            
            costs = angle_cost + velocity_cost
            return costs
            
        elif self.obs_dim == 4:
            x = states_denorm[:, 0]
            theta = states_denorm[:, 2]
            
            costs = x**2 + 10.0 * theta**2
            return costs
            
        elif self.obs_dim == 8:
            costs = np.sum(states_denorm**2, axis=-1)
            return costs
            
        else:
            costs = np.sum(states_denorm**2, axis=-1)
            return costs
    
    def add_data(self, obs, action, next_obs, reward):
        self.buffer.add(obs, action, next_obs, reward, False)
    
    def train_models(self, n_epochs=5, batch_size=128):
        """
        Train ensemble on collected data
        """
        if self.buffer.size < 10:
            print(f"Buffer too small ({self.buffer.size}), skipping training")
            return [0.0]
        
        # Sample all data
        actual_batch_size = min(batch_size, self.buffer.size)
        buffer_data = self.buffer.sample(self.buffer.size)
        
        obs = buffer_data[0].cpu().numpy()
        action = buffer_data[1].cpu().numpy()
        next_obs = buffer_data[2].cpu().numpy()
        
        # CRITICAL: Compute normalization statistics
        self.obs_mean = np.mean(obs, axis=0)
        self.obs_std = np.std(obs, axis=0) + 1e-6  # Add epsilon to prevent division by zero
        
        # Normalize observations
        obs_norm = (obs - self.obs_mean) / self.obs_std
        next_obs_norm = (next_obs - self.obs_mean) / self.obs_std
        
        # Prepare inputs and targets (on NORMALIZED data)
        inputs = np.concatenate([obs_norm, action], axis=-1)
        targets = next_obs_norm - obs_norm  # Predict normalized delta
        
        # Train ensemble
        losses = self.ensemble.train_ensemble(inputs, targets, actual_batch_size, n_epochs)
        
        print(f"Trained models: loss={losses[-1]:.4f}, buffer_size={self.buffer.size}")
        
        return losses
    
    def act(self, obs):
        action = self.plan(obs)
        return {"action": action}