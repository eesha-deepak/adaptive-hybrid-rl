"""
PETS (Probabilistic Ensembles with Trajectory Sampling) Planner
Model-based planning with ensemble uncertainty quantification
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from hybrid.replay_buffer import ReplayBuffer


class EnsembleDynamicsModel(nn.Module):
    """
    Ensemble of probabilistic neural network dynamics models
    Predicts next state distribution: p(s' | s, a)
    """
    
    def __init__(self, obs_dim, act_dim, n_models=5, hidden_size=400, device='cpu'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_models = n_models
        self.device = device
        
        # Learnable log variance bounds (prevents numerical instability)
        self.max_logvar = nn.Parameter(
            torch.ones(1, obs_dim, device=device) * -3,
            requires_grad=False
        )
        self.min_logvar = nn.Parameter(
            torch.ones(1, obs_dim, device=device) * -7,
            requires_grad=False
        )
        
        # Create ensemble of networks
        self.networks = nn.ModuleList([
            self._create_network(hidden_size) 
            for _ in range(n_models)
        ])
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.to(device)
    
    def _create_network(self, hidden_size):
        """Single 3-layer MLP for dynamics prediction"""
        return nn.Sequential(
            nn.Linear(self.obs_dim + self.act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * self.obs_dim)  # Output: [mean, logvar]
        )
    
    def _get_output(self, raw_output):
        """Split raw output into mean and bounded log variance"""
        mean = raw_output[..., :self.obs_dim]
        raw_logvar = raw_output[..., self.obs_dim:]
        
        # Bound log variance to prevent numerical issues
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        
        return mean, logvar
    
    def forward(self, obs, action, model_idx=None):
        """
        Predict next state delta: Δs = s' - s
        
        Args:
            obs: (batch, obs_dim) current observations
            action: (batch, act_dim) actions
            model_idx: which model to use (None = all models)
        
        Returns:
            mean, logvar for state delta prediction
        """
        x = torch.cat([obs, action], dim=-1)
        
        if model_idx is not None:
            # Use specific model
            output = self.networks[model_idx](x)
            return self._get_output(output)
        else:
            # Use all models (returns list of (mean, logvar) tuples)
            outputs = [self._get_output(net(x)) for net in self.networks]
            return outputs
    
    def sample_prediction(self, obs, action, model_idx):
        """Sample from a specific model's predictive distribution"""
        mean, logvar = self.forward(obs, action, model_idx=model_idx)
        std = torch.exp(0.5 * logvar)
        
        # Sample delta and add to current state
        delta = mean + std * torch.randn_like(std)
        next_obs = obs + delta
        
        return next_obs
    
    def get_disagreement(self, obs, action):
        """
        Compute ensemble disagreement (epistemic uncertainty)
        
        Returns:
            Scalar disagreement (std across ensemble predictions)
        """
        with torch.no_grad():
            predictions = []
            for i in range(self.n_models):
                mean, _ = self.forward(obs, action, model_idx=i)
                predictions.append(mean)
            
            # Stack and compute std across models
            predictions = torch.stack(predictions, dim=0)  # (n_models, batch, obs_dim)
            disagreement = torch.std(predictions, dim=0).mean(dim=-1)  # (batch,)
            
        return disagreement
    
    def train_ensemble(self, inputs, targets, batch_size=128, num_epochs=5):
        """
        Train ensemble with bootstrapped data
        
        Args:
            inputs: (N, obs_dim + act_dim) concatenated [obs, action]
            targets: (N, obs_dim) state deltas (next_obs - obs)
            batch_size: mini-batch size
            num_epochs: number of training epochs
        
        Returns:
            losses: list of average loss per epoch
        """
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        
        N = inputs.shape[0]
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Train each model in ensemble with bootstrapped data
            for net in self.networks:
                # Bootstrap sampling (different data subset for each model)
                idx = np.random.randint(0, N, size=min(batch_size, N))
                x_batch = inputs[idx]
                y_batch = targets[idx]
                
                # Forward pass
                mean, logvar = self._get_output(net(x_batch))
                
                # Gaussian negative log-likelihood loss
                var = torch.exp(logvar)
                loss = 0.5 * (torch.log(2 * np.pi * var) + (y_batch - mean)**2 / var)
                loss = loss.mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            losses.append(np.mean(epoch_losses))
        
        return losses


class PETSAgent:
    """
    PETS (Probabilistic Ensembles with Trajectory Sampling) Agent
    Model-based planning with CEM optimization and ensemble uncertainty
    """
    
    def __init__(self, env_info: Dict, 
                 replay_buffer: Optional[ReplayBuffer] = None,
                 n_models=5, horizon=15, 
                 n_candidates=500, n_elite=50, n_iterations=5,
                 num_particles=6, device='cpu'):
        """
        Args:
            env_info: Dict with obs_dim, act_dim, act_low, act_high
            replay_buffer: External replay buffer (for unified buffer in hybrid)
            n_models: Number of ensemble models
            horizon: Planning horizon (MPC)
            n_candidates: CEM population size
            n_elite: CEM elite size
            n_iterations: CEM iterations
            num_particles: Trajectory sampling particles (TS1)
        """
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
        
        # Create ensemble
        self.ensemble = EnsembleDynamicsModel(
            self.obs_dim, self.act_dim, n_models, device=device
        )
        
        # Replay buffer (use external or create new)
        if replay_buffer is None:
            self.buffer = ReplayBuffer(
                state_dim=self.obs_dim,
                action_dim=self.act_dim,
                max_size=100000,
                save_timestep=False
            )
        else:
            self.buffer = replay_buffer
        
        # Normalization statistics (CRITICAL for good performance!)
        self.obs_mean = np.zeros(self.obs_dim)
        self.obs_std = np.ones(self.obs_dim)
        
        # CEM distribution parameters (warm-started between steps)
        self.mean = np.zeros(horizon * self.act_dim)
        self.var = np.ones(horizon * self.act_dim) * 0.25
        
        print(f"✅ PETS: {n_models} models, horizon={horizon}, candidates={n_candidates}")
    
    def plan(self, obs: np.ndarray) -> np.ndarray:
        """
        Plan action sequence using CEM + TS1
        
        Args:
            obs: Current observation
        
        Returns:
            action: First action of optimized sequence (MPC)
        """
        # Normalize observation
        obs_norm = (obs - self.obs_mean) / (self.obs_std + 1e-6)
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # CEM optimization
        mean = self.mean.copy()
        var = self.var.copy()
        
        # Action bounds
        action_bounds_low = np.tile(self.act_low, self.horizon)
        action_bounds_high = np.tile(self.act_high, self.horizon)
        
        for iteration in range(self.n_iterations):
            # Sample candidate action sequences
            action_seqs = np.random.randn(self.n_candidates, self.horizon * self.act_dim)
            action_seqs = action_seqs * np.sqrt(var) + mean
            action_seqs = np.clip(action_seqs, action_bounds_low, action_bounds_high)
            
            # Evaluate sequences with trajectory sampling
            costs = self._evaluate_action_sequences_ts1(obs_t, action_seqs)
            
            # Select elites (best sequences)
            elite_idx = np.argsort(costs)[:self.n_elite]
            elites = action_seqs[elite_idx]
            
            # Update distribution
            mean = np.mean(elites, axis=0)
            var = np.var(elites, axis=0) + 1e-6  # Prevent collapse
        
        # Warm start for next step (shift and pad)
        best_action_seq = mean.reshape(self.horizon, self.act_dim)
        self.mean = np.concatenate([
            mean[self.act_dim:],
            np.zeros(self.act_dim)
        ])
        self.var = np.concatenate([
            var[self.act_dim:],
            np.ones(self.act_dim) * 0.25
        ])
        
        # Return first action (Model Predictive Control)
        return best_action_seq[0]
    
    def _evaluate_action_sequences_ts1(self, obs, action_seqs):
        """
        Evaluate action sequences using TS1 (Trajectory Sampling)
        
        For each sequence, sample multiple trajectories using different models
        to estimate expected cost under model uncertainty
        
        Args:
            obs: (1, obs_dim) initial observation (NORMALIZED)
            action_seqs: (n_seqs, horizon * act_dim) action sequences
        
        Returns:
            costs: (n_seqs,) expected costs
        """
        n_seqs = action_seqs.shape[0]
        
        # Replicate obs for each (sequence, particle) pair
        obs_particles = obs.repeat(n_seqs * self.num_particles, 1)
        
        # Cost accumulator
        costs = np.zeros((n_seqs, self.num_particles))
        
        # Reshape actions: (horizon, n_seqs, act_dim)
        actions = action_seqs.reshape(n_seqs, self.horizon, self.act_dim)
        actions = np.transpose(actions, (1, 0, 2))
        
        # Rollout trajectories
        current_obs = obs_particles
        
        for t in range(self.horizon):
            # Current actions: (n_seqs, act_dim)
            current_actions = actions[t]
            
            # Repeat for particles: (n_seqs * n_particles, act_dim)
            current_actions_particles = torch.tensor(
                np.repeat(current_actions, self.num_particles, axis=0),
                dtype=torch.float32,
                device=self.device
            )
            
            # TS1: Randomly assign each particle to a model
            with torch.no_grad():
                model_indices = np.random.randint(
                    0, self.n_models,
                    size=n_seqs * self.num_particles
                )
                
                # Sample next states from assigned models
                next_obs_list = []
                for i, model_idx in enumerate(model_indices):
                    next_obs = self.ensemble.sample_prediction(
                        current_obs[i:i+1],
                        current_actions_particles[i:i+1],
                        model_idx=model_idx
                    )
                    next_obs_list.append(next_obs)
                
                next_obs = torch.cat(next_obs_list, dim=0)
            
            # Compute step costs (on NORMALIZED states)
            step_costs = self._cost_fn(next_obs.cpu().numpy())
            step_costs = step_costs.reshape(n_seqs, self.num_particles)
            costs += step_costs
            
            current_obs = next_obs
        
        # Return average cost over particles
        return np.mean(costs, axis=1)
    
    def _cost_fn(self, states):
        """
        Cost function for planning
        Must denormalize states and compute actual environment-specific cost
        
        Args:
            states: (batch, obs_dim) NORMALIZED states
            
        Returns:
            costs: (batch,) cost for each state (NEGATIVE reward)
        """
        # Denormalize states back to original scale
        states_denorm = states * self.obs_std + self.obs_mean
        
        # Environment-specific costs
        if self.obs_dim == 3:  
            # Pendulum-v1: [cos(θ), sin(θ), θ_dot]
            # Goal: θ=0 (upright) → cos(θ)=1, sin(θ)=0, θ_dot=0
            cos_th = states_denorm[:, 0]
            sin_th = states_denorm[:, 1]
            thdot = states_denorm[:, 2]
            
            # Pendulum reward (from Gymnasium source code)
            # reward = -(theta^2 + 0.1*theta_dot^2 + 0.001*action^2)
            # Approximate as: -(angle_from_upright^2 + 0.1*thdot^2)
            angle_cost = (cos_th - 1.0)**2 + sin_th**2  # Distance from upright
            velocity_cost = 0.1 * thdot**2
            
            costs = angle_cost + velocity_cost
            return costs
            
        elif self.obs_dim == 4:
            # InvertedPendulum-v5: [x, x_dot, theta, theta_dot]
            # Goal: keep pole upright (theta ≈ 0) and cart centered (x ≈ 0)
            x = states_denorm[:, 0]
            theta = states_denorm[:, 2]
            
            costs = x**2 + 10.0 * theta**2  # Heavily penalize pole angle
            return costs
            
        elif self.obs_dim == 8:
            # LunarLander or similar
            # Goal: minimize distance and velocity
            costs = np.sum(states_denorm**2, axis=-1)
            return costs
            
        else:
            # Generic: minimize state magnitude (works for many envs)
            costs = np.sum(states_denorm**2, axis=-1)
            return costs
    
    def add_data(self, obs, action, next_obs, reward):
        """Add transition to replay buffer"""
        self.buffer.add(obs, action, next_obs, reward, False)
    
    def train_models(self, n_epochs=5, batch_size=128):
        """
        Train ensemble on collected data
        
        CRITICAL: Computes normalization statistics and normalizes data
        
        Args:
            n_epochs: Training epochs
            batch_size: Batch size
        
        Returns:
            losses: Training loss history
        """
        if self.buffer.size < 10:
            print(f"⚠️  Buffer too small ({self.buffer.size}), skipping training")
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
        
        print(f"✅ Trained models: loss={losses[-1]:.4f}, buffer_size={self.buffer.size}")
        
        return losses
    
    def act(self, obs):
        """Compatibility wrapper for HybridAgent"""
        action = self.plan(obs)
        return {"action": action}