import os, sys, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
import gymnasium as gym
from tqdm import trange

# ---------- tiny utils ----------
def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1]),
                   act() if i < len(sizes)-2 else out_act()]
    return nn.Sequential(*layers)

# ---------- replay buffer ----------
class Buffer:
    def __init__(self, size, obs_dim, act_dim, device="cpu"):
        self.size_limit = size
        self.device = device
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, *, obs, next_obs, action, reward, done, **kwargs):
        i = self.ptr % self.size_limit
        self.obs[i] = obs if isinstance(obs, np.ndarray) else obs.cpu().numpy()
        self.next_obs[i] = next_obs if isinstance(next_obs, np.ndarray) else next_obs.cpu().numpy()
        self.actions[i] = action if isinstance(action, np.ndarray) else action.cpu().numpy()
        self.rewards[i, 0] = reward
        self.dones[i, 0] = done
        self.ptr += 1
        self.size = min(self.size + 1, self.size_limit)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        d = self.device
        return {
            "obs": torch.as_tensor(self.obs[idx], device=d),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=d),
            "actions": torch.as_tensor(self.actions[idx], device=d),
            "rewards": torch.as_tensor(self.rewards[idx], device=d),
            "dones": torch.as_tensor(self.dones[idx], device=d),
        }

# ---------- tanh-Gaussian policy pieces ----------
class _TanhDiagGaussian:
    def __init__(self, mu, log_std, act_low, act_high):
        self.mu = mu
        self.std = torch.exp(torch.clamp(log_std, -5.0, 2.0))
        self.base = torch.distributions.Normal(self.mu, self.std)
        self.act_low, self.act_high = act_low, act_high
        self.scale = (act_high - act_low) / 2.0
        self.shift = (act_high + act_low) / 2.0
        with torch.no_grad():
            self.mean_action = torch.tanh(self.mu) * self.scale + self.shift

    def _squash(self, u):
        return torch.tanh(u) * self.scale + self.shift

    def rsample(self):
        u = self.base.rsample()
        return self._squash(u)

    def log_prob(self, a):
        eps = 1e-6
        a_unscaled = (a - self.shift) / (self.scale + eps)
        a_unscaled = torch.clamp(a_unscaled, -1 + 1e-6, 1 - 1e-6)
        u = torch.atanh(a_unscaled)
        logp = self.base.log_prob(u).sum(-1, keepdim=True)
        logp -= torch.log(1 - a_unscaled.pow(2) + 1e-6).sum(-1, keepdim=True)
        return logp

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden], act=nn.ReLU, out_act=nn.ReLU)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
        self.act_low = act_low
        self.act_high = act_high

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h)
        return _TanhDiagGaussian(mu, log_std, self.act_low, self.act_high)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden, 1], act=nn.ReLU)
    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))

# ---------- SAC Agent (with auto-α + adaptive UTD) ----------
class SACAgent:
    """
    Soft Actor-Critic with adaptive UTD ratio
    """
    def __init__(self, env_info, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 batch_size=256, update_every=1, buffer_size=100000,
                 warmup_steps=2000, utd_ratio=1, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_dim = env_info["obs_dim"]
        self.act_dim = env_info["act_dim"]
        self.act_low = torch.as_tensor(env_info["act_low"], dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(env_info["act_high"], dtype=torch.float32, device=self.device)
        self.gamma, self.tau = gamma, tau
        self.batch_size, self.update_every = batch_size, update_every
        self.warmup_steps = warmup_steps
        self.base_utd_ratio = utd_ratio

        self.actor = Actor(self.obs_dim, self.act_dim, self.act_low, self.act_high).to(self.device)
        self.critic1 = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.critic2 = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.critic1_target = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.critic2_target = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)

        # Auto entropy temperature (α)
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        # Use -0.5 * act_dim for Pendulum to prevent entropy collapse
        self.target_entropy = -0.5 * float(self.act_dim)

        self._buffer = Buffer(size=buffer_size, obs_dim=self.obs_dim, act_dim=self.act_dim, device=self.device.type)
        self.total_steps = 0

    def get_utd_ratio(self):
        """Adaptive UTD: Start with 1, gradually increase to base_utd_ratio"""
        if self.total_steps < self.warmup_steps:
            return 1
        # Ramp over 10k steps
        progress = min(1.0, max(0.0, (self.total_steps - self.warmup_steps) / 10_000))
        utd = 1 + (self.base_utd_ratio - 1) * progress
        utd = int(max(1, round(utd)))
        # Cap by how many batches we can draw (avoid thrashing a small buffer)
        max_batches = max(1, self._buffer.size // self.batch_size)
        return min(utd, max_batches)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor(obs_t)
        action = dist.mean_action if deterministic else dist.rsample()
        action = torch.clamp(action, self.act_low, self.act_high)
        return {"action": action.squeeze(0).cpu().numpy()}

    def step(self, transition: Dict[str, Any]) -> Dict[str, float]:
        obs_t = torch.as_tensor(transition["obs"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(transition["next_obs"], dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(transition["action"], dtype=torch.float32, device=self.device)
        self._buffer.add(
            obs=obs_t, next_obs=next_obs_t, action=action_t,
            reward=float(transition["reward"]), done=float(transition["done"])
        )
        self.total_steps += 1
        if self.total_steps < self.warmup_steps or self._buffer.size < self.batch_size:
            return {}
        if self.total_steps % self.update_every != 0:
            return {}
        return self._perform_update()

    def _perform_update(self) -> Dict[str, float]:
        utd = self.get_utd_ratio()
        all_stats = []
        for _ in range(max(1, utd)):
            batch = self._buffer.sample(self.batch_size)
            stats = self._sac_update_step(batch)
            all_stats.append(stats)
        avg_stats = {k: float(np.mean([s[k] for s in all_stats])) for k in all_stats[0].keys()} if all_stats else {}
        avg_stats['utd'] = float(utd)
        return avg_stats

    def _sac_update_step(self, batch) -> Dict[str, float]:
        obs, actions, rewards, next_obs, dones = (
            batch["obs"], batch["actions"], batch["rewards"], batch["next_obs"], batch["dones"]
        )
        alpha = self.log_alpha.exp()

        # ----- Critic target -----
        with torch.no_grad():
            dist_next = self.actor(next_obs)
            a_next = dist_next.rsample()
            logp_next = torch.clamp(dist_next.log_prob(a_next), -20, 20)
            q_min = torch.min(self.critic1_target(next_obs, a_next),
                              self.critic2_target(next_obs, a_next))
            r = torch.clamp(rewards, -1000.0, 1000.0)
            target_q = r + self.gamma * (1.0 - dones) * (q_min - alpha * logp_next)
            target_q = target_q.detach()

        # ----- Critic update -----
        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic_opt.step()

        # ----- Actor update -----
        dist = self.actor(obs)
        a = dist.rsample()
        logp = torch.clamp(dist.log_prob(a), -20, 20)
        q_min_pi = torch.min(self.critic1(obs, a), self.critic2(obs, a))
        actor_loss = (alpha * logp - q_min_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ----- Temperature (alpha) update -----
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # ----- Target soft-updates -----
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic1_loss": float(nn.functional.mse_loss(q1, target_q).item()),
            "critic2_loss": float(nn.functional.mse_loss(q2, target_q).item()),
            "alpha": float(alpha.item()),
            "q1": float(q1.mean().item()),
            "q2": float(q2.mean().item()),
        }

    @torch.no_grad()
    def _soft_update(self, local, target):
        for p, tp in zip(local.parameters(), target.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

# ---------- evaluation function ----------
def evaluate(agent, env, n_episodes=10, deterministic=True):
    """Evaluate agent with deterministic or stochastic actions"""
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            action = agent.act(obs, deterministic=deterministic)["action"]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)
    return np.mean(returns), np.std(returns)

# ---------- training entrypoint ----------
def train(env_id="Pendulum-v1", steps=100_000, seed=0, save_path="artifacts/sac_actor_hw.pth"):
    os.makedirs("artifacts", exist_ok=True)
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    env.reset(seed=seed)
    eval_env.reset(seed=seed+1000)
    np.random.seed(seed); torch.manual_seed(seed)

    obs, _ = env.reset()
    env_info = {
        "obs_dim": env.observation_space.shape[0],
        "act_dim": env.action_space.shape[0],
        "act_low": env.action_space.low,
        "act_high": env.action_space.high,
    }
    # KEY: Longer warmup (2000) + adaptive UTD that ramps from 1→3
    agent = SACAgent(env_info, device=None, warmup_steps=2000, batch_size=256, utd_ratio=3)

    ep_ret, returns = 0.0, []
    eval_returns = []
    
    for step in trange(steps, desc=f"SAC({env_id})"):
        # Random actions during warmup
        if agent.total_steps < agent.warmup_steps:
            a = env.action_space.sample()
        else:
            a = agent.act(obs, deterministic=False)["action"]
        
        nxt, r, d, tr, _ = env.step(a)
        stats = agent.step({"obs": obs, "action": a, "reward": r, "next_obs": nxt, "done": float(d or tr)})
        obs = nxt; ep_ret += r
        
        if d or tr:
            returns.append(ep_ret); obs, _ = env.reset(); ep_ret = 0.0
        
        # Periodic evaluation
        if (step + 1) % 5000 == 0:
            eval_mean, eval_std = evaluate(agent, eval_env, n_episodes=10)
            eval_returns.append(eval_mean)
            current_utd = agent.get_utd_ratio()
            current_alpha = agent.log_alpha.exp().item()
            print(f"\nStep {step+1}: Eval = {eval_mean:.2f} ± {eval_std:.2f} | UTD={current_utd} | alpha={current_alpha:.4f}")

    # Final evaluation
    eval_mean, eval_std = evaluate(agent, eval_env, n_episodes=20)
    print(f"\n{'='*60}")
    print(f"Final Evaluation (20 episodes): {eval_mean:.2f} ± {eval_std:.2f}")
    print(f"Final alpha: {agent.log_alpha.exp().item():.4f}")
    print(f"{'='*60}")

    torch.save({
        "actor": agent.actor.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
    }, save_path)
    print("Saved actor and log_alpha to", save_path)
    print("Avg Return (last 10):", np.mean(returns[-10:]) if returns else None)
    return returns, eval_returns

if __name__ == "__main__":
    train()