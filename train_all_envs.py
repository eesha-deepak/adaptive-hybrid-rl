import os
import numpy as np
import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt
from hybrid.agent import HybridAgent
from sac import SACAgent
from hybrid.planner_pets import PETSAgent
import torch
from hybrid.planner_pets_improved import ImprovedPETSAgent

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

def make_pushing2d_env():
    from pusher2d import Pusher2d
    from gym_wrapper import OldGymWrapper
    return OldGymWrapper(Pusher2d())

def get_env_info(env_id):
    try:
        if env_id == "Pushing2D-v1":
            env = make_pushing2d_env()
        else:
            env = gym.make(env_id)
        
        env_info = {
            "obs_dim": env.observation_space.shape[0],
            "act_dim": env.action_space.shape[0],
            "act_low": env.action_space.low,
            "act_high": env.action_space.high,
        }
        if hasattr(env, 'close'):
            env.close()
        return env_info
    except Exception as e:
        print(f"Error loading {env_id}: {e}")
        return None

ENV_CONFIGS = {
    "Pendulum-v1": {
        "sac_steps": 100_000,
        "pets_steps": 50_000,
        "hybrid_steps": 5000,
        "warmup_steps": 2000,
        "horizon": 15,
        "model_train_freq": 1000,
        "eval_freq": 5000,
        "expected_sac": -143,
        "expected_pets": -250,
        "dynamics": "smooth",
        "description": "Inverted pendulum - smooth dynamics",
    },
    
    "InvertedPendulum-v5": {
        "sac_steps": 50_000,
        "pets_steps": 25_000,
        "hybrid_steps": 3000,
        "warmup_steps": 1000,
        "horizon": 10,
        "model_train_freq": 1000,
        "eval_freq": 2500,
        "expected_sac": 40,
        "expected_pets": 40,
        "dynamics": "smooth",
        "description": "Simple pole balancing - very smooth",
    },
    
    "Reacher-v5": {
        "sac_steps": 100_000,
        "pets_steps": 50_000,
        "hybrid_steps": 5000,
        "warmup_steps": 2000,
        "horizon": 5,
        "model_train_freq": 1000,
        "eval_freq": 5000,
        "expected_sac": -6,
        "expected_pets": -10,
        "dynamics": "smooth",
        "description": "Robotic arm reaching - smooth continuous control",
    },
    
    "Swimmer-v5": {
        "sac_steps": 100_000,
        "pets_steps": 30_000,
        "hybrid_steps": 5000,
        "warmup_steps": 2000,
        "horizon": 5,
        "model_train_freq": 1000,
        "eval_freq": 5000,
        "expected_sac": 50,
        "expected_pets": 35,
        "dynamics": "smooth",
        "description": "Swimming robot - smooth fluid dynamics",
    },
    
    "LunarLanderContinuous-v3": {
        "sac_steps": 300_000,
        "pets_steps": 100_000,
        "hybrid_steps": 10000,
        "warmup_steps": 5000,
        "horizon": 20,
        "model_train_freq": 2000,
        "eval_freq": 15000,
        "expected_sac": 270,
        "expected_pets": -190,
        "dynamics": "contact",
        "description": "Rocket landing - contact dynamics, aerodynamics",
    },
    
    "Hopper-v5": {
        "sac_steps": 300_000,
        "pets_steps": 100_000,
        "hybrid_steps": 10000,
        "warmup_steps": 5000,
        "horizon": 15,
        "model_train_freq": 2000,
        "eval_freq": 15000,
        "expected_sac": 3600,
        "expected_pets": 38,
        "dynamics": "contact",
        "description": "One-legged hopping - ground contact",
    },
    
    "Walker2d-v5": {
        "sac_steps": 500_000,
        "pets_steps": 150_000,
        "hybrid_steps": 15000,
        "warmup_steps": 10000,
        "horizon": 15,
        "model_train_freq": 2000,
        "eval_freq": 25000,
        "expected_sac": 2869,
        "expected_pets": 34,
        "dynamics": "contact",
        "description": "Bipedal walking - complex contact dynamics",
    },
    
    "HalfCheetah-v5": {
        "sac_steps": 500_000,
        "pets_steps": 150_000,
        "hybrid_steps": 15000,
        "warmup_steps": 10000,
        "horizon": 10,
        "model_train_freq": 2000,
        "eval_freq": 25000,
        "expected_sac": 7446,
        "expected_pets": 5,
        "dynamics": "contact",
        "description": "Quadruped running - mix of smooth and contact",
    },
    
    "Pushing2D-v1": {
        "sac_steps": 100_000,
        "pets_steps": 50_000,
        "hybrid_steps": 5000,
        "warmup_steps": 2000,
        "horizon": 10,
        "model_train_freq": 1000,
        "eval_freq": 5000,
        "expected_sac": -10,
        "expected_pets": -41,
        "dynamics": "contact",
        "description": "Box pushing - contact-rich manipulation",
    },

    "Ant-v5": {
        "sac_steps": 300_000,
        "pets_steps": 100_000,
        "hybrid_steps": 150_000,
        "warmup_steps": 5000,
        "horizon": 10,
        "model_train_freq": 2000,
        "eval_freq": 15000,
        "expected_sac": 3000,
        "expected_pets": 1500,
        "dynamics": "smooth-contact",
        "description": "Quadruped locomotion - 8 legs, mix of smooth motion and ground contact, more stable than biped",
        "expected_mb_usage": "30-45%",
        "plan_every_n_steps": 5,
        "cem_iterations": 3,
        "cem_candidates": 400,
        "cem_top_candidates": 40,
    }
}

ENV_CONFIGS_IMPROVED = {
    "Pendulum-v1": {
        "sac_steps": 100_000,
        "hybrid_steps": 50_000,
        "warmup_steps": 2000,
        "horizon": 5,  # Shorter!
        "model_train_freq": 1000,
        "eval_freq": 5000,
        "expected_sac": -150,
        "dynamics": "smooth",
        "expected_mb_usage": "60-70%",
        "expected_pets": -200,
    },
    "LunarLanderContinuous-v3": {
        "sac_steps": 200_000,
        "hybrid_steps": 100_000,
        "warmup_steps": 5000,
        "horizon": 5,
        "model_train_freq": 2000,
        "eval_freq": 10000,
        "expected_sac": 270,
        "dynamics": "contact",
        "expected_mb_usage": "20-30%",
        "expected_pets": -100,
    },
    "Pushing2D-v1": {
        "sac_steps": 100_000,
        "hybrid_steps": 50_000,
        "warmup_steps": 3000,
        "horizon": 5,
        "model_train_freq": 1500,
        "eval_freq": 5000,
        "expected_sac": -5,
        "dynamics": "contact",
        "expected_mb_usage": "25-35%",
        "expected_pets": -10,
    },
}

def train_sac(env_id, config, seed=0):
    """Train SAC from scratch on any environment"""
    print(f"\n{'='*70}")
    print(f"Training SAC on {env_id}")
    print(f"{'='*70}")
    print(f"Dynamics: {config['dynamics']} - {config['description']}")
    print(f"Steps: {config['sac_steps']:,}")
    
    if env_id == "Pushing2D-v1":
        env = make_pushing2d_env()
        eval_env = make_pushing2d_env()
    else:
        env = gym.make(env_id)
        eval_env = gym.make(env_id)
    
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env_info = get_env_info(env_id)
    if env_info is None:
        return None
    
    device = get_device()
    agent = SACAgent(
        env_info,
        warmup_steps=config["warmup_steps"],
        batch_size=256,
        utd_ratio=3,
        device=device
    )
    
    obs, _ = env.reset()
    returns = []
    eval_returns = []
    ep_ret = 0
    
    for step in trange(config["sac_steps"], desc=f"SAC-{env_id}"):
        if agent.total_steps < agent.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)["action"]
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.step({
            "obs": obs,
            "action": action,
            "next_obs": next_obs,
            "reward": reward,
            "done": float(done)
        })
        
        obs = next_obs
        ep_ret += reward
        
        if done:
            returns.append(ep_ret)
            obs, _ = env.reset()
            ep_ret = 0
            
            if len(returns) % 10 == 0:
                recent = np.mean(returns[-10:])
                print(f"Episodes: {len(returns)} | Return: {recent:.2f}")
        
        if (step + 1) % config["eval_freq"] == 0:
            eval_mean, eval_std = evaluate_sac(agent, eval_env, n_episodes=10)
            eval_returns.append(eval_mean)
            print(f"Eval at {step+1}: {eval_mean:.2f} ± {eval_std:.2f}")

    final_mean, final_std = evaluate_sac(agent, eval_env, n_episodes=20)
    
    print(f"\n{'='*70}")
    print(f"SAC Final: {final_mean:.2f} ± {final_std:.2f}")
    print(f"{'='*70}")

    os.makedirs("artifacts", exist_ok=True)
    ckpt_path = f"artifacts/sac_{env_id.replace('-', '_').lower()}.pth"
    torch.save({
        "actor": agent.actor.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
    }, ckpt_path)
    print(f"Saved to {ckpt_path}")
    
    if hasattr(env, 'close'):
        env.close()
    if hasattr(eval_env, 'close'):
        eval_env.close()
    
    return {
        "returns": returns,
        "eval_returns": eval_returns,
        "final_mean": final_mean,
        "final_std": final_std,
        "checkpoint": ckpt_path
    }

def evaluate_sac(agent, env, n_episodes=10):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret = 0
        done = False
        while not done:
            action = agent.act(obs, deterministic=True)["action"]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)
    return np.mean(returns), np.std(returns)

def train_pets_baseline(env_id, config, seed=0):
    print(f"\n{'='*70}")
    print(f"Training PETS Baseline on {env_id}")
    print(f"{'='*70}")
    
    if env_id == "Pushing2D-v1":
        env = make_pushing2d_env()
    else:
        env = gym.make(env_id)
    
    env.reset(seed=seed)
    np.random.seed(seed)
    
    env_info = get_env_info(env_id)
    if env_info is None:
        return None
    
    device = get_device()
    agent = PETSAgent(
        env_info,
        n_models=2,
        horizon=config["horizon"],
        n_candidates=500,
        n_elite=50,
        device=device
    )
    
    print(f"\n1. Collecting {config['warmup_steps']} random transitions...")
    obs, _ = env.reset()
    for _ in trange(config["warmup_steps"], desc="Random data"):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.add_data(obs, action, next_obs, reward)
        obs = next_obs if not done else env.reset()[0]
    
    print(f"Collected {agent.buffer.size} samples")
    
    print("\n2. Training models...")
    agent.train_models(n_epochs=100)
    print("Models trained")
    
    print("\n3. Testing PETS planning (500 episodes)...")
    returns = []
    for ep in trange(500, desc="PETS episodes"):
        obs, _ = env.reset()
        ep_ret = 0
        done = False
        while not done:
            result = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(result["action"])
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)
        
        if (ep + 1) % 10 == 0:
            agent.train_models(n_epochs=25)
    
    final_mean = np.mean(returns[-20:])
    final_std = np.std(returns[-20:])
    
    print(f"\n{'='*70}")
    print(f"PETS Final: {final_mean:.2f} ± {final_std:.2f}")
    print(f"{'='*70}")
    
    if hasattr(env, 'close'):
        env.close()
    
    return {
        "returns": returns,
        "final_mean": final_mean,
        "final_std": final_std
    }

def train_pets_improved(env_id, config, seed=0):
    """
    Improved PETS with active learning (like homework)
    """
    print(f"\n{'='*70}")
    print(f"Training Improved PETS on {env_id}")
    print(f"{'='*70}")
    
    if env_id == "Pushing2D-v1":
        env = make_pushing2d_env()
    else:
        env = gym.make(env_id)
    
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env_info = get_env_info(env_id)
    if env_info is None:
        return None
    
    device = get_device()
    
    agent = ImprovedPETSAgent(
        env_info,
        n_models=2,          # Homework uses 2
        horizon=5,           # Homework uses 5  
        n_candidates=200,    # Homework uses 200
        n_elite=20,          # Homework uses 20
        n_iterations=5,      # Homework uses 5
        num_particles=6,     # TS1 with 6 particles
        device=device
    )
    print(f"\n1. Warmup with random policy (100 episodes)...")
    warmup_returns = []
    
    for ep in trange(100, desc="Warmup"):
        obs, _ = env.reset()
        ep_return = 0
        
        for t in range(40):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.add_data(obs, action, next_obs, reward)
            
            obs = next_obs
            ep_return += reward
            
            if done:
                break
        
        warmup_returns.append(ep_return)
    
    print(f"Warmup: {np.mean(warmup_returns):.2f} ± {np.std(warmup_returns):.2f}")
    print(f"Buffer size: {agent.buffer.size}")
    
    print("\n2. Training models on warmup data...")
    losses = agent.train_models(n_epochs=10, batch_size=128)
    print(f"Initial model loss: {losses[-1]:.4f}")
    
    print(f"\n3. Active learning (100 iterations)...")
    training_returns = []
    eval_returns = []
    
    for iter in trange(100, desc="Active Learning"):
        obs, _ = env.reset()
        ep_return = 0
        
        for t in range(40):
            try:
                action = agent.plan(obs)
            except:
                action = env.action_space.sample()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.add_data(obs, action, next_obs, reward)
            
            obs = next_obs
            ep_return += reward
            
            if done:
                break
        
        training_returns.append(ep_return)
        losses = agent.train_models(n_epochs=10, batch_size=128)
        if losses and len(losses) > 0:
            print(f"Initial model loss: {losses[-1]:.4f}")
        else:
            print(f"Models trained (no loss reported)")
        
        if (iter + 1) % 10 == 0:
            eval_eps = []
            for _ in range(5):
                obs, _ = env.reset()
                ep_ret = 0
                for t in range(40):
                    try:
                        action = agent.plan(obs)
                    except:
                        action = env.action_space.sample()
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_ret += reward
                    if terminated or truncated:
                        break
                eval_eps.append(ep_ret)
            
            eval_mean = np.mean(eval_eps)
            eval_returns.append(eval_mean)
            
            recent_train = np.mean(training_returns[-10:])
            print(f"\n  Iter {iter+1}: Train={recent_train:.1f}, Eval={eval_mean:.1f}, Loss={losses[-1]:.4f}")
    
    print("\n4. Final evaluation (20 episodes)...")
    final_returns = []
    
    for ep in trange(20, desc="Final eval"):
        obs, _ = env.reset()
        ep_return = 0
        
        for t in range(40):
            try:
                action = agent.plan(obs)
            except:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            
            if terminated or truncated:
                break
        
        final_returns.append(ep_return)
    
    final_mean = np.mean(final_returns)
    final_std = np.std(final_returns)
    
    print(f"\n{'='*70}")
    print(f"Improved PETS Final: {final_mean:.2f} ± {final_std:.2f}")
    print(f"{'='*70}")
    
    if hasattr(env, 'close'):
        env.close()
    
    return {
        "returns": training_returns,
        "eval_returns": eval_returns,
        "final_returns": final_returns,
        "final_mean": final_mean,
        "final_std": final_std
    }

def train_hybrid(env_id, sac_ckpt, config, seed=0):
    print(f"\n{'='*70}")
    print(f"Training Hybrid on {env_id}")
    print(f"{'='*70}")
    
    if env_id == "Pushing2D-v1":
        env = make_pushing2d_env()
        eval_env = make_pushing2d_env()
    else:
        env = gym.make(env_id)
        eval_env = gym.make(env_id)
    
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)
    np.random.seed(seed)
    
    env_info = get_env_info(env_id)
    if env_info is None:
        return None

    dynamics_type = config.get("dynamics", "smooth")
    if dynamics_type == "smooth":
        base_threshold = 0.05  # Low for smooth environments (Pendulum)
    elif dynamics_type == "contact":
        base_threshold = 0.40  # HIGHER for contact (was 0.18, now 0.25!)
    elif dynamics_type == "smooth-contact":
        base_threshold = 0.15  # Medium for mixed dynamics (was 0.12)
    else:
        base_threshold = 0.12  # Default fallback

    print(f"Dynamics type: {dynamics_type}")
    print(f"Using disagreement threshold: {base_threshold}")

    device = get_device()
    agent = HybridAgent(
        env_info,
        n_models=2,
        horizon=config["horizon"],
        sac_ckpt=sac_ckpt,
        disagreement_threshold=base_threshold,
        calibration_quantile=0.9,
        device=device
    )
    
    print(f"\n1. Warmup ({config['warmup_steps']} steps)")
    obs, _ = env.reset()
    for _ in trange(config["warmup_steps"], desc="Warmup"):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.add_data(obs, action, next_obs, reward, done)
        obs = next_obs if not done else env.reset()[0]
    
    print("\n2. Training models...")
    agent.train_models(n_epochs=100)
    print("Models trained")

    print(f"\n3. Hybrid training ({config['hybrid_steps']} steps)")
    obs, _ = env.reset()
    returns = []
    eval_returns = []
    mb_usage_history = []
    ep_ret = 0
    
    for step in trange(config["hybrid_steps"], desc="Hybrid"):
        result = agent.act(obs, mode="mixed")
        next_obs, reward, terminated, truncated, _ = env.step(result["action"])
        done = terminated or truncated
        
        agent.add_data(obs, result["action"], next_obs, reward, done)
        obs = next_obs
        ep_ret += reward
        
        if done:
            returns.append(ep_ret)
            obs, _ = env.reset()
            ep_ret = 0
            
            if len(returns) % 5 == 0:
                stats = agent.get_stats()
                recent = np.mean(returns[-10:])
                print(f"\n  Ep {len(returns)} | Return: {recent:.2f} | "
                      f"MB: {stats['model_based_pct']:.1f}%")
        
        if (step + 1) % config["model_train_freq"] == 0:
            agent.train_models(n_epochs=50)
        
        if (step + 1) % (config["model_train_freq"] * 2) == 0:
            agent.calibrate_threshold()

        if (step + 1) % config["eval_freq"] == 0:
            eval_mean, eval_std = evaluate_hybrid(agent, eval_env, n_episodes=10)
            eval_returns.append(eval_mean)
            stats = agent.get_stats()
            mb_usage_history.append(stats['model_based_pct'])
            print(f"Eval: {eval_mean:.2f} | MB: {stats['model_based_pct']:.1f}%")

    final_stats = agent.get_stats()
    final_mean = np.mean(returns[-20:])
    final_std = np.std(returns[-20:])
    
    print(f"\n{'='*70}")
    print(f"Hybrid Final: {final_mean:.2f} ± {final_std:.2f}")
    print(f"MB Usage: {final_stats['model_based_pct']:.1f}%")
    print(f"{'='*70}")
    
    if hasattr(env, 'close'):
        env.close()
    if hasattr(eval_env, 'close'):
        eval_env.close()
    
    return {
        "returns": returns,
        "eval_returns": eval_returns,
        "mb_usage_history": mb_usage_history,
        "final_mean": final_mean,
        "final_std": final_std,
        "final_mb_pct": final_stats['model_based_pct']
    }

def evaluate_hybrid(agent, env, n_episodes=10):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret = 0
        done = False
        while not done:
            result = agent.sac.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(result["action"])
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)
    return np.mean(returns), np.std(returns)

def run_full_experiment(env_id, seed=0):
    config = ENV_CONFIGS.get(env_id)
    if config is None:
        print(f"Unknown environment: {env_id}")
        print(f"Available: {list(ENV_CONFIGS.keys())}")
        return None
    
    print("\n" + "="*70)
    print(f"FULL EXPERIMENT: {env_id}")
    print("="*70)
    print(f"Dynamics: {config['dynamics']}")
    print(f"Description: {config['description']}")
    print(f"Expected MB usage: {config['expected_mb_usage']}")
    print("="*70)
    
    results = {"env_id": env_id, "config": config}

    print("\n" + "-"*35)
    print("STEP 1: Training SAC Baseline")
    print("-"*35)
    sac_results = train_sac(env_id, config, seed)
    if sac_results is None:
        return None
    results["sac"] = sac_results

    print("\n" + "-"*35)
    print("STEP 2: Training PETS Baseline")
    print("-"*35)
    pets_results = train_pets_improved(env_id, config, seed)
    if pets_results is None:
        return None
    results["pets"] = pets_results

    print("\n" + "-"*35)
    print("STEP 3: Training Hybrid")
    print("-"*35)
    hybrid_results = train_hybrid(env_id, sac_results["checkpoint"], config, seed)
    if hybrid_results is None:
        return None
    results["hybrid"] = hybrid_results

    print("\n" + "-"*70)
    print(f"FINAL RESULTS: {env_id}")
    print("="*70)
    print(f"SAC: {sac_results['final_mean']:7.2f} ± {sac_results['final_std']:.2f}")
    print(f"PETS: {pets_results['final_mean']:7.2f} ± {pets_results['final_std']:.2f}")
    print(f"Hybrid: {hybrid_results['finql_mean']:7.2f} ± {hybrid_results['final_std']:.2f}")
    print(f"\nMB Usage: {hybrid_results['final_mb_pct']:.1f}%")
    print(f"Expected: {config['expected_mb_usage']}")
    print("-"*70)

    os.makedirs("results", exist_ok=True)
    save_path = f"results/{env_id.replace('-', '_').lower()}_full.npz"
    np.savez(save_path,
             env_id=env_id,
             sac_final=sac_results['final_mean'],
             pets_final=pets_results['final_mean'],
             hybrid_final=hybrid_results['final_mean'],
             hybrid_mb_pct=hybrid_results['final_mb_pct'],
             hybrid_mb_history=hybrid_results['mb_usage_history'])
    print(f"Saved to {save_path}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL agents")
    parser.add_argument("--envs", nargs="+", 
                        default=["Pendulum-v1", "LunarLanderContinuous-v3"],
                        help="Environments to train on")
    parser.add_argument("--only", 
                        choices=["sac", "pets", "hybrid", "all"], 
                        default="all",
                        help="Which component to train")
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()

    print("\n" + "-"*70)
    print("AVAILABLE ENVIRONMENTS")
    print("-"*70)
    for env_id, config in ENV_CONFIGS.items():
        print(f"\n{env_id}:")
        print(f"Dynamics: {config['dynamics']}")
        print(f"{config['description']}")
    print("-"*70)

    all_results = {}
    
    for env_id in args.envs:
        config = ENV_CONFIGS.get(env_id)
        if config is None:
            print(f"Unknown environment: {env_id}")
            continue
        
        results = {"env_id": env_id, "config": config}
        if args.only == "sac":
            print(f"\n{'='*70}")
            print(f"Training SAC on {env_id}")
            print(f"{'='*70}")
            sac_results = train_sac(env_id, config, args.seed)
            if sac_results:
                results["sac"] = sac_results
                print(f"\nSAC: {sac_results['final_mean']:.2f} ± {sac_results['final_std']:.2f}")

        elif args.only == "pets":
            print(f"\n{'='*70}")
            print(f"Training PETS on {env_id}")
            print(f"{'='*70}")
            pets_results = train_pets_improved(env_id, config, args.seed)
            if pets_results:
                results["pets"] = pets_results
                print(f"PETS: {pets_results['final_mean']:.2f} ± {pets_results['final_std']:.2f}")

        elif args.only == "hybrid":
            print(f"\n{'='*70}")
            print(f"Training Hybrid on {env_id}")
            print(f"{'='*70}")

            sac_ckpt = f"artifacts/sac_{env_id.replace('-', '_').lower()}.pth"
            
            if not os.path.exists(sac_ckpt):
                print(f"SAC checkpoint not found: {sac_ckpt}")
                print(f"Run with --only sac first, or --only all")
                continue
            
            print(f"Found SAC checkpoint: {sac_ckpt}")

            results_path = f"results/{env_id.replace('-', '_').lower()}_full.npz"
            if os.path.exists(results_path):
                saved = np.load(results_path, allow_pickle=True)
                sac_mean = float(saved['sac_final'])
                print(f"SAC baseline: {sac_mean:.2f} (from saved results)")

            hybrid_results = train_hybrid(env_id, sac_ckpt, config, args.seed)
            if hybrid_results:
                results["hybrid"] = hybrid_results
                print(f"Hybrid: {hybrid_results['final_mean']:.2f} ± {hybrid_results['final_std']:.2f}")
                print(f"MB Usage: {hybrid_results['final_mb_pct']:.1f}%")

        elif args.only == "all":
            print(f"\n{'='*70}")
            print(f"FULL EXPERIMENT: {env_id}")
            print(f"{'='*70}")

            exp_results = run_full_experiment(env_id, seed=args.seed)
            if exp_results:
                results = exp_results

        if results:
            all_results[env_id] = results

    if len(all_results) > 0:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        if args.only == "all":
            print(f"{'Environment':<30} {'SAC':<12} {'PETS':<12} {'Hybrid':<12} {'MB%'}")
            print("-"*70)
            for env_id, res in all_results.items():
                sac = res.get('sac', {}).get('final_mean', 0)
                pets = res.get('pets', {}).get('final_mean', 0)
                hybrid = res.get('hybrid', {}).get('final_mean', 0)
                mb_pct = res.get('hybrid', {}).get('final_mb_pct', 0)
                print(f"{env_id:<30} {sac:>10.1f}  {pets:>10.1f}  {hybrid:>10.1f}  {mb_pct:>6.1f}%")
        else:
            for env_id, res in all_results.items():
                print(f"\n{env_id}:")
                if args.only == "sac" and "sac" in res:
                    print(f"SAC: {res['sac']['final_mean']:.2f} ± {res['sac']['final_std']:.2f}")
                elif args.only == "pets" and "pets" in res:
                    print(f"PETS: {res['pets']['final_mean']:.2f} ± {res['pets']['final_std']:.2f}")
                elif args.only == "hybrid" and "hybrid" in res:
                    print(f"Hybrid: {res['hybrid']['final_mean']:.2f} ± {res['hybrid']['final_std']:.2f}")
                    print(f"MB Usage: {res['hybrid']['final_mb_pct']:.1f}%")
        
        print("="*70)