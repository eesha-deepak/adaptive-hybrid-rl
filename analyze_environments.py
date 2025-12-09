"""
Quantitative Environment Analysis
Characterize when environments are amenable to model-based planning

Metrics:
1. Contact frequency (how often contact events occur)
2. State-space smoothness (local Lipschitz constants)
3. Reward landscape complexity (variance and gradients)
"""
import numpy as np
import gymnasium as gym
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from pusher2d import Pusher2d
from gym_wrapper import OldGymWrapper


def make_env(env_id, seed=0):
    """Create environment"""
    if env_id == "Pushing2D-v1":
        env = OldGymWrapper(Pusher2d())
    else:
        env = gym.make(env_id)
    env.reset(seed=seed)
    return env


# ============================================================================
# METRIC 1: CONTACT FREQUENCY
# ============================================================================

def analyze_contact_frequency(env_id, n_episodes=50):
    """
    Measure how often contact events occur
    
    Contact detection:
    - Velocity discontinuities (sudden changes)
    - Large acceleration (v_t - v_{t-1})
    - Environment-specific contact signals
    """
    env = make_env(env_id)
    
    contact_counts = []
    total_steps = 0
    
    print(f"\n📊 Analyzing contact frequency: {env_id}")
    
    for episode in trange(n_episodes, desc="Episodes"):
        obs, _ = env.reset()
        done = False
        episode_contacts = 0
        prev_obs = obs
        
        while not done:
            # Random policy for exploration
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Detect contact via velocity discontinuity
            # Assumes velocity is in observation (common for MuJoCo)
            if len(obs) >= 6:  # Has velocity components
                # Extract velocity (environment-specific, adjust as needed)
                try:
                    if "Hopper" in env_id or "Walker" in env_id or "HalfCheetah" in env_id:
                        # MuJoCo envs: velocities in second half
                        vel_idx = len(obs) // 2
                        prev_vel = prev_obs[vel_idx:]
                        curr_vel = obs[vel_idx:]
                        
                        # Large acceleration = contact
                        accel = np.linalg.norm(curr_vel - prev_vel)
                        if accel > 0.5:  # Threshold for contact
                            episode_contacts += 1
                    
                    elif "LunarLander" in env_id:
                        # LunarLander: vel is index 2,3
                        prev_vel = prev_obs[2:4]
                        curr_vel = obs[2:4]
                        accel = np.linalg.norm(curr_vel - prev_vel)
                        if accel > 0.3:
                            episode_contacts += 1
                            
                except:
                    pass  # Skip if velocity extraction fails
            
            prev_obs = obs
            obs = next_obs
            total_steps += 1
        
        contact_counts.append(episode_contacts)
    
    env.close()
    
    # Statistics
    mean_contacts = np.mean(contact_counts)
    contact_freq = mean_contacts / (total_steps / n_episodes)  # Contacts per step
    
    return {
        "mean_contacts_per_episode": mean_contacts,
        "contact_frequency": contact_freq,  # Contacts per timestep
        "total_episodes": n_episodes,
        "std": np.std(contact_counts)
    }


# ============================================================================
# METRIC 2: STATE-SPACE SMOOTHNESS (LIPSCHITZ CONSTANTS)
# ============================================================================

def estimate_lipschitz_constant(env_id, n_samples=1000):
    """
    Estimate local Lipschitz constant of dynamics
    
    L = max ||f(s,a) - f(s',a')|| / ||[s,a] - [s',a']||
    
    Where f(s,a) = s' (next state)
    
    High L = non-smooth (hard to model)
    Low L = smooth (easy to model)
    """
    env = make_env(env_id)
    
    lipschitz_estimates = []
    
    print(f"\n📊 Estimating Lipschitz constant: {env_id}")
    
    # Collect (s,a,s') tuples
    transitions = []
    obs, _ = env.reset()
    
    for _ in trange(n_samples, desc="Collecting transitions"):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        transitions.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'next_obs': next_obs.copy()
        })
        
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    
    # Estimate Lipschitz constant
    print("Computing Lipschitz estimates...")
    for i in range(min(500, len(transitions) - 1)):
        t1 = transitions[i]
        
        # Find nearby transition
        for j in range(i + 1, min(i + 10, len(transitions))):
            t2 = transitions[j]
            
            # Input distance: ||[s,a] - [s',a']||
            input_diff = np.concatenate([
                t1['obs'] - t2['obs'],
                t1['action'] - t2['action']
            ])
            input_dist = np.linalg.norm(input_diff)
            
            if input_dist < 1e-6:
                continue
            
            # Output distance: ||s_next - s'_next||
            output_dist = np.linalg.norm(t1['next_obs'] - t2['next_obs'])
            
            # Lipschitz estimate
            L = output_dist / input_dist
            lipschitz_estimates.append(L)
    
    return {
        "mean_lipschitz": np.mean(lipschitz_estimates),
        "median_lipschitz": np.median(lipschitz_estimates),
        "max_lipschitz": np.max(lipschitz_estimates),
        "std_lipschitz": np.std(lipschitz_estimates),
        "smoothness_score": 1.0 / (1.0 + np.mean(lipschitz_estimates))  # Higher = smoother
    }


# ============================================================================
# METRIC 3: REWARD LANDSCAPE COMPLEXITY
# ============================================================================

def analyze_reward_landscape(env_id, n_samples=1000):
    """
    Analyze reward landscape complexity
    
    Metrics:
    - Reward variance (how much reward changes)
    - Reward gradient smoothness
    - Sparse vs dense rewards
    """
    env = make_env(env_id)
    
    rewards = []
    reward_gradients = []
    prev_reward = 0
    
    print(f"\n📊 Analyzing reward landscape: {env_id}")
    
    obs, _ = env.reset()
    for _ in trange(n_samples, desc="Sampling rewards"):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        rewards.append(reward)
        
        # Reward gradient (change in reward)
        reward_grad = abs(reward - prev_reward)
        reward_gradients.append(reward_grad)
        prev_reward = reward
        
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            prev_reward = 0
    
    env.close()
    
    # Statistics
    rewards = np.array(rewards)
    reward_gradients = np.array(reward_gradients)
    
    # Sparsity: what fraction of rewards are zero?
    sparsity = np.mean(rewards == 0)
    
    return {
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "reward_variance": np.var(rewards),
        "reward_gradient_mean": np.mean(reward_gradients),
        "reward_gradient_std": np.std(reward_gradients),
        "reward_sparsity": sparsity,
        "complexity_score": np.std(reward_gradients)  # Higher = more complex
    }


# ============================================================================
# COMBINED ANALYSIS
# ============================================================================

def analyze_all_environments(env_ids):
    """Run all analyses on all environments"""
    
    results = {}
    
    for env_id in env_ids:
        print(f"\n{'='*70}")
        print(f"ANALYZING: {env_id}")
        print(f"{'='*70}")
        
        try:
            results[env_id] = {
                "contact": analyze_contact_frequency(env_id, n_episodes=30),
                "smoothness": estimate_lipschitz_constant(env_id, n_samples=500),
                "reward": analyze_reward_landscape(env_id, n_samples=500)
            }
        except Exception as e:
            print(f"❌ Error analyzing {env_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def create_analysis_table(results):
    """Create summary table of environment characteristics"""
    
    print("\n" + "="*100)
    print("ENVIRONMENT CHARACTERISTICS SUMMARY")
    print("="*100)
    print(f"{'Environment':<30} {'Contact Freq':<15} {'Smoothness':<15} {'Reward Var':<15} {'MB Amenable?':<15}")
    print("-"*100)
    
    for env_id, data in results.items():
        contact_freq = data["contact"]["contact_frequency"]
        smoothness = data["smoothness"]["smoothness_score"]
        reward_var = data["reward"]["reward_variance"]
        
        # Heuristic: amenable if smooth + low contact + stable rewards
        amenable = (smoothness > 0.3 and contact_freq < 0.1 and reward_var < 100)
        amenable_str = "✅ YES" if amenable else "❌ NO"
        
        print(f"{env_id:<30} {contact_freq:<15.4f} {smoothness:<15.4f} {reward_var:<15.2f} {amenable_str:<15}")
    
    print("="*100)


def plot_environment_comparison(results, save_path='results/environment_analysis.png'):
    """Create visualization comparing environment characteristics"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    envs = list(results.keys())
    
    # Extract metrics
    contact_freqs = [results[e]["contact"]["contact_frequency"] for e in envs]
    smoothness = [results[e]["smoothness"]["smoothness_score"] for e in envs]
    reward_vars = [results[e]["reward"]["reward_variance"] for e in envs]
    
    # Color by dynamics type
    smooth_envs = ['Pendulum-v1', 'InvertedPendulum-v5', 'Reacher-v5', 'Swimmer-v5']
    colors = ['#2ecc71' if e in smooth_envs else '#e74c3c' for e in envs]
    
    # Plot 1: Contact Frequency
    ax = axes[0]
    ax.bar(range(len(envs)), contact_freqs, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Environment', fontweight='bold')
    ax.set_ylabel('Contact Frequency', fontweight='bold')
    ax.set_title('Contact Frequency (Lower = Better for MB)', fontweight='bold')
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels([e.replace('-', '\n') for e in envs], fontsize=9, rotation=45, ha='right')
    ax.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Smoothness
    ax = axes[1]
    ax.bar(range(len(envs)), smoothness, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Environment', fontweight='bold')
    ax.set_ylabel('Smoothness Score', fontweight='bold')
    ax.set_title('State-Space Smoothness (Higher = Better for MB)', fontweight='bold')
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels([e.replace('-', '\n') for e in envs], fontsize=9, rotation=45, ha='right')
    ax.axhline(0.3, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Reward Variance
    ax = axes[2]
    ax.bar(range(len(envs)), reward_vars, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Environment', fontweight='bold')
    ax.set_ylabel('Reward Variance', fontweight='bold')
    ax.set_title('Reward Complexity (Lower = Better for MB)', fontweight='bold')
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels([e.replace('-', '\n') for e in envs], fontsize=9, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Smooth Dynamics'),
        Patch(facecolor='#e74c3c', label='Contact Dynamics')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved plot: {save_path}")


def save_results(results, filepath='results/environment_analysis.npz'):
    """Save analysis results"""
    os.makedirs('results', exist_ok=True)
    
    # Flatten results for saving
    data = {}
    for env_id, metrics in results.items():
        prefix = env_id.replace('-', '_').lower()
        data[f'{prefix}_contact_freq'] = metrics['contact']['contact_frequency']
        data[f'{prefix}_smoothness'] = metrics['smoothness']['smoothness_score']
        data[f'{prefix}_lipschitz'] = metrics['smoothness']['mean_lipschitz']
        data[f'{prefix}_reward_var'] = metrics['reward']['reward_variance']
    
    np.savez(filepath, **data)
    print(f"✅ Saved results: {filepath}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", 
                        default=['Pendulum-v1', 'LunarLanderContinuous-v3', 'Hopper-v5'],
                        help="Environments to analyze")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("QUANTITATIVE ENVIRONMENT ANALYSIS")
    print("="*70)
    print(f"Environments: {args.envs}")
    print("="*70 + "\n")
    
    # Run analysis
    results = analyze_all_environments(args.envs)
    
    # Create summary
    create_analysis_table(results)
    
    # Save results
    save_results(results)
    
    # Create visualizations
    plot_environment_comparison(results)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/environment_analysis.npz (data)")
    print("  - results/environment_analysis.png (plot)")


if __name__ == "__main__":
    main()