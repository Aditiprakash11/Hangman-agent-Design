import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def plot_training_results(rewards, success_rates, losses, save_path='results/training_results.png'):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards
    axes[0, 0].plot(rewards, alpha=0.3, color='blue')
    axes[0, 0].plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), 
                    label='Moving Avg (100)', color='darkblue', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Reward per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Success rate
    success_ma = np.convolve(success_rates, np.ones(100)/100, mode='valid')
    axes[0, 1].plot(success_ma, color='green', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title('Success Rate (Moving Avg 100)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Loss
    if losses:
        axes[1, 0].plot(losses, alpha=0.5, color='red')
        axes[1, 0].plot(np.convolve(losses, np.ones(100)/100, mode='valid'), 
                       label='Moving Avg (100)', color='darkred', linewidth=2)
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Summary
    axes[1, 1].axis('off')
    final_success = np.mean(success_rates[-100:])
    final_reward = np.mean(rewards[-100:])
    summary_text = f"""
    Final Training Performance (Last 100):
    
    Success Rate: {final_success:.2%}
    Average Reward: {final_reward:.2f}
    Total Episodes: {len(rewards)}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=14, 
                   verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved training plots to '{save_path}'")
    plt.close()

def plot_evaluation_results(results, save_path='results/evaluation_summary.png'):
    """Plot evaluation summary"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Metrics bar chart
    metrics = ['Success\nRate (%)', 'Avg Wrong\nGuesses', 'Avg Repeated\nGuesses']
    values = [
        results['success_rate'] * 100,
        results['avg_wrong_guesses'],
        results['avg_repeated_guesses']
    ]
    colors = ['green', 'orange', 'red']
    
    axes[0].bar(metrics, values, color=colors, alpha=0.7)
    axes[0].set_ylabel('Value')
    axes[0].set_title('Evaluation Metrics')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Score breakdown
    score_components = {
        'Success\nBonus': results['success_rate'] * 2000,
        'Wrong\nPenalty': -results['total_wrong_guesses'] * 5,
        'Repeated\nPenalty': -results['total_repeated_guesses'] * 2
    }
    
    axes[1].bar(score_components.keys(), score_components.values(), 
               color=['green', 'red', 'orange'], alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].set_ylabel('Score Contribution')
    axes[1].set_title(f'Final Score Breakdown\n(Total: {results["final_score"]:.1f})')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved evaluation plots to '{save_path}'")
    plt.close()