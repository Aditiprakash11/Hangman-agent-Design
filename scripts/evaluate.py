import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.hmm_model import HMMHangmanOracle
from models.rl_agent import HangmanRLAgent
from utils.environment import HangmanEnvironment  # IMPORTANT: Import environment
from utils.visualization import plot_evaluation_results

def main():
    print("="*60)
    print("EVALUATING RL AGENT")
    print("="*60)
    
    # Load corpus
    with open('data/corpus.txt', 'r') as f:
        words = [line.strip().lower() for line in f if line.strip().isalpha()]
    print(f"\n✓ Loaded {len(words)} words")
    
    # Load models
    oracle = HMMHangmanOracle()
    oracle.load('saved_models/hmm_oracle.pkl')
    print("✓ Loaded HMM oracle")
    
    agent = HangmanRLAgent(oracle)
    agent.load('saved_models/trained_agent.pth')
    print("✓ Loaded trained agent")
    
    # CREATE ENVIRONMENT (This was missing!)
    env = HangmanEnvironment(words, max_wrong_guesses=6)
    print("✓ Created Hangman environment")
    
    # Evaluate - pass environment, not word list
    results = agent.evaluate(env, num_games=2000)
    
    # Save results
    with open('results/evaluation_results.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print("\n✓ Evaluation complete!")
    print("✓ Results saved to: results/evaluation_results.txt")
    
    # Plot results if visualization is available
    try:
        plot_evaluation_results(results)
        print("✓ Plots saved to: results/evaluation_summary.png")
    except Exception as e:
        print(f"Note: Could not generate plots: {e}")

if __name__ == "__main__":
    main()