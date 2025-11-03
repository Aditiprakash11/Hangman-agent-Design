import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.hmm_model import HMMHangmanOracle
from models.rl_agent import HangmanRLAgent
from utils.environment import HangmanEnvironment  # IMPORTANT: Import environment
from utils.visualization import plot_training_results

def main():
    print("="*60)
    print("TRAINING RL AGENT")
    print("="*60)
    
    # Load corpus
    with open('data/corpus.txt', 'r') as f:
        words = [line.strip().lower() for line in f if line.strip().isalpha()]
    print(f"\n✓ Loaded {len(words)} words")
    
    # Load HMM
    oracle = HMMHangmanOracle()
    oracle.load('saved_models/hmm_oracle.pkl')
    print("✓ Loaded HMM oracle")
    
    # CREATE ENVIRONMENT (This was missing!)
    env = HangmanEnvironment(words, max_wrong_guesses=6)
    print("✓ Created Hangman environment")
    
    # Train agent - pass environment, not word list
    agent = HangmanRLAgent(oracle)
    rewards, successes, losses = agent.train(env, episodes=2000)
    
    # Save
    agent.save('saved_models/trained_agent.pth')
    plot_training_results(rewards, successes, losses)
    
    print("\n✓ RL training complete!")
    print("✓ Model saved to: saved_models/trained_agent.pth")
    print("✓ Plots saved to: results/training_results.png")

if __name__ == "__main__":
    main()