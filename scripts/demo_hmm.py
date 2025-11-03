import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.hmm_model import HMMHangmanOracle
import matplotlib.pyplot as plt
import numpy as np

def visualize_hmm_probabilities(oracle, masked_word, guessed_letters):
    """Visualize HMM probability distribution for a game state"""
    
    # Get probabilities from HMM
    probs = oracle.get_letter_probabilities(masked_word, guessed_letters)
    
    # Sort by probability (descending)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print(f"HMM Probability Analysis")
    print("="*60)
    print(f"Masked Word: {masked_word}")
    print(f"Guessed Letters: {sorted(guessed_letters)}")
    print(f"Word Length: {len(masked_word)}")
    print("\n" + "-"*60)
    print("Top 10 Letter Predictions:")
    print("-"*60)
    
    for i, (letter, prob) in enumerate(sorted_probs[:10], 1):
        bar = "█" * int(prob * 50)  # Visual bar
        print(f"{i:2d}. {letter.upper()}: {prob:6.4f} {bar}")
    
    print("-"*60)
    
    return probs


def plot_hmm_distribution(probs, masked_word, guessed_letters):
    """Create a bar plot of HMM probabilities"""
    
    letters = list(probs.keys())
    probabilities = list(probs.values())
    
    # Sort by probability
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_letters = [letters[i] for i in sorted_indices]
    sorted_probs = [probabilities[i] for i in sorted_indices]
    
    # Color code: red for guessed, green for available
    colors = ['red' if letter in guessed_letters else 'green' for letter in sorted_letters]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(26), sorted_probs, color=colors, alpha=0.7)
    plt.xticks(range(26), [l.upper() for l in sorted_letters], fontsize=10)
    plt.xlabel('Letters', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'HMM Letter Probabilities for "{masked_word}"\n(Red=Already Guessed, Green=Available)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Available Letters'),
        Patch(facecolor='red', alpha=0.7, label='Already Guessed')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('results/hmm_probabilities_demo.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved probability plot to: results/hmm_probabilities_demo.png")
    plt.show()


def interactive_demo(oracle):
    """Interactive demo to test HMM with different game states"""
    
    print("\n" + "="*60)
    print("INTERACTIVE HMM DEMO")
    print("="*60)
    print("Test the HMM with different Hangman game states!")
    print("Format: masked_word,guessed_letters")
    print("Example: _pp_e,rstn")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        user_input = input("\nEnter game state (or 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        try:
            if ',' in user_input:
                masked_word, guessed_str = user_input.split(',')
                guessed_letters = set(guessed_str.strip())
            else:
                masked_word = user_input
                guessed_letters = set()
            
            masked_word = masked_word.strip()
            
            # Get and visualize probabilities
            probs = visualize_hmm_probabilities(oracle, masked_word, guessed_letters)
            
            # Show recommendation
            available = {k: v for k, v in probs.items() if k not in guessed_letters}
            if available:
                best_letter = max(available.items(), key=lambda x: x[1])
                print(f"\n💡 HMM Recommendation: Guess '{best_letter[0].upper()}' (probability: {best_letter[1]:.4f})")
            
        except Exception as e:
            print(f"Error: {e}. Please use format: masked_word,guessed_letters")


def main():
    print("\n" + "="*60)
    print("HMM PROBABILITY DEMONSTRATION")
    print("="*60)
    
    # Load trained HMM
    print("\nLoading HMM oracle...")
    oracle = HMMHangmanOracle()
    oracle.load('saved_models/hmm_oracle.pkl')
    print("✓ HMM loaded successfully!")
    
    # Example test cases
    test_cases = [
        ("_pp__", set(['e', 'r', 's', 't'])),
        ("h_ll_", set(['e', 'a', 'i'])),
        ("____", set(['e'])),
        ("_a__a__", set(['e', 'i', 'o'])),
        ("pr___a_", set(['e', 'i', 'o', 's'])),
    ]
    
    print("\n" + "="*60)
    print("RUNNING EXAMPLE TEST CASES")
    print("="*60)
    
    for masked_word, guessed in test_cases:
        probs = visualize_hmm_probabilities(oracle, masked_word, guessed)
        input("\nPress Enter to continue to next example...")
    
    # Plot the last example
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60)
    plot_hmm_distribution(probs, test_cases[-1][0], test_cases[-1][1])
    
    # Interactive mode
    print("\n" + "="*60)
    response = input("Would you like to try interactive mode? (y/n): ").strip().lower()
    if response == 'y':
        interactive_demo(oracle)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. HMM provides probability distribution over all 26 letters")
    print("2. Probabilities are based on positional letter patterns in corpus")
    print("3. RL agent uses these probabilities as part of its state representation")
    print("4. The combination of HMM + RL leads to intelligent guessing!")
    print("="*60)


if __name__ == "__main__":
    main()