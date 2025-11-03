import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hmm_model import HMMHangmanOracle

def main():
    print("="*60)
    print("TRAINING HIDDEN MARKOV MODEL")
    print("="*60)
    
    oracle = HMMHangmanOracle()
    oracle.train('data/corpus.txt')
    oracle.save('saved_models/hmm_oracle.pkl')
    
    print("\n✓ HMM training complete!")
    print("✓ Model saved to: saved_models/hmm_oracle.pkl")

if __name__ == "__main__":
    main()