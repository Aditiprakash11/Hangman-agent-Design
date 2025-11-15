import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple
import json

class PositionalHMM:
    """
    Hidden Markov Model for Hangman where:
    - Hidden states = letter positions (0 to word_length-1)
    - Emissions = letters (a-z)
    """
    
    def __init__(self, word_length: int):
        self.word_length = word_length
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_to_idx = {letter: idx for idx, letter in enumerate(self.alphabet)}
        
        # Emission probabilities: P(letter | position)
        # Shape: (word_length, 26)
        self.emission_probs = np.ones((word_length, 26)) * 1e-10  # Smoothing
        
        # Initial state probabilities (uniform for positional model)
        self.start_probs = np.ones(word_length) / word_length
        
    def train(self, words: List[str]):
        """Train the HMM on a list of words of the same length"""
        if not words:
            return
        
        # Count letter occurrences at each position
        position_counts = np.zeros((self.word_length, 26))
        
        for word in words:
            word = word.lower()
            if len(word) != self.word_length:
                continue
            
            for pos, letter in enumerate(word):
                if letter in self.letter_to_idx:
                    letter_idx = self.letter_to_idx[letter]
                    position_counts[pos, letter_idx] += 1
        
        # Convert counts to probabilities with Laplace smoothing
        for pos in range(self.word_length):
            total = position_counts[pos].sum() + 26 * 0.1  # Laplace smoothing
            self.emission_probs[pos] = (position_counts[pos] + 0.1) / total
    
    def get_letter_probabilities(self, masked_word: str, guessed_letters: set) -> Dict[str, float]:
        """
        Given a masked word (e.g., "_pp_e"), return probability distribution
        over remaining letters
        """
        # Find blank positions
        blank_positions = [i for i, char in enumerate(masked_word) if char == '_']
        
        if not blank_positions:
            return {letter: 0.0 for letter in self.alphabet}
        
        # Aggregate probabilities across blank positions
        letter_probs = np.zeros(26)
        
        for pos in blank_positions:
            if pos < self.word_length:
                letter_probs += self.emission_probs[pos]
        
        # Average across positions
        letter_probs /= len(blank_positions)
        
        # Convert to dictionary and filter out guessed letters
        probs_dict = {}
        for letter in self.alphabet:
            if letter not in guessed_letters:
                probs_dict[letter] = letter_probs[self.letter_to_idx[letter]]
            else:
                probs_dict[letter] = 0.0
        
        # Normalize
        total = sum(probs_dict.values())
        if total > 0:
            probs_dict = {k: v/total for k, v in probs_dict.items()}
        
        return probs_dict


class HMMHangmanOracle:
    """
    Collection of HMMs for different word lengths
    """
    
    def __init__(self):
        self.hmms: Dict[int, PositionalHMM] = {}
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    def train(self, corpus_file: str):
        """Train HMMs on corpus grouped by word length"""
        print("Loading corpus...")
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        print(f"Loaded {len(words)} words")
        
        # Group words by length
        words_by_length = defaultdict(list)
        for word in words:
            if word.isalpha():  # Only alphabetic words
                words_by_length[len(word)].append(word)
        
        print(f"Training HMMs for {len(words_by_length)} different word lengths...")
        
        # Train separate HMM for each word length
        for length, word_list in words_by_length.items():
            print(f"  Training HMM for length {length} ({len(word_list)} words)")
            hmm = PositionalHMM(length)
            hmm.train(word_list)
            self.hmms[length] = hmm
        
        print("Training complete!")
    
    def get_letter_probabilities(self, masked_word: str, guessed_letters: set) -> Dict[str, float]:
        """Get probability distribution for next letter to guess"""
        word_length = len(masked_word)
        
        if word_length in self.hmms:
            return self.hmms[word_length].get_letter_probabilities(masked_word, guessed_letters)
        else:
            # Fallback: uniform distribution for unguessed letters
            remaining_letters = [l for l in self.alphabet if l not in guessed_letters]
            if remaining_letters:
                prob = 1.0 / len(remaining_letters)
                return {l: prob if l in remaining_letters else 0.0 for l in self.alphabet}
            return {l: 0.0 for l in self.alphabet}
    
    def save(self, filepath: str):
        """Save trained HMMs to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.hmms, f)
        print(f"Saved HMM oracle to {filepath}")
    
    def load(self, filepath: str):
        """Load trained HMMs from file"""
        with open(filepath, 'rb') as f:
            self.hmms = pickle.load(f)
        print(f"Loaded HMM oracle from {filepath}")


# Training script
if __name__ == "__main__":
    print("="*60)
    print("PART 1: TRAINING HIDDEN MARKOV MODEL")
    print("="*60)
    
    # Initialize and train oracle
    oracle = HMMHangmanOracle()
    oracle.train('corpus.txt')
    
    # Save the trained model
    oracle.save('hmm_oracle.pkl')
    
    # Test the oracle
    print("\n" + "="*60)
    print("Testing HMM Oracle")
    print("="*60)
    
    test_cases = [
        ("_pp__", set(['e', 'r', 's', 't'])),
        ("____", set(['e'])),
        ("h_ll_", set(['e', 'a'])),
    ]
    
    for masked_word, guessed in test_cases:
        probs = oracle.get_letter_probabilities(masked_word, guessed)
        top_5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nMasked word: {masked_word}, Guessed: {guessed}")
        print(f"Top 5 letters: {top_5}")
    
    print("\n" + "="*60)
    print("HMM Training Complete!")
    print("Output file: hmm_oracle.pkl")
    print("="*60)