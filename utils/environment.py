import random
from typing import List, Set, Dict

class HangmanEnvironment:
    """Hangman game environment for RL agent"""
    
    def __init__(self, word_list: List[str], max_wrong_guesses: int = 6):
        self.word_list = word_list
        self.max_wrong_guesses = max_wrong_guesses
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.reset()
    
    def reset(self):
        """Start a new game with a random word"""
        self.target_word = random.choice(self.word_list).lower()
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        self.game_over = False
        self.won = False
        return self.get_state()
    
    def get_masked_word(self):
        """Return the word with unguessed letters as '_'"""
        return ''.join([letter if letter in self.guessed_letters else '_' 
                       for letter in self.target_word])
    
    def get_state(self):
        """Return current state of the game"""
        return {
            'masked_word': self.get_masked_word(),
            'guessed_letters': self.guessed_letters.copy(),
            'wrong_guesses': self.wrong_guesses,
            'lives_remaining': self.max_wrong_guesses - self.wrong_guesses,
            'game_over': self.game_over,
            'won': self.won
        }
    
    def step(self, letter: str):
        """Make a guess and return (state, reward, done)"""
        letter = letter.lower()
        
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            reward = -5
            return self.get_state(), reward, self.game_over
        
        self.guessed_letters.add(letter)
        
        if letter in self.target_word:
            reward = 5
            if all(l in self.guessed_letters for l in self.target_word):
                self.game_over = True
                self.won = True
                reward = 100
        else:
            self.wrong_guesses += 1
            reward = -10
            if self.wrong_guesses >= self.max_wrong_guesses:
                self.game_over = True
                self.won = False
                reward = -50
        
        return self.get_state(), reward, self.game_over