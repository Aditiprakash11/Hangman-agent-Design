import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from typing import List, Dict, Tuple
import pickle
import json
from collections import defaultdict
import itertools

from models.rl_agent import HangmanState, QLearningAgent, DQNAgent


class HangmanEnvironment:
    """Hangman game environment for RL training"""
    
    def __init__(self, words: List[str], max_lives: int = 6):
        self.words = words
        self.max_lives = max_lives
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.reset()
    
    def reset(self, word: str = None) -> Tuple[str, set, int]:
        """Reset environment with a new word"""
        if word is None:
            self.current_word = random.choice(self.words).lower()
        else:
            self.current_word = word.lower()
        
        self.guessed_letters = set()
        self.lives_left = self.max_lives
        self.masked_word = '_' * len(self.current_word)
        self.game_over = False
        self.won = False
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        
        return self.masked_word, self.guessed_letters, self.lives_left
    
    def step(self, letter: str) -> Tuple[str, float, bool, Dict]:
        """
        Take a step in the environment
        Returns: (masked_word, reward, done, info)
        """
        letter = letter.lower()
        
        # Check for repeated guess
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            reward = -2.0  # Penalty for repeated guess
            info = {
                'won': False,
                'lives_left': self.lives_left,
                'repeated': True,
                'wrong_guesses': self.wrong_guesses,
                'repeated_guesses': self.repeated_guesses
            }
            return self.masked_word, reward, False, info
        
        self.guessed_letters.add(letter)
        
        # Check if letter is in word
        if letter in self.current_word:
            # Correct guess - reveal letters
            new_masked = list(self.masked_word)
            revealed_count = 0
            for i, char in enumerate(self.current_word):
                if char == letter:
                    new_masked[i] = letter
                    revealed_count += 1
            self.masked_word = ''.join(new_masked)
            
            # Reward proportional to revealed letters and progress
            progress_bonus = revealed_count * 0.5
            if '_' not in self.masked_word:
                # Won the game
                self.won = True
                self.game_over = True
                reward = 10.0 + self.lives_left * 2.0  # Bonus for lives remaining
            else:
                reward = 1.0 + progress_bonus
        else:
            # Wrong guess
            self.lives_left -= 1
            self.wrong_guesses += 1
            reward = -1.0  # Penalty for wrong guess
            
            if self.lives_left <= 0:
                # Lost the game
                self.game_over = True
                reward = -10.0
        
        info = {
            'won': self.won,
            'lives_left': self.lives_left,
            'repeated': False,
            'wrong_guesses': self.wrong_guesses,
            'repeated_guesses': self.repeated_guesses
        }
        
        return self.masked_word, reward, self.game_over, info
    
    def get_available_letters(self) -> set:
        """Get letters that haven't been guessed yet"""
        return set(self.alphabet) - self.guessed_letters


class HangmanTrainer:
    """Trainer for RL agents on Hangman"""
    
    def __init__(self, hmm_oracle, agent, env: HangmanEnvironment, agent_type: str = 'qlearning'):
        self.hmm_oracle = hmm_oracle
        self.agent = agent
        self.env = env
        self.agent_type = agent_type
        
        # Training metrics
        self.episode_rewards = []
        self.episode_wins = []
        self.episode_wrong_guesses = []
        self.episode_repeated_guesses = []
    
    def train_episode(self) -> Dict:
        """Train on a single episode"""
        masked_word, guessed_letters, lives_left = self.env.reset()
        
        episode_reward = 0
        episode_steps = 0
        
        while not self.env.game_over:
            # Get HMM probabilities
            hmm_probs = self.hmm_oracle.get_letter_probabilities(masked_word, guessed_letters)
            
            # Create state
            state = HangmanState(masked_word, guessed_letters, lives_left, hmm_probs)
            
            # Get available actions
            available_letters = self.env.get_available_letters()
            
            if not available_letters:
                break
            
            # Choose action
            action = self.agent.get_action(state, available_letters)
            
            # Take action
            next_masked, reward, done, info = self.env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Get next state
            if not done:
                next_hmm_probs = self.hmm_oracle.get_letter_probabilities(
                    next_masked, self.env.guessed_letters
                )
                next_state = HangmanState(
                    next_masked, self.env.guessed_letters, 
                    self.env.lives_left, next_hmm_probs
                )
            else:
                next_state = None
            
            # Update agent
            if self.agent_type == 'qlearning':
                self.agent.update(state, action, reward, next_state, done)
            elif self.agent_type == 'dqn':
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.replay()
            
            # Update for next iteration
            masked_word = next_masked
            guessed_letters = self.env.guessed_letters
            lives_left = self.env.lives_left
        
        return {
            'reward': episode_reward,
            'won': self.env.won,
            'wrong_guesses': self.env.wrong_guesses,
            'repeated_guesses': self.env.repeated_guesses,
            'steps': episode_steps
        }
    
    def train(self, num_episodes: int, eval_interval: int = 1000):
        """Train agent for specified number of episodes"""
        print(f"\n{'='*60}")
        print(f"Training {self.agent_type.upper()} Agent")
        print(f"{'='*60}\n")
        
        for episode in range(num_episodes):
            result = self.train_episode()
            
            self.episode_rewards.append(result['reward'])
            self.episode_wins.append(1 if result['won'] else 0)
            self.episode_wrong_guesses.append(result['wrong_guesses'])
            self.episode_repeated_guesses.append(result['repeated_guesses'])
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Print progress
            if (episode + 1) % eval_interval == 0:
                recent_wins = sum(self.episode_wins[-eval_interval:])
                recent_reward = np.mean(self.episode_rewards[-eval_interval:])
                recent_wrong = np.mean(self.episode_wrong_guesses[-eval_interval:])
                recent_repeated = np.mean(self.episode_repeated_guesses[-eval_interval:])
                
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Win Rate: {recent_wins/eval_interval*100:.2f}%")
                print(f"  Avg Reward: {recent_reward:.2f}")
                print(f"  Avg Wrong Guesses: {recent_wrong:.2f}")
                print(f"  Avg Repeated Guesses: {recent_repeated:.2f}")
                print(f"  Epsilon: {self.agent.epsilon:.4f}")
                print()
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}\n")
    
    def evaluate(self, test_words: List[str]) -> Dict:
        """Evaluate agent on test set"""
        print(f"\n{'='*60}")
        print("Evaluating Agent")
        print(f"{'='*60}\n")
        
        # Temporarily set epsilon to 0 for evaluation
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        wins = 0
        total_wrong_guesses = 0
        total_repeated_guesses = 0
        
        for i, word in enumerate(test_words):
            self.env.reset(word)
            masked_word, guessed_letters, lives_left = (
                self.env.masked_word, self.env.guessed_letters, self.env.lives_left
            )
            
            while not self.env.game_over:
                hmm_probs = self.hmm_oracle.get_letter_probabilities(masked_word, guessed_letters)
                state = HangmanState(masked_word, guessed_letters, lives_left, hmm_probs)
                available_letters = self.env.get_available_letters()
                
                if not available_letters:
                    break
                
                action = self.agent.get_action(state, available_letters)
                masked_word, _, done, info = self.env.step(action)
                guessed_letters = self.env.guessed_letters
                lives_left = self.env.lives_left
            
            if self.env.won:
                wins += 1
            total_wrong_guesses += self.env.wrong_guesses
            total_repeated_guesses += self.env.repeated_guesses
            
            if (i + 1) % 500 == 0:
                print(f"Evaluated {i + 1}/{len(test_words)} words...")
        
        # Restore epsilon
        self.agent.epsilon = original_epsilon
        
        success_rate = wins / len(test_words)
        final_score = (success_rate * len(test_words)) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)
        
        results = {
            'total_games': len(test_words),
            'wins': wins,
            'success_rate': success_rate,
            'total_wrong_guesses': total_wrong_guesses,
            'avg_wrong_guesses': total_wrong_guesses / len(test_words),
            'total_repeated_guesses': total_repeated_guesses,
            'avg_repeated_guesses': total_repeated_guesses / len(test_words),
            'final_score': final_score
        }
        
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"Total Games: {results['total_games']}")
        print(f"Wins: {results['wins']}")
        print(f"Success Rate: {results['success_rate']*100:.2f}%")
        print(f"Total Wrong Guesses: {results['total_wrong_guesses']}")
        print(f"Avg Wrong Guesses: {results['avg_wrong_guesses']:.2f}")
        print(f"Total Repeated Guesses: {results['total_repeated_guesses']}")
        print(f"Avg Repeated Guesses: {results['avg_repeated_guesses']:.2f}")
        print(f"Final Score: {results['final_score']:.2f}")
        print(f"{'='*60}\n")
        
        return results


def hyperparameter_search(hmm_oracle, train_words: List[str], val_words: List[str], 
                         agent_type: str = 'qlearning'):
    """Perform grid search for best hyperparameters"""
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING FOR {agent_type.upper()}")
    print(f"{'='*60}\n")
    
    if agent_type == 'qlearning':
        param_grid = {
            'alpha': [0.05, 0.1, 0.2],
            'gamma': [0.9, 0.95, 0.99],
            'epsilon_decay': [0.995, 0.997, 0.999]
        }
    else:  # dqn
        param_grid = {
            'alpha': [0.0001, 0.001, 0.01],
            'gamma': [0.9, 0.95, 0.99],
            'epsilon_decay': [0.995, 0.997, 0.999],
            'hidden_size': [64, 128, 256]
        }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"Testing {len(combinations)} hyperparameter combinations...\n")
    
    best_score = -float('inf')
    best_params = None
    best_results = None
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n--- Combination {i+1}/{len(combinations)} ---")
        print(f"Parameters: {params}")
        
        # Create agent with these parameters
        if agent_type == 'qlearning':
            agent = QLearningAgent(
                alpha=params['alpha'],
                gamma=params['gamma'],
                epsilon=1.0,
                epsilon_decay=params['epsilon_decay'],
                epsilon_min=0.01
            )
        else:
            agent = DQNAgent(
                alpha=params['alpha'],
                gamma=params['gamma'],
                epsilon=1.0,
                epsilon_decay=params['epsilon_decay'],
                epsilon_min=0.01,
                hidden_size=params['hidden_size']
            )
        
        # Train on subset of data
        env = HangmanEnvironment(train_words, max_lives=6)
        trainer = HangmanTrainer(hmm_oracle, agent, env, agent_type=agent_type)
        
        # Quick training (fewer episodes for hyperparameter search)
        num_episodes = 5000 if agent_type == 'qlearning' else 3000
        trainer.train(num_episodes, eval_interval=1000)
        
        # Evaluate on validation set
        results = trainer.evaluate(val_words[:500])  # Use subset for faster evaluation
        
        score = results['final_score']
        print(f"Score: {score:.2f}")
        
        if score > best_score:
            best_score = score
            best_params = params
            best_results = results
            print("*** NEW BEST! ***")
    
    print(f"\n{'='*60}")
    print("HYPERPARAMETER TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"\nBest Parameters: {best_params}")
    print(f"Best Score: {best_score:.2f}")
    print(f"Best Results: {best_results}")
    print(f"{'='*60}\n")
    
    return best_params, best_score


def main():
    """Main training script"""
    print("="*60)
    print("PART 2: REINFORCEMENT LEARNING TRAINING")
    print("="*60)
    
    # Load HMM oracle
    print("\nLoading HMM oracle...")
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.hmm_model import HMMHangmanOracle
    
    hmm_oracle = HMMHangmanOracle()
    hmm_oracle.load('saved_models/hmm_oracle.pkl')
    
    # Load corpus
    print("Loading corpus...")
    with open('data/corpus.txt', 'r') as f:
        words = [line.strip().lower() for line in f if line.strip() and line.strip().isalpha()]
    
    print(f"Loaded {len(words)} words")
    
    # Split into train/val/test
    random.shuffle(words)
    train_size = int(0.7 * len(words))
    val_size = int(0.15 * len(words))
    
    train_words = words[:train_size]
    val_words = words[train_size:train_size + val_size]
    test_words = words[train_size + val_size:]
    
    print(f"Train: {len(train_words)}, Val: {len(val_words)}, Test: {len(test_words)}")
    
    # Choose agent type
    agent_type = 'qlearning'  # or 'dqn'
    print(f"\nAgent Type: {agent_type.upper()}")
    
    # Hyperparameter tuning
    do_hyperparameter_tuning = True  # Set to False to use default params
    
    if do_hyperparameter_tuning:
        best_params, best_score = hyperparameter_search(
            hmm_oracle, train_words, val_words, agent_type=agent_type
        )
    else:
        # Default parameters
        if agent_type == 'qlearning':
            best_params = {'alpha': 0.1, 'gamma': 0.95, 'epsilon_decay': 0.997}
        else:
            best_params = {'alpha': 0.001, 'gamma': 0.95, 'epsilon_decay': 0.997, 'hidden_size': 128}
    
    # Train final model with best parameters
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*60}\n")
    print(f"Using parameters: {best_params}")
    
    if agent_type == 'qlearning':
        agent = QLearningAgent(
            alpha=best_params['alpha'],
            gamma=best_params['gamma'],
            epsilon=1.0,
            epsilon_decay=best_params['epsilon_decay'],
            epsilon_min=0.01
        )
    else:
        agent = DQNAgent(
            alpha=best_params['alpha'],
            gamma=best_params['gamma'],
            epsilon=1.0,
            epsilon_decay=best_params['epsilon_decay'],
            epsilon_min=0.01,
            hidden_size=best_params['hidden_size']
        )
    
    # Create environment and trainer
    env = HangmanEnvironment(train_words, max_lives=6)
    trainer = HangmanTrainer(hmm_oracle, agent, env, agent_type=agent_type)
    
    # Train
    num_episodes = 20000 if agent_type == 'qlearning' else 10000
    trainer.train(num_episodes, eval_interval=2000)
    
    # Save agent
    output_file = f'{agent_type}_agent.pkl'
    agent.save(output_file)
    
    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}\n")
    test_results = trainer.evaluate(test_words)
    
    # Save results
    results_data = {
        'agent_type': agent_type,
        'best_params': best_params,
        'test_results': test_results,
        'training_episodes': num_episodes
    }
    
    with open(f'{agent_type}_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {agent_type}_results.json")
    print(f"Agent saved to {output_file}")
    print(f"\n{'='*60}")
    print("ALL TRAINING COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()