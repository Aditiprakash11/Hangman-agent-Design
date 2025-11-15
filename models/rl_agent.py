import numpy as np
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json

class HangmanState:
    """Represents the state of a Hangman game"""
    
    def __init__(self, masked_word: str, guessed_letters: set, lives_left: int, 
                 hmm_probs: Dict[str, float]):
        self.masked_word = masked_word
        self.guessed_letters = frozenset(guessed_letters)
        self.lives_left = lives_left
        self.hmm_probs = hmm_probs
        
    def to_tuple(self) -> tuple:
        """Convert state to hashable tuple for Q-table"""
        return (self.masked_word, self.guessed_letters, self.lives_left)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert state to feature vector for DQN"""
        features = []
        
        # 1. Word length feature (normalized)
        features.append(len(self.masked_word) / 20.0)
        
        # 2. Progress feature (ratio of revealed letters)
        revealed = sum(1 for c in self.masked_word if c != '_')
        features.append(revealed / len(self.masked_word) if len(self.masked_word) > 0 else 0)
        
        # 3. Lives left (normalized)
        features.append(self.lives_left / 6.0)
        
        # 4. Number of guessed letters (normalized)
        features.append(len(self.guessed_letters) / 26.0)
        
        # 5. HMM probability features (top 10 letter probabilities)
        sorted_probs = sorted(self.hmm_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        for _, prob in sorted_probs:
            features.append(prob)
        # Pad if less than 10
        while len(features) < 14:
            features.append(0.0)
        
        # 6. Position pattern features (simplified)
        # Count blanks at start, middle, end
        blanks_start = sum(1 for c in self.masked_word[:len(self.masked_word)//3] if c == '_')
        blanks_middle = sum(1 for c in self.masked_word[len(self.masked_word)//3:2*len(self.masked_word)//3] if c == '_')
        blanks_end = sum(1 for c in self.masked_word[2*len(self.masked_word)//3:] if c == '_')
        total_blanks = sum(1 for c in self.masked_word if c == '_')
        
        if total_blanks > 0:
            features.append(blanks_start / total_blanks)
            features.append(blanks_middle / total_blanks)
            features.append(blanks_end / total_blanks)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)


class QLearningAgent:
    """Q-Learning agent for Hangman"""
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.q_table: Dict[tuple, Dict[str, float]] = defaultdict(lambda: {l: 0.0 for l in self.alphabet})
        
        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
    def get_action(self, state: HangmanState, available_letters: set) -> str:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Explore: weighted random based on HMM probabilities
            probs = [state.hmm_probs.get(l, 0.0) for l in available_letters]
            total = sum(probs)
            if total > 0:
                probs = [p/total for p in probs]
                return np.random.choice(list(available_letters), p=probs)
            else:
                return random.choice(list(available_letters))
        else:
            # Exploit: choose best action based on Q-values and HMM
            state_key = state.to_tuple()
            q_values = self.q_table[state_key]
            
            # Combine Q-values with HMM probabilities
            combined_scores = {}
            for letter in available_letters:
                q_val = q_values.get(letter, 0.0)
                hmm_prob = state.hmm_probs.get(letter, 0.0)
                # Weighted combination
                combined_scores[letter] = q_val + 0.3 * hmm_prob
            
            return max(combined_scores.items(), key=lambda x: x[1])[0]
    
    def update(self, state: HangmanState, action: str, reward: float, 
               next_state: Optional[HangmanState], done: bool):
        """Update Q-value using Q-learning update rule"""
        state_key = state.to_tuple()
        
        if done or next_state is None:
            target = reward
        else:
            next_state_key = next_state.to_tuple()
            # Get available letters for next state
            available_next = [l for l in self.alphabet if l not in next_state.guessed_letters]
            if available_next:
                max_next_q = max(self.q_table[next_state_key][l] for l in available_next)
            else:
                max_next_q = 0.0
            target = reward + self.gamma * max_next_q
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        self.q_table[state_key][action] = current_q + self.alpha * (target - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save Q-table and parameters"""
        data = {
            'q_table': dict(self.q_table),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved Q-Learning agent to {filepath}")
    
    def load(self, filepath: str):
        """Load Q-table and parameters"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: {l: 0.0 for l in self.alphabet}, data['q_table'])
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.epsilon_decay = data['epsilon_decay']
        self.epsilon_min = data['epsilon_min']
        print(f"Loaded Q-Learning agent from {filepath}")


class DQNAgent:
    """Deep Q-Network agent for Hangman (simplified version)"""
    
    def __init__(self, state_size: int = 17, action_size: int = 26,
                 alpha: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, hidden_size: int = 128):
        self.state_size = state_size
        self.action_size = action_size
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_to_idx = {letter: idx for idx, letter in enumerate(self.alphabet)}
        
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural network weights (simple 2-layer network)
        self.hidden_size = hidden_size
        self.W1 = np.random.randn(state_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, action_size) * 0.01
        self.b2 = np.zeros((1, action_size))
        
        # Experience replay buffer
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 32
    
    def _forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # Hidden layer with ReLU
        z1 = np.dot(state, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        
        # Output layer
        q_values = np.dot(a1, self.W2) + self.b2
        
        return q_values.flatten(), a1
    
    def get_action(self, state: HangmanState, available_letters: set) -> str:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Explore: weighted random based on HMM
            probs = [state.hmm_probs.get(l, 0.0) for l in available_letters]
            total = sum(probs)
            if total > 0:
                probs = [p/total for p in probs]
                return np.random.choice(list(available_letters), p=probs)
            else:
                return random.choice(list(available_letters))
        else:
            # Exploit: choose best action
            state_vector = state.to_feature_vector()
            q_values, _ = self._forward(state_vector)
            
            # Mask unavailable actions
            for letter in self.alphabet:
                if letter not in available_letters:
                    q_values[self.letter_to_idx[letter]] = -np.inf
            
            # Add HMM bonus
            for letter in available_letters:
                idx = self.letter_to_idx[letter]
                q_values[idx] += 0.3 * state.hmm_probs.get(letter, 0.0)
            
            best_idx = np.argmax(q_values)
            return self.alphabet[best_idx]
    
    def remember(self, state: HangmanState, action: str, reward: float,
                 next_state: Optional[HangmanState], done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def replay(self):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_vector = state.to_feature_vector()
            action_idx = self.letter_to_idx[action]
            
            # Compute target
            if done or next_state is None:
                target = reward
            else:
                next_vector = next_state.to_feature_vector()
                next_q_values, _ = self._forward(next_vector)
                
                # Mask unavailable actions
                available_next = [l for l in self.alphabet if l not in next_state.guessed_letters]
                for letter in self.alphabet:
                    if letter not in available_next:
                        next_q_values[self.letter_to_idx[letter]] = -np.inf
                
                max_next_q = np.max(next_q_values) if len(available_next) > 0 else 0.0
                target = reward + self.gamma * max_next_q
            
            # Forward pass
            q_values, hidden = self._forward(state_vector)
            
            # Compute loss gradient
            td_error = target - q_values[action_idx]
            
            # Backpropagation (simplified)
            dq = np.zeros_like(q_values)
            dq[action_idx] = -td_error
            dq = dq.reshape(1, -1)
            
            # Update output layer
            dW2 = np.dot(hidden.T, dq)
            db2 = dq
            
            # Update hidden layer
            dhidden = np.dot(dq, self.W2.T)
            dhidden[hidden <= 0] = 0  # ReLU gradient
            
            state_vector_reshaped = state_vector.reshape(1, -1)
            dW1 = np.dot(state_vector_reshaped.T, dhidden)
            db1 = dhidden
            
            # Gradient descent
            self.W2 -= self.alpha * dW2
            self.b2 -= self.alpha * db2
            self.W1 -= self.alpha * dW1
            self.b1 -= self.alpha * db1
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save model"""
        data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'hidden_size': self.hidden_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved DQN agent to {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.epsilon_decay = data['epsilon_decay']
        self.epsilon_min = data['epsilon_min']
        self.hidden_size = data['hidden_size']
        print(f"Loaded DQN agent from {filepath}")