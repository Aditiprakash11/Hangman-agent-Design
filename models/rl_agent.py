import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    """Deep Q-Network for Hangman agent"""
    
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class HangmanRLAgent:
    """Reinforcement Learning agent for Hangman using DQN"""
    
    def __init__(self, hmm_oracle, max_word_length: int = 20):
        self.hmm_oracle = hmm_oracle
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.action_size = 26
        self.max_word_length = max_word_length
        
        # State size: masked_word_encoding + guessed_letters + lives + hmm_probs
        self.state_size = max_word_length + 26 + 1 + 26
        
        # DQN parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def encode_state(self, game_state: Dict) -> np.ndarray:
        """Convert game state to feature vector"""
        masked_word = game_state['masked_word']
        guessed_letters = game_state['guessed_letters']
        lives = game_state['lives_remaining']
        
        # Encode masked word (one-hot like, with special encoding for blanks)
        word_encoding = np.zeros(self.max_word_length)
        for i, char in enumerate(masked_word[:self.max_word_length]):
            if char == '_':
                word_encoding[i] = -1
            else:
                word_encoding[i] = self.alphabet.index(char) / 26.0
        
        # Encode guessed letters (binary vector)
        guessed_encoding = np.array([1 if letter in guessed_letters else 0 
                                     for letter in self.alphabet])
        
        # Lives remaining (normalized)
        lives_encoding = np.array([lives / 6.0])
        
        # Get HMM probabilities
        hmm_probs = self.hmm_oracle.get_letter_probabilities(masked_word, guessed_letters)
        hmm_encoding = np.array([hmm_probs[letter] for letter in self.alphabet])
        
        # Concatenate all features
        state_vector = np.concatenate([word_encoding, guessed_encoding, lives_encoding, hmm_encoding])
        
        return state_vector
    
    def act(self, game_state: Dict, training: bool = True) -> str:
        """Choose an action using epsilon-greedy policy"""
        guessed_letters = game_state['guessed_letters']
        available_actions = [i for i, letter in enumerate(self.alphabet) 
                           if letter not in guessed_letters]
        
        if not available_actions:
            return random.choice(list(self.alphabet))
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            action_idx = random.choice(available_actions)
        else:
            state_vector = self.encode_state(game_state)
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Mask out guessed letters
            for i in range(26):
                if i not in available_actions:
                    q_values[i] = -float('inf')
            
            action_idx = np.argmax(q_values)
        
        return self.alphabet[action_idx]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        action_idx = self.alphabet.index(action)
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self):
        """Train on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([self.encode_state(s) for s, _, _, _, _ in batch]).to(self.device)
        actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(self.device)
        next_states = torch.FloatTensor([self.encode_state(ns) for _, _, _, ns, _ in batch]).to(self.device)
        dones = torch.FloatTensor([d for _, _, _, _, d in batch]).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self, env, episodes: int = 2000):
        """
        Train the agent
        
        Args:
            env: HangmanEnvironment instance
            episodes: Number of training episodes
        
        Returns:
            Tuple of (rewards_history, success_history, loss_history)
        """
        rewards_history = []
        success_history = []
        loss_history = []
        
        print("\n" + "="*60)
        print("TRAINING RL AGENT")
        print("="*60)
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            
            while not state['game_over']:
                action = self.act(state, training=True)
                next_state, reward, done = env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                
                # Train on experiences
                loss = self.replay()
                if loss is not None:
                    loss_history.append(loss)
            
            rewards_history.append(episode_reward)
            success_history.append(1 if state['won'] else 0)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Update target network periodically
            if episode % 10 == 0:
                self.update_target_network()
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                success_rate = np.mean(success_history[-100:])
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Success Rate: {success_rate:.2%} | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        print("\nTraining complete!")
        return rewards_history, success_history, loss_history
    
    def evaluate(self, env, num_games: int = 2000):
        """
        Evaluate the agent
        
        Args:
            env: HangmanEnvironment instance
            num_games: Number of evaluation games
        
        Returns:
            Dictionary with evaluation metrics
        """
        wins = 0
        total_wrong_guesses = 0
        total_repeated_guesses = 0
        
        print("\n" + "="*60)
        print(f"EVALUATING AGENT ON {num_games} GAMES")
        print("="*60)
        
        for game in range(num_games):
            state = env.reset()
            
            while not state['game_over']:
                action = self.act(state, training=False)
                next_state, reward, done = env.step(action)
                state = next_state
            
            if state['won']:
                wins += 1
            total_wrong_guesses += env.wrong_guesses
            total_repeated_guesses += env.repeated_guesses
            
            if (game + 1) % 500 == 0:
                print(f"Progress: {game + 1}/{num_games} games")
        
        success_rate = wins / num_games
        avg_wrong = total_wrong_guesses / num_games
        avg_repeated = total_repeated_guesses / num_games
        
        # Calculate final score
        final_score = (success_rate * num_games) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Success Rate: {success_rate:.2%} ({wins}/{num_games})")
        print(f"Total Wrong Guesses: {total_wrong_guesses}")
        print(f"Total Repeated Guesses: {total_repeated_guesses}")
        print(f"Avg Wrong Guesses per Game: {avg_wrong:.2f}")
        print(f"Avg Repeated Guesses per Game: {avg_repeated:.2f}")
        print(f"\nFINAL SCORE: {final_score:.2f}")
        print("="*60)
        
        return {
            'success_rate': success_rate,
            'wins': wins,
            'total_wrong_guesses': total_wrong_guesses,
            'total_repeated_guesses': total_repeated_guesses,
            'avg_wrong_guesses': avg_wrong,
            'avg_repeated_guesses': avg_repeated,
            'final_score': final_score
        }
    
    def save(self, filepath: str):
        """Save the trained agent"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Saved agent to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"Loaded agent from {filepath}")