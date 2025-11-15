
"""
Models package for Hangman agents
"""

from .hmm_model import HMMHangmanOracle
#from .rl_agent import HangmanRLAgent, DuelingDQN, PrioritizedReplayBuffer


__all__ = [
    'HMMHangmanOracle',
    'HangmanRLAgent',
    'DuelingDQN',
    'DQN',  # Alias for backwards compatibility
    'PrioritizedReplayBuffer'
]