# Hangman-agent-Design
Our aim is to design and build an intelligent Hangman assistant that effectively guesses letters to solve puzzles with maximum efficiency. The goal is not just to win, but to win with the fewest possible mistakes, leveraging machine learning to peer into the hidden word.

# Hangman AI: HMM + Reinforcement Learning

An intelligent Hangman agent that combines Hidden Markov Models with Deep Q-Learning to achieve optimal performance.

## 🎯 Project Overview

This project implements a hybrid AI system for playing Hangman:
- **Part 1**: Hidden Markov Model (HMM) for probabilistic letter prediction
- **Part 2**: Deep Q-Network (DQN) agent for strategic decision-making

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/hangman-ai.git
cd hangman-ai
pip install -r requirements.txt
```

### Training
```bash
# Step 1: Train HMM
python scripts/train_hmm.py

# Step 2: Train RL Agent
python scripts/train_rl.py

# Step 3: Evaluate
python scripts/evaluate.py
```

## 📊 Results

- **Success Rate**: 80%+
- **Average Wrong Guesses**: <2.5 per game
- **Final Score**: 800+

See `results/evaluation_results.txt` for detailed metrics.

## 🏗️ Architecture

### HMM Oracle
- Position-based states
- Letter emissions
- Length-specific models (2-20 chars)

### RL Agent
- Algorithm: Deep Q-Network (DQN)
- State: 73-dimensional (word + guessed + lives + HMM probs)
- Action: 26 letters
- Exploration: ε-greedy with decay

## 📁 Project Structure