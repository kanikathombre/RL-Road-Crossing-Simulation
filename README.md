# Road Crossing Simulation using Q-Learning

This project simulates a pedestrian learning to cross a busy road safely using **Q-learning**, a reinforcement learning algorithm. The environment includes moving vehicles and traffic lights on a 6x6 grid. The agent learns through trial and error to reach the goal without collisions.

## Features
- Q-learning agent (no neural networks)
- Grid-based environment with traffic logic
- Dynamic traffic lights and vehicle behavior
- Streamlit-based interactive simulation
- Emoji-based visualization (🚶, 🚗)

## How It Works
- Agent starts at the bottom-left and must reach the top row.
- Rewards:
  - +10 for safe crossing
  - -1 for collision
  -  0 otherwise
- Uses epsilon-greedy strategy for exploration and learning.
- Updates Q-values with the Bellman equation.

## Simulation UI
![image](https://github.com/user-attachments/assets/db8eb5f1-19f6-4b4f-970f-2f0dc74dbe68)
