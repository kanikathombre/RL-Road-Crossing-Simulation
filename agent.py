import numpy as np

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99):
        # Initialize the Q-table as an empty dictionary
        self.q_table = {}
        # Define total number of possible actions (e.g., up, down, left, right = 4)
        self.action_space = action_space
        # Learning rate (alpha): how much new info overrides old info
        self.alpha = learning_rate
        # Discount factor (gamma): importance of future rewards vs. immediate ones
        self.gamma = discount_factor
        # Exploration rate (epsilon): how often to explore random actions
        self.epsilon = epsilon
        # How much to reduce epsilon each episode (encourages exploration early, exploitation later)
        self.epsilon_decay = epsilon_decay


    def get_state_key(self, state):
        # Converts the state into a tuple 
        return tuple(state.flatten()) if isinstance(state, np.ndarray) else tuple(state)

    def choose_action(self, state):
        # Get a key version of the state
        state_key = self.get_state_key(state)
        # If this state hasn't been seen before, initialize with zero values
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)
        # With probability epsilon, explore by picking a random action
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        # Otherwise, exploit: pick the best-known action
        return np.argmax(self.q_table[state_key])

    def update_q_table(self, state, action, reward, next_state):
        # Convert states to keys
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Initialize Q-values for unseen states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space)

        # Get current Q-value
        old_value = self.q_table[state_key][action]

        # Estimate future max Q-value
        next_max = np.max(self.q_table[next_state_key])

        # Q-learning update formula
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        self.q_table[state_key][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

