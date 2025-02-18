import numpy as np
import random

# Actions: Up, Down, Left, Right
ACTIONS = ['U', 'D', 'L', 'R']
ACTION_DICT = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

class QLearning:
    def __init__(self, grid_size, goal_state, alpha=0.5, gamma=0.9, epsilon=0.1, episodes=2000):
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((grid_size, grid_size, len(ACTIONS)))

    def get_next_state(self, state, action):
        x, y = state
        dx, dy = ACTION_DICT[action]
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            return (new_x, new_y)
        return state  # Stay in place if hitting boundary

    def get_reward(self, state):
        return 10 if state == self.goal_state else -1

    def train_q_learning(self):
        """ Train agent using Q-learning """
        for episode in range(self.episodes):
            state = (0, 0)  # Start at top-left corner

            while state != self.goal_state:
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(ACTIONS)  # Explore
                else:
                    action = ACTIONS[np.argmax(self.q_table[state[0], state[1]])]  # Exploit

                next_state = self.get_next_state(state, action)
                reward = self.get_reward(next_state)

                action_idx = ACTIONS.index(action)
                max_next_q = np.max(self.q_table[next_state[0], next_state[1]])
                self.q_table[state[0], state[1], action_idx] += self.alpha * (
                    reward + self.gamma * max_next_q - self.q_table[state[0], state[1], action_idx]
                )

                state = next_state
