import numpy as np

# Actions: Up, Down, Left, Right
ACTIONS = ['U', 'D', 'L', 'R']
ACTION_DICT = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

class ValueIteration:
    def __init__(self, grid_size, goal_state, gamma=0.9, theta=1e-4):
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.gamma = gamma
        self.theta = theta
        self.values = np.zeros((grid_size, grid_size))
        self.policy = np.full((grid_size, grid_size), ' ', dtype=str)

    def get_next_state(self, state, action):
        """ Get next state given an action """
        x, y = state
        dx, dy = ACTION_DICT[action]
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            return (new_x, new_y)
        return state  # Stay in place if hitting boundary

    def get_reward(self, state):
        """ Reward function """
        return 10 if state == self.goal_state else -1

    def run_value_iteration(self):
        """ Perform Value Iteration algorithm """
        while True:
            delta = 0
            new_values = np.copy(self.values)

            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    state = (x, y)
                    if state == self.goal_state:
                        continue

                    best_value = float('-inf')
                    best_action = None

                    for action in ACTIONS:
                        next_state = self.get_next_state(state, action)
                        reward = self.get_reward(next_state)
                        value = reward + self.gamma * self.values[next_state]
                        
                        if value > best_value:
                            best_value = value
                            best_action = action

                    new_values[x, y] = best_value
                    self.policy[x, y] = best_action if best_action else ' '

                    delta = max(delta, abs(new_values[x, y] - self.values[x, y]))

            self.values = new_values
            if delta < self.theta:
                break
