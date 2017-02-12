__author__ = 'dot'
import random
import numpy as np


class QAgent():
    def __init__(self, X, U, initial_value=0, gamma=0.99, alpha=0.01, epsilon=0.01, seed=1):
        self.X = X
        self.U = U
        self.gamma = gamma
        self.epsilon = epsilon
        self.random = random.Random()
        self.random.seed(seed)
        self.Q = np.ones((X, U), dtype=np.float32) * initial_value
        self.cur_state = 0
        self.alpha = alpha

    def get_action(self, state):
        tmp_actions = np.where(self.Q[state] == self.Q[state].max())[0]
        idx = self.random.randint(0, len(tmp_actions) - 1)
        return tmp_actions[idx]

    def get_action_epsilon_greedy(self, state):
        tmp_actions = []
        if random.uniform(0, 1) < self.epsilon:
            tmp_actions = range(self.U)
        else:
            tmp_actions = np.where(self.Q[state] == self.Q[state].max())[0]
        idx = self.random.randint(0, len(tmp_actions) - 1)
        return tmp_actions[idx]

    def update(self, cur_action, next_state, next_reward):
        if self.cur_state == -1:
            self.cur_state = next_state
            return

        next_action = self.get_action(next_state)
        self.Q[self.cur_state][cur_action] = self.Q[self.cur_state][cur_action] + self.alpha * \
                                                                              (next_reward + self.gamma * \
                                                                               self.Q[next_state][next_action] - \
                                                                               self.Q[self.cur_state][cur_action])
        self.cur_state = next_state
        return

    def show_q(self):
        sb = []
        for x in range(self.X):
            sb1 = ["%3d:" % x]
            for u in range(self.U):
                sb1.append("%.3f" % self.Q[x][u])
            line = ",".join(sb1)
            sb.append(line)
        return "\n".join(sb)