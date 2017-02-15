__author__ = 'dot'
import random
from collections import deque
import numpy as np

class Simulator:
    def __init__(self, mdp, initial_state, terminal_state = -1, seed = 1, max_deque_size = 1000):
        self.mdp = mdp
        self.initial_state = initial_state
        self.current_state = self.initial_state
        self.random = np.random.RandomState()
        self.terminal_state = terminal_state
        self.random.seed(seed)
        self.trace = deque(maxlen=max_deque_size)

    def reset(self):
        self.current_state = self.initial_state
        return self.current_state

    def get_transition(self, x, u):
        return self.mdp.P[u][x]

    def step(self, u):
        vec = self.get_transition(self.current_state, u)
        vec = np.squeeze(vec)
        next_state = self.choose_random(vec)
        done = False
        if next_state == self.terminal_state:
            done = True
        self.current_state = next_state
        self.trace.append(next_state)
        return next_state

    def choose_random(self, vec):
        return self.random.choice(range(self.mdp.X), 1, p=vec)


