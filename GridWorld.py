__author__ = 'dot'
import numpy as np


# mdp class
class GridWorld:
    def __init__(self, size=3, max_time=0):
        self.size = size
        self.x = 0
        self.y = 0
        if max_time != 0:
            self.max_time = max_time
        else:
            self.max_time = size*4
        self.time_counter = 0

    def __translate2state_int(self, x, y):
        return y*self.size + x

    def get_number_of_states(self):
        return self.size*self.size

    @staticmethod
    def get_number_of_actions():
        return 4

    def reset(self):
        self.x = 0
        self.y = 0
        state_int = self.__translate2state_int(self.x,self.y)
        reward = 0
        done = 0
        info = {'x': self.x, 'y': self.y}
        return state_int, reward, done, info

    '''
     0->left
     1->down
     2->right
     3->up
     '''
    def step(self, action):
        reward = 0
        done = 0

        # to minus x
        if action == 0 and self.x >= 1:
            self.x -= 1
        # to plus y
        elif action == 1 and self.y <= self.size-2:
            self.y += 1
        # to plus x
        elif action == 2 and self.x <= self.size-2:
            self.x += 1
        # to minus y
        elif action == 3 and self.y >= 1:
            self.y -= 1

        if self.x == self.size-1 and self.y == self.size-1:
            reward = 1.0
            done = 1

        if self.time_counter == self.max_time:
            done = 1

        state_int = self.__translate2state_int(self.x, self.y)
        info = {'x': self.x, 'y': self.y}

        return state_int, reward, done, info

    @staticmethod
    def get_random_action(this):
        return np.random.randint(low=0, high=4)

    def get_state(self):
        state_int = self.__translate2state_int(self.x, self.y)
        return state_int
