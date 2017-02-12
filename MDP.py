__author__ = 'dot'
import random
import numpy as np
from numpy.random import RandomState


class MDP:
    def __init__(self, X=4, U=2, seed=1, method="long", params={}):
        self.X = X
        self.U = U
        self.P = []
        self.R = []
        self.random = RandomState(seed)
        self.method = method
        if method == "random":
            self.init_mdp_random(params)
        elif method == "long":
            self.init_long(params)
        else:
            print("No such method!")


    def init_mdp_random(self,params = {}):
        # dealing with the transition
        for u in range(self.U):
            self.P.append(self.create_transition_matrix(self.X, self.X))
        self.P = np.array(self.P)

        # dealing with the reward
        self.R = self.random.normal(size=(self.X,self.X))

    def init_long(self, params = {}):
        '''
         should be with two actions. start in the left most
         action 0 is going right
         action 1 is going left
         s0 <-> s1 <-> s2 ...
        '''
        # process noise
        transition_noise = 0
        if params.has_key("transition_noise"):
            transition_noise = params.get("transition_noise")

        self.U = 2
        self.P = np.zeros(shape=(self.U,self.X, self.X))
        self.R = np.zeros(shape=(self.X, self.X))
        for x in range(self.X):
            if x<(self.X-1):
                self.P[0][x][x+1] = 1.0
            if x>0:
                self.P[1][x][x-1] = 1.0
            if x==self.X-1:
                self.P[0][x][x] = 1.0
                self.P[1][0][0] = 1.0
            if x==self.X-2:
                self.R[x][x+1]=1.0

    def create_transition_matrix(self, X, Y):
        P = self.random.uniform(0, 1, size=(X, Y))
        sum = np.sum(P, axis=1)
        for i in range(len(P)):
            P[i] = P[i] / sum[i]
        return P

    @staticmethod
    def show_vector(V, accuracy_digits=3):
        terms = []
        format = "%" + str(accuracy_digits) + "f"
        for i in range(len(V)):
            terms.append("%.3f" % (V[i]) )
        return ", ".join(terms)

    @staticmethod
    def show_matrix(mat, accuracy_digits = 3):
        lines = []
        for i in range(len(mat)):
            lines.append(MDP.show_vector(mat[i],accuracy_digits))
        return "\n".join(lines)

    def show(self, accuracy_digits = 3):
        lines = []
        for u in range(self.U):
            lines.append("Transition for Action num %d:" % u)
            lines.append(MDP.show_matrix(self.P[u]))
        lines.append("Reward for Action num %d:" % u)
        lines.append(MDP.show_matrix(self.R))
        return "\n".join(lines)

