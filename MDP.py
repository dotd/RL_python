__author__ = 'dot'
import random
import numpy as np
from numpy.random import RandomState
import sys


class MDP:
    def __init__(self, X=4, U=2, seed=1, method="symmetric", params={}, dim = ()):
        self.X = X
        self.U = U
        self.UDescription = []
        self.P = []
        self.R = []
        self.random = RandomState(seed)
        self.method = method
        if method == "random":
            self.init_mdp_random(params)
        elif method == "long":
            self.init_long(params)
        elif method == "grid":
            self.init_grid_world(params)
        elif method == "symmetric":
            self.init_symmetric(params)
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
        noise = 0.1
        if params.has_key("transition_noise"):
            noise = params.get("transition_noise")

        self.U = 2
        self.P = np.zeros(shape=(self.U,self.X, self.X))

        self.R = np.zeros(shape=(self.X, self.X))
        for x in range(self.X):
            if x==0:
                self.P[0][0][0:1] += np.array([noise/2, (1.0-noise)])
                self.P[1][x][x-1] += [(1.0-noise), noise/2]
            if x<(self.X-1) and x>0:
                self.P[0][x][x-1:x+1] += [noise/2, (1.0-noise), noise/2]
            if x==(self.X-1):
                self.P[0][x][0] = 1.0
                self.P[1][x][0] = 1.0

    @staticmethod
    def join_params(params, params_default):
        for key in params_default:
            if not params.has_key(key):
                params.update({key:params_default[key]})
        return params

    def init_grid_world(self, params):
        '''
        ----> x
        |
        |
        v

        y


        0 Left -x
        1 Down +y
        2 Right +x
        3 up -y

        '''
        params_default = {"noise":0.1}
        params = MDP.join_params(params, params_default)
        noise = params["noise"]
        self.UDescription = ["left","down","right","up"]
        dim = self.X
        self.X = dim*dim
        self.U = 4
        self.P = np.zeros(shape=(self.U,self.X, self.X))
        grid_indices = np.arange(self.X).reshape((dim, dim))
        grid_obstacles = np.zeros(shape=(dim, dim))

        # Build maps
        map_idx_2_two_dim = {}
        two_dim_2_map_idx = {}
        for x in range(dim):
            for y in range(dim):
                map_idx_2_two_dim.update({grid_indices[y][x]:(x,y)})
                two_dim_2_map_idx.update({(x,y):grid_indices[y][x]})

        for x in range(dim):
            for y in range(dim):
                cur_state = two_dim_2_map_idx[(x, y)]
                state_left = two_dim_2_map_idx[(x-1 if x > 0 else x, y)]
                state_down = two_dim_2_map_idx[(x, y+1 if y < dim-1 else y)]
                state_right = two_dim_2_map_idx[(x+1 if x < dim-1 else x, y)]
                state_up = two_dim_2_map_idx[(x, y-1 if y > 0 else y)]
                for action in range(4):
                    self.P[action][cur_state][cur_state] += noise/4
                    if action == 0:
                        self.P[action][cur_state][state_left] += 1-noise
                        self.P[action][cur_state][state_down] += noise/4
                        self.P[action][cur_state][state_right] += noise/4
                        self.P[action][cur_state][state_up] += noise/4
                    if action == 1:
                        self.P[action][cur_state][state_left] += noise/4
                        self.P[action][cur_state][state_down] += 1-noise
                        self.P[action][cur_state][state_right] += noise/4
                        self.P[action][cur_state][state_up] += noise/4
                    if action == 2:
                        self.P[action][cur_state][state_left] += noise/4
                        self.P[action][cur_state][state_down] += noise/4
                        self.P[action][cur_state][state_right] += 1-noise
                        self.P[action][cur_state][state_up] += noise/4
                    if action == 3:
                        self.P[action][cur_state][state_left] += noise/4
                        self.P[action][cur_state][state_down] += noise/4
                        self.P[action][cur_state][state_right] += noise/4
                        self.P[action][cur_state][state_up] += 1-noise

        self.R = np.zeros(shape=(self.X,self.X))
        self.R[:,self.X-1]=1.0

    def init_symmetric(self, params):
        params_default = {"prob_out":0.1}
        params = MDP.join_params(params, params_default)
        self.X = 2
        self.U = 2
        prob = params["prob_out"]
        self.P = np.array([[[1-prob, prob],[ 1-prob, prob]],[[prob, 1-prob], [prob, 1-prob]]])
        self.R = np.array([[0, 1], [0, 1]])

    def check_mdp_validity(self, tolerance = sys.float_info.epsilon):
        res = np.abs(np.sum(self.P, axis=2)-1)
        for x in self.X:
            for u in self.U:
                if res[u][x] >= tolerance:
                    return False, (u, x)
        return True, (0, 0)


    '''
    def createObstacles(self,shape,prob):
        gridIndices = np.arange(self.X).reshape()
    '''


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
            lines.append(MDP.show_matrix(self.P[u],accuracy_digits=accuracy_digits ))
        lines.append("Reward for Action num %d:" % u)
        lines.append(MDP.show_matrix(self.R))
        return "\n".join(lines)

