__author__ = 'dot'
import numpy as np
import Utils
from numpy.random import RandomState


def policy_iteration(mdp, gamma, seed=1):
    X = mdp.X
    policy_not_stable = True
    debug = []
    random = RandomState(seed)
    policy = get_random_policy(X, mdp.U, random)
    debug.append(",".join([str(i) + "->" + str(policy[i]) for i in range(len(policy))]))
    iter_count = 0
    while policy_not_stable:
        debug.append("iter=%d"%(iter_count))
        prev_policy = policy
        # compute V
        V = policy_evaluation(mdp,policy,gamma)

        debug.append("V:")
        debug.append(Utils.show_vec(V,"\n"))

        Q = compute_Q_table(mdp,V,gamma)

        debug.append("Q:")
        debug.append(Utils.show(Q))


        policy = np.argmax(Q,axis=1)

        debug.append("policy:")
        debug.append(",".join([str(i) + "->" + str(policy[i]) for i in range(len(policy))]))

        policy_not_stable = (policy!=prev_policy).any()
        iter_count += 1
    return policy, "\n".join(debug)

def compute_Q_table(mdp, V, gamma):
    Q = np.zeros(shape=(mdp.X,mdp.U))
    for u in range(mdp.U):
        policy = np.ones(shape=(mdp.X,1),dtype="int")*u
        P = get_matrix_transition(mdp.P,policy)
        R = get_effective_reward(P,mdp.R)
        Qcolumn = R + gamma * P.dot(V)
        Q[:,u] = Qcolumn
    return Q

def policy_evaluation(mdp, mu, gamma):
    X = mdp.X
    P = get_matrix_transition(mdp.P, mu)
    R = get_effective_reward(P,mdp.R)
    A = np.eye(X) - gamma * P
    V = np.linalg.solve(A,R)
    return V

def get_matrix_transition(tensor, mu):
    shape = tensor.shape
    X = shape[1]
    P = np.zeros(shape = (X,X))

    for x in range(X):
        P[x,:] =tensor[mu[x],x,:]
    return P

def get_effective_reward(transition, reward):
    return np.sum(np.multiply(transition,reward),axis=1)

def get_random_policy(X, U, random):
    policy = -np.ones(shape=(X),dtype="int")
    for x in range(X):
        policy[x] = random.randint(0,U)
    return policy

# test
'''
transition = np.array([[[1,0],[0,1]] , [[2,0],[0,2]]])
reward = np.array([[1,2],[3,4]])
mu = np.array([0,1],dtype='int')
print(mu)

P = get_matrix_transition(transition,mu)
print("Effective transition")
print(P)
print("reward matrix")
print(reward)
r = get_effective_reward(P,reward)
print(r)
'''


