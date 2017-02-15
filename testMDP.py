__author__ = 'dot'
import numpy as np

from MDP import MDP
import Planning as plan
from Simulator import Simulator
import random

mdp = MDP(2,params={"prob_out":0.1 })
print(np.sum(mdp.P,axis=2))
print(mdp.show(accuracy_digits = 0))
policy,debug = plan.policy_iteration(mdp, 0.9)
print(debug)
simulator = Simulator(mdp,0)
print("start state = %d" % (simulator.current_state))
for i in range(10):
    action = random.randint(0,1)
    print("action=%d" % (action))
    next_state = simulator.step(action)
    print("state=%d" % (next_state))