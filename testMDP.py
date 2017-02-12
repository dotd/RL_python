__author__ = 'dot'

from MDP import MDP
import Planning as plan

mdp = MDP(X = 10)
print(mdp.show())
policy,debug = plan.policy_iteration(mdp, 0.9)
print(debug)