__author__ = 'dot'

from GridWorld import GridWorld
from QAgent import QAgent
import numpy as np
import matplotlib.pyplot as plt

##
env = GridWorld(size=10)
q_agent = QAgent(env.get_number_of_states(), GridWorld.get_number_of_actions())
res = []
for idx_epoch in range(400):
    s, r, d, info = env.reset()
    print("Reset:st=%d,r=%f,d=%d,%s"%(s,r,d,str(info)))
    for t in range(100):
        # env.render()
        curAction = q_agent.get_action_epsilon_greedy(env.get_state())
        # print("State=%d,Action=%d"%(env.getState(),curAction))
        nxtSt, nxtR, done, info = env.step(curAction)
        # print("nxtSt=%d,nxtR=%f,d=%d,info=%s"%(nxtSt,nxtR,done,str(info)))
        q_agent.update(curAction, nxtSt, nxtR)
        if done:
            print("Episode %d finished after %d time steps" %(idx_epoch,t+1))
            #print(q_agent.show_q())
            print("=============")
            res.append(t+1)
            break

print("p1")

#plt.plot([1,2,3,4])
#plt.ylabel('some numbers')
#plt.show()
#print(res)

plt.plot(res)
plt.ylabel('some numbers')
plt.show()
print("p3")
