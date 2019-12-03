import numpy as np
#import pandas as pd 
import seaborn as sb 
import matplotlib.pyplot as plt


wall = -40
back = 0
Q = np.zeros((4,5))
#Q = np.random.randint(size=(4,5),low=0, high=200)
print(Q)
U = np.array([[0,1,1,2,4],[0,2,3,3,4],[1,1,2,3,3],[0,0,2,4,4]])
B = np.array([[wall,wall,back,back,wall],[wall,back,back,wall,wall],[back,wall,wall,wall,back],[wall,back,wall,421.1149,wall]])
N = np.zeros((4,4))

#Q = np.zeros((4,6))
#U = np.array([[0,1,1,2,5,0],[5,2,3,3,4,4],[1,1,2,3,3,2],[0,0,5,4,4,5]])
#B = np.array([[wall,wall,back,back,back,back],[back,back,back,wall,wall,400],[back,wall,wall,wall,back,back],[wall,back,back,400,wall,wall]])
#N = np.zeros((4,6))
gamma = 0.98



def learn():
    z = 0
    n = 0
    while z != 4:
        a = np.random.randint(4)
        #a2 = np.argmax([Q[i][z] for i in range(0,4)])
        #a = np.random.choice([a1,a2])
        zz = U[a][z]
        r = B[a][z]
        N[a][z]=N[a][z]+1
        qMax = np.max([Q[i][zz] for i in range(0,4)])
        alpha = 1/(1+N[a][z])
        Q[a][z] = (1-alpha)*Q[a][z] + alpha*(r+(gamma**n)*qMax)
        z = zz
        n = n+1


for i in range(50000):
    learn()

print(Q)
sb.heatmap(Q)
plt.show()
print("fertig")


