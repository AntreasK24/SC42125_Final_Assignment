import numpy as np
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def discretize(A, B, Ts):
    A_d = expm(A * Ts)
    B_d = np.linalg.pinv(A) @ (A_d - np.eye(A_d.shape[0])) @ B
    return A_d, B_d


g = 9.81 #m/s^2
Ix = 1
Iy = 1
Iz = 1

m = 1

A = np.zeros((12,12))
A[0,6] = 1
A[1,7] = 1
A[2,8] = 1
A[3,9] = 1
A[4,10] = 1
A[5,11] = 1

A[6,4] = g
A[7,3] = -g


B = np.zeros((12,4))
B[8,0] = 1/m
B[9,1] = 1/Ix
B[10,2] = 1/Iy
B[11,3] = 1/Iz

print(A)

Ad,Bd = discretize(A,B,Ts=0.1)

#print(Ad)
#print(Bd)

x0 = np.zeros(12).T
x = np.zeros((100,12))

x[0] = x0

t = np.linspace(0,1,99)

u = np.array([m*g,0,0,0])

for i in range(t.shape[0]):
    x[i+1] = Ad @ x[i] + Bd @ u
    
    


x_pos = x[:, 0]  
y_pos = x[:, 1]  
z_pos = x[:, 2]  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_pos, y_pos, z_pos, label='Trajectory', color='b')

ax.set_xlabel('x (Position)')
ax.set_ylabel('y (Position)')
ax.set_zlabel('z (Position)')

ax.set_title('3D Position Trajectory')


plt.show()