import numpy as np
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from lqr_control import LQRControl
from control import dlqr    
import cvxpy as cp


def discretize(A, B, Ts):

    n = A.shape[0]
    p = B.shape[1]
    
    M = np.zeros((n + p, n + p))
    M[:n, :n] = A
    M[:n, n:] = B
    M[n:, n:] = np.zeros((p, p))
    
    M_d = expm(M * Ts)
    
    A_d = M_d[:n, :n]
    B_d = M_d[:n, n:]
    
    return A_d, B_d


def mpc(A,B,N,x0,x_ref,u_ref,Q,R):

    cost = 0.0
    constraints = []

    x = cp.Variable((12, N + 1)) # cp.Variable((dim_1, dim_2))
    u = cp.Variable((4, N))

    Q = np.array(Q)
    R = np.array(R)

    for k in range(N):
        cost += cp.quad_form(x[:, k] - x_ref, Q)
        cost += cp.quad_form(u[:, k], R)
        constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]

    constraints += [x[:, 0] == x0] 
    problem = cp.Problem(cp.Minimize(cost),constraints)
    problem.solve(solver=cp.OSQP)

    return u[:, 0].value, x[:, 1].value, x[:, :].value, None



N = 1000

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


Ad,Bd = discretize(A,B,Ts=0.1)



Q = np.eye(Ad.shape[0])
R = np.eye(Bd.shape[1])
K = dlqr(Ad, Bd, Q, R)[0]

#print(Ad)
#print(Bd)

x0 = np.zeros(12)

x_ref = np.array([5.0,5.0,5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
u_ref = np.zeros(4)

u_0, x_1, x_traj, _ = mpc(Ad, Bd, N, x0, x_ref, u_ref,Q,R)

x_pos = x_traj[0, :]  
y_pos = x_traj[1, :]  
z_pos = x_traj[2, :]  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_pos, y_pos, z_pos, label='Trajectory', color='b')

ax.set_xlabel('x (Position)')
ax.set_ylabel('y (Position)')
ax.set_zlabel('z (Position)')

ax.set_title('3D Position Trajectory')


plt.show()

