import numpy as np
import cvxpy as cp
from scipy.linalg import expm
import OptimalTargetSelection

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

# Define constants and system matrices
g = 9.81  # m/s^2
Ix = 3.3e-3 #kg m^2
Iy = 3.3e-3 #kg m^2
Iz = 5.8e-3 #kg m^2
m = 0.547 #kg

A = np.zeros((12, 12))
A[0, 6] = 1
A[1, 7] = 1
A[2, 8] = 1
A[3, 9] = 1
A[4, 10] = 1
A[5, 11] = 1

A[6, 4] = g
A[7, 3] = -g

B = np.zeros((12, 4))
B[8, 0] = 1 / m
B[9, 1] = 1 / Ix
B[10, 2] = 1 / Iy
B[11, 3] = 1 / Iz

C = np.zeros((3, 12))
for i in range(3):
    C[i, i] = 1

I = np.eye(A.shape[0])

Q = np.eye(A.shape[0]) * 0.01
Q[0, 0] = 100
Q[1, 1] = 100
Q[2, 2] = 100
R = np.eye(B.shape[1]) * 1

Ts = 0.01
Ad, Bd = discretize(A, B, Ts)



# Create an instance of the OptimalTargetSelection class
target_selector = OptimalTargetSelection.OptimalTargetSelection(Ad, Bd, C, Q, R,m,"circle")

# Example usage of the class
x0 = np.zeros((Ad.shape[0], 1))  # Initial state

target_selector.trajectory_gen()

# xr = np.load("xr_opt.npy",allow_pickle=True)
# ur = np.load("ur_opt.npy",allow_pickle=True)

