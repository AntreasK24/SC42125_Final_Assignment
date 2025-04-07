import numpy as np
import cvxpy as cp
from scipy.signal import StateSpace
from scipy.linalg import eig, svd, solve_discrete_are
import matplotlib.pyplot as plt


### Functions
# Hautus test
def hautus_test(A, B):
    eigvals = eig(A)[0]  # Get eigenvalues only
    for lambda_ in eigvals:
        test_matrix = np.hstack([(lambda_ * np.eye(A.shape[0]) - A), B])
        rk = rank(test_matrix)
        print(f"Eigenvalue: {lambda_:.4f} \t -> rank: {rk}")

def rank(matrix, tol=1e-10):
    """Compute the rank of a matrix using SVD."""
    s = svd(matrix, compute_uv=False)
    return np.sum(s > tol)

def ctrb(A, B):
    """Construct the controllability matrix."""
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])


def predmodgen(LTI, dim):
    """
    Prediction matrices generation.
    Computes the prediction matrices P, S, and W for the optimization problem.
    """

    # Set output matrix and number of outputs
    LTI['C'] = np.eye(dim['nx'])
    dim['ny'] = dim['nx']

    # Prediction matrix from initial state
    P = np.zeros((dim['ny'] * dim['N'], dim['nx']))
    for k in range(dim['N']):
        P[k * dim['ny'] : (k + 1) * dim['ny'], :] = LTI['C'] @ np.linalg.matrix_power(LTI['A'], k)

    # Prediction matrix from inputs
    S = np.zeros((dim['ny'] * dim['N'], dim['nu'] * dim['N']))
    for k in range(1, dim['N']):
        for i in range(k):
            row_start = k * dim['ny']
            col_start = i * dim['nu']
            S[row_start : row_start + dim['ny'], col_start : col_start + dim['nu']] = \
                LTI['C'] @ np.linalg.matrix_power(LTI['A'], k - 1 - i) @ LTI['B']

    # Matrix W
    W = np.zeros((dim['ny'], dim['nu'] * dim['N']))
    for i in range(dim['N']):
        col_start = i * dim['nu']
        W[:, col_start : col_start + dim['nu']] = \
            LTI['C'] @ np.linalg.matrix_power(LTI['A'], dim['N'] - i - 1) @ LTI['B']

    return P, S, W


# Parameters 
# TODO: Change to ours
g = 9.81     # m/s^2
m = 0.5        # kg
L = 0.565      # meters (Length of pendulum to center of mass)
l = 0.17       # meters (Quadrotor center to rotor center)
Iy = 3.2e-3  # kg m^2 (Quadrotor inertia around y-axis)
Ix = Iy;    # kg m^2 (Quadrotor inertia around x-axis)
Iz = 5.5e-3  # kg m^2 (Quadrotor inertia around z-axis)

Ts = 0.1  # seconds (Sampling time)
T_sim = 15 # seconds (Simulation time)
T = int(T_sim/Ts) # Number of time steps

# System dynamics
dim_x = 12
dim_u = 4
dim_y = 6
dim_d = 2

lower_bounds = [-1000,-1000,0.0,-0.17,-0.17,-2*np.pi,-5,-5,-3,-2,-3,-3]
upper_bounds = [1000,1000,1000,0.17,0.17,2*np.pi,5,5,3,2,3,3]

# ------------------------------

# State vector
# [𝑥 𝑦 𝑧] Linear position in Earth reference frame
# [𝜙 𝜃 𝜓] Angular position in Earth reference frame
# [x_dot y_dot z_dot] Linear velocity in Earth reference frame
# [𝑝 𝑞 𝑟] Angular velocity in body reference frame


Ac = np.zeros((dim_x,dim_x))
Ac[0,6] = 1
Ac[1,7] = 1
Ac[2,8] = 1
Ac[3,9] = 1
Ac[4,10] = 1
Ac[5,11] = 1

Ac[6,4] = g
Ac[7,3] = -g

print(eig(Ac)[1])

Bc = np.zeros((dim_x,dim_u))
Bc[8,0] = 1/m
Bc[9,1] = 1/Ix
Bc[10,2] = 1/Iy
Bc[11,3] = 1/Iz


Cc = np.ones((dim_x,dim_x))
Dc = np.zeros((dim_x,dim_u))


sysc = StateSpace(Ac, Bc, Cc, Dc)  # creates a continuous-time system



# Check controllability
Ctrb_matrix = ctrb(Ac, Bc)
Ctrb_rank = rank(Ctrb_matrix)

print('Number of states:', Ac.shape[0])
print('Rank of controllability matrix:', Ctrb_rank)

hautus_test(Ac, Bc)

# Discretize the system
sysd = sysc.to_discrete(Ts, method='zoh')  # discrete-time system
A = sysd.A
B = sysd.B
C_plot = sysd.C # extract all states

# Output vector
# [𝑥 𝑦 𝑧] Linear position in Earth reference frame
# [𝑝 𝑞 𝜓] Roll (gamma) and pitch (beta) rates, yaw angle

C = np.zeros((dim_y,dim_x))

C[0,0] = 1 # x
C[1,1] = 1 # y
C[2,2] = 1 # z

C[3,9] = 1 # Roll rate (gamma)
C[4,10]  = 1 # Pitch rate (beta)
C[5,5] = 1 # Yaw

# Check observability

Oc = ctrb(A.T,C.T)
Obs_rank = rank(Oc)
print('Rank of observability matrix')
print(Obs_rank)

# ------------------------------

# State vector
# [𝑥 𝑦 𝑧] Linear position in Earth reference frame
# [𝜙 𝜃 𝜓] Angular position in Earth reference frame
# [x_dot y_dot z_dot] Linear velocity in Earth reference frame
# [𝑝 𝑞 𝑟] Angular velocity in body reference frame

# Actual initial state
# [x y z phi theta psi x_dot y_dot z_dot p q r]
x0 = np.array([[0.01],[0.04],[0.02],[0.0],[0.0],[0.03],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])

# observer initial state
xhat0 = np.zeros((1,12))
#TODO: Why not 12,1?

# desired output reference 

y_ref_OTS = np.vstack([
    np.hstack([np.zeros(T // 3), -1 * np.ones(2 * T // 3)]),  # x
    np.zeros(T),                                              # z
    np.hstack([np.zeros(T // 3), 1 * np.ones(2 * T // 3)]),   # y
    np.zeros(T),                                              # gamma
    np.zeros(T),                                              # beta
    np.hstack([np.zeros(T // 3), 1 * np.ones(2 * T // 3)])    # yaw
])


# Optimal Target Selection reference states and inputs
x_ref_OTS = np.zeros((12,T))
u_ref_OTS = np.zeros((4,T))
dhat = np.zeros((dim_d,T))
 
dim_d = 1
# disturbance input (x)
d_dist = np.vstack([
    np.hstack([np.zeros((dim_d, T // 15)), 0.01 * np.ones((dim_d, T // 15)), np.zeros((dim_d, 8 * T // 15)), 0.01 * np.ones((dim_d, 5 * T // 15))]),
    np.hstack([np.zeros((dim_d, T // 15)), 0.01 * np.ones((dim_d, T // 15)), np.zeros((dim_d, 5 * T // 15)), 0.01 * np.ones((dim_d, 8 * T // 15))])
])
dim_d = 2


x = np.zeros((len(A[:,0]),T))        # state trajectory
yplot = np.zeros((len(A[:,0]),T))    # output to plot
xhat = np.zeros((len(A[:,0]),T))     # estimated trajectories 
xhaug = np.zeros((len(A[:,0])+dim_d,T))  # augmented states (12 + 2) x + d

u = np.zeros((len(B[0,:]),T))     # control inputs
y = np.zeros((len(C[:,0]),T))        # measurements 
yhat = np.zeros((len(C[:,0]),T))     # estimated output

e = np.zeros((len(A[0,:]),T))        # observer error
t = np.zeros((1,T))                     # time vector

Vf = np.zeros((1,T))                   # terminal cost sequence
l = np.zeros((1,T))                    # stage cost sequence

x[:,0] = x0.T.flatten()

# Define MPC Control Problem

# MPC cost function
#          N-1
# V(u_N) = Sum 1/2[ x(k)'Qx(k) + u(k)'Ru(k) ] + x(N)'Sx(N) 
#          k = 0


# tuning weights
Q = 10*np.eye(A.shape[0])           # state cost
R = 0.1*np.eye(len(B[0,:]))    # input cost

# terminal cost = unconstrained optimal cost (Lec 5 pg 6)
S = solve_discrete_are(A, B, Q, R)  # terminal cost (we know it as P)

# prediction horizon
N = 10; 

Qbar = np.kron(Q,np.eye(N))
Rbar = np.kron(R,np.eye(N))
Sbar = S

# System matrices
LTI = {
    'A': A,   # Define A earlier
    'B': B,   # Define B earlier
    'C': C    # Define C earlier (will be overwritten in predmodgen anyway)
}

# Dimensions dictionary
dim = {
    'N': N,
    'nx': dim_x,
    'nu': dim_u,
    'ny': dim_y
}

# Generate prediction matrices
P, Z, W = predmodgen(LTI, dim)

# Cost function matrices
H = Z.T @ Qbar @ Z + Rbar + 2 * W.T @ Sbar @ W

# Initial state (must be defined earlier)
d = (x0.T @ P.T @ Qbar @ Z + 2 * x0.T @ (np.linalg.matrix_power(A, N).T) @ Sbar @ W).T

# ------------------------------

# Disturbance model
Bd = np.array([
    [1, 0],     # wind disturbance in x direction
    [0, 1],     # wind disturbance in y direction
    [0, 0],    
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
])  # shape (12, 2)

Cd = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1],
    [0, 0]
])  # shape (6, 2)


# Augmented system
Aaug = np.block([
    [A, Bd],
    [np.zeros((dim_d, A.shape[1])), np.eye(dim_d)]
])

Baug = np.vstack([B, np.zeros((dim_d, B.shape[1]))])

Caug = np.hstack([C, Cd])

# Observability check
test_Obs = np.vstack([
    np.hstack([np.eye(12) - A, -Bd]),
    np.hstack([C, Cd])
])
print("Rank of Augmented System n+nd = 14 for full rank")
print(np.linalg.matrix_rank(test_Obs))

# Kalman filter design
Q_kf = 1 * np.eye(Caug.shape[0])     # R in DARE (output noise covariance)
R_kf = 1 * np.eye(Aaug.shape[0])     # Q in DARE (state noise covariance)


# Solve DARE: P = A'.P.A - A'.P.C'.inv(C.P.C'+R).C.P.A + Q

# Inputs match MATLAB dare(A', C', R, Q)
P_obs = solve_discrete_are(Aaug.T, Caug.T, R_kf, Q_kf)

# Kalman observer gain: K = P * C^T * (C * P * C^T + Q)^(-1)
#Obs_gain = P_obs @ Caug.conj() @ np.linalg.inv(Caug @ P_obs @ Caug.conj() + Q_kf)
Obs_gain = np.linalg.inv(Caug @ P_obs @ Caug.T + Q_kf) @ (Caug @ P_obs @ Aaug.T)
Obs_gain = Obs_gain.T
# Closed-loop eigenvalues
Obs_eigvals = np.linalg.eigvals(Aaug - Obs_gain @ Caug)

print("Eigenvalues of Kalman Filter Observer:")
print(np.abs(Obs_eigvals))


# ------------------------------


# Control limits
u_ub = np.array([
    [10.0],  # Thrust limit  
    [0.5],  # Roll limit
    [0.5],  # Pitch limit
    [0.5]   # Yaw limit
])
u_lb = np.array([
    [0.0],  # Thrust limit  
    [-0.5],  # Roll limit
    [-0.5],  # Pitch limit
    [-0.5]   # Yaw limit
])

u_ub = np.tile(np.array([
    [10.0],  # Thrust limit  
    [0.5],  # Roll limit
    [0.5],  # Pitch limit
    [0.5]   # Yaw limit
]), (N, 1))

u_lb = np.tile(np.array([
    [0.0],  # Thrust limit  
    [-0.5],  # Roll limit
    [-0.5],  # Pitch limit
    [-0.5]   # Yaw limit
]), (N, 1))


print("------------------------------------------------------------------")
print("                Simulating Output MPC System")
print("------------------------------------------------------------------\n")
print(f"Simulation time: {T_sim} seconds\n")
print(range(T))
for k in range(T-1):
    t[0,k] = k * Ts
    if (t[0,k] % 1) == 0:
        print(f"t = {int(t[0,k])} sec")
        
    # --- Optimal Target Selector (OTS) ---
    Q_OTS = np.eye(12)
    R_OTS = np.eye(4)
    J_OTS = np.block([
        [Q_OTS, np.zeros((12, 4))],
        [np.zeros((4, 12)), R_OTS]
    ])

    A_OTS = np.block([
        [np.eye(12) - A, B],
        [C, np.zeros((6, 4))]
    ])
    b_OTS = np.vstack([
        Bd @ dhat[:, k].reshape(-1, 1),
        y_ref_OTS[:, k].reshape(-1, 1) - Cd @ dhat[:, k].reshape(-1, 1)
    ])

    #print("A_OTS:")
    #print(A_OTS.shape)

    #print("b_OTS:")
    #print(b_OTS.shape)

    # Solve QP for OTS
    xr_ur = cp.Variable((16, 1))
    '''
    prob_OTS = cp.Problem(
        cp.Minimize(cp.quad_form(xr_ur, J_OTS)),
        [A_OTS @ xr_ur == b_OTS]
    )
    prob_OTS.solve(solver=cp.OSQP,verbose=True)
    '''
    alpha = 1e-4  # regularization weight
    objective = cp.Minimize(cp.sum_squares(A_OTS @ xr_ur - b_OTS) + alpha * cp.quad_form(xr_ur, J_OTS))

    prob = cp.Problem(objective)
    prob.solve(solver=cp.OSQP)  # or ECOS
    
    

    x_ref_OTS[:, k] = xr_ur.value[:12, 0]
    u_ref_OTS[:, k] = xr_ur.value[12:16, 0]

    # --- Compute control input using MPC ---
    x0_est = xhaug[:12, k] - x_ref_OTS[:, k]
    x0_est = x0_est.reshape(-1, 1)



    d = (x0_est.T @ P.T @ Qbar @ Z + 2 * x0_est.T @ np.linalg.matrix_power(A, N).T @ Sbar @ W).T

    u_N = cp.Variable((4 * N, 1))
    prob_MPC = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(u_N, H) + d.T @ u_N),
        [u_N >= u_lb, u_N <= u_ub]
    )
    prob_MPC.solve(solver=cp.OSQP,verbose=False)

    u[:, k] = u_N.value[:4, 0]

    # --- System update ---
    B_dist = Bd
    x[:, k+1] = A @ x[:, k] + B @ u[:, k] + B_dist @ d_dist[:, k]
    y[:, k] = C @ x[:, k]+Cd @ d_dist[:, k]

    Cdplot = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)

    yplot[:, k] = C_plot @ x[:, k] + Cdplot * d_dist[0, k]

    # --- Augmented Observer ---
    yhat[:, k] = Caug @ xhaug[:, k]
    xhaug[:, k+1] = Aaug @ xhaug[:, k] + Baug @ u[:, k] + Obs_gain @ (y[:, k] - yhat[:, k])
    dhat[:, k+1] = xhaug[12:14, k+1]

    # --- Stability Analysis ---
    Q_stab = 10 * np.eye(12)
    R_stab = 0.1 * np.eye(4)
    X = solve_discrete_are(A, B, Q_stab, R_stab)

    Vf[:,k] = 0.5 * x[:, k].T @ X @ x[:, k]
    l[:,k] = 0.5 * x[:, k].T @ Q_stab @ x[:, k]

# Final state trajectory
states_trajectory = yplot.T

# Extract first three states
x = states_trajectory[:, 0]
y = states_trajectory[:, 1]
z = states_trajectory[:, 2]

# Plot in 3D
fig_3d = plt.figure(figsize=(10, 6))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.plot(x, y, z, label='Trajectory (states 1–3)')
ax_3d.set_xlabel('State 1 (x)')
ax_3d.set_ylabel('State 2 (y)')
ax_3d.set_zlabel('State 3 (z)')
ax_3d.set_title('3D Trajectory of First Three States')
ax_3d.legend()
plt.tight_layout()
plt.show()

# Plot in 2D
fig_2d, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
time = np.arange(T)

axs[0].plot(time, x)
axs[0].set_ylabel('State 1 (x)')

axs[1].plot(time, y)
axs[1].set_ylabel('State 2 (y)')

axs[2].plot(time, z)
axs[2].set_ylabel('State 3 (z)')
axs[2].set_xlabel('Time Step')

fig_2d.suptitle('2D Plots of First Three States Over Time')
plt.tight_layout()
plt.show()

# Save results
saved_data = {
    't': t,
    'x': yplot,
    'u': u
}
