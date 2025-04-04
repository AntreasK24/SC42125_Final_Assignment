# dynamics from the paper DroneMPC Papers/Quadcopter Modeling with LQR.pdf - Quadcopter Modeling and Linear Quadratic Regulator Design Using Simulink

import numpy as np
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from lqr_control import LQRControl
from control import dlqr  
from scipy.linalg import solve_discrete_are,eigh
import cvxpy as cp
from tqdm import tqdm
import yaml
from scipy.linalg import solve_discrete_lyapunov
from scipy.signal import place_poles
import OptimalTargetSelection



def fd_rk4(xk, uk, dt):
    f1 = dynamics(xk, uk)
    f2 = dynamics(xk + 0.5 * dt * f1, uk)
    f3 = dynamics(xk + 0.5 * dt * f2, uk)
    f4 = dynamics(xk + dt * f3, uk)
    return xk + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)


def dynamics(xk, uk,Ts):
    xk1 = Ad @ xk + Bd @ uk
    xk1[8] -= g * Ts  # Gravity affects vertical velocity
    return xk1

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

def terminal_set(P,K,c):

    stable = True

    eigenvalues, eigenvectors = eigh(P)
    axes_length = np.sqrt(c/eigenvalues)

    corners = []
    sign_combinations = np.array(np.meshgrid(*[[-1, 1]] * len(axes_length))).T.reshape(-1, len(axes_length))

    corners = sign_combinations * axes_length

    corners = corners @ eigenvectors.T

    for corner in corners:
        if not np.linalg.norm(K@corner) <= 0.5:
            stable = False

    if stable:
        print("System is Stable")
    else:
        print("System is Unstable")
def mpc(A, B, N, x0, x_ref, u_ref,yref, Q, R, P,dim_x,dim_u,d):

    #Solve OTS online
    
    # ots_cost = 0
    # ots_constraints = 0
    # xr = cp.Variable((12))
    # ur = cp.Variable((4))
    # for k in range(N-1):
    #     cost += cp.quad_form(xr[k], Q) + cp.quad_form(ur[k], R)
    #     constraints += [xr[k+1] == A @ xr[k] + B @ ur[k]]
    #     constraints += [C @ xr[k] == yref]
    #     constraints += [ur[k][0] >= m * (-9.81)]

    # cost += cp.quad_form(xr[N-1], Q)

    # ots_problem = cp.Problem(cp.Minimize(ots_cost), ots_constraints)
    # ots_problem.solve(solver=cp.OSQP)

    if disturbances:
        xref,uref = target_selector.trajectory_gen_with_disturbances(d,yref)

        x_ref = xref
        u_ref = uref

    cost = 0.0
    constraints = []

    x = cp.Variable((dim_x, N + 1))  # cp.Variable((dim_1, dim_2))
    u = cp.Variable((dim_u, N))

    Q = np.array(Q)
    R = np.array(R)

    for k in range(N):
        cost += cp.quad_form(x[:, k] - x_ref[k, :], Q)
        cost += cp.quad_form(u[:, k] - u_ref[k, :], R)
        constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]
        constraints += [u[0, k] >= m * (-9.81)]

    constraints += [x[:, 0] == x0]
    #constraints += [cp.quad_form(0.5*x[:,N], Q) <= 5]
    # Terminal cost
    cost += 1*cp.quad_form(x[:, N] - x_ref[N, :], P)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.SCS)

    return u[:, 0].value, x[:, 1].value, x[:, :].value, u[:, :].value


disturbances = True
new_trajectory = True

N = 20

with open("quadrotor.yaml", "r") as file:
    params = yaml.safe_load(file)

g = params["g"]  # m/s^2
Ix = params["Ix"]  # kg m^2
Iy = params["Iy"]  # kg m^2
Iz = params["Iz"]  # kg m^2
m = params["m"]  # kg

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

system_poles = np.linalg.eigvals(Ad)
print("Poles of Ad: ", system_poles)

dim_x = 12
dim_u = 4



Q = np.eye(Ad.shape[0]) * 10
R = np.eye(Bd.shape[1])


P = solve_discrete_are(A,B,Q,R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)


Qk = Q + K.T @ R @ K
Ak = A + B@K

Pl = solve_discrete_lyapunov(Ak, Qk)


terminal_set(Pl,K,5)


x0 = np.zeros(12)
d = np.zeros(12)
d[8] = -g * 0.1  # Gravity in the z-velocity
C = np.zeros((3, 12))
for i in range(3):
    C[i, i] = 1

target_selector = OptimalTargetSelection.OptimalTargetSelection(Ad, Bd, C, Q, R,m,"circle")
if new_trajectory:
    target_selector.trajectory_gen()



x_ref = np.load("trajectories/xr_opt.npy", allow_pickle=True)
u_ref = np.load("trajectories/ur_opt.npy", allow_pickle=True)
y_ref = np.load("trajectories/yref.npy",allow_pickle=True)


print(y_ref)

print(y_ref.shape)


# Simulation
N_sim = x_ref.shape[0]
x_hist = np.zeros((N_sim + 1, 12))
u_hist = np.zeros((N_sim, 4))
x_hist[0, :] = x0

if disturbances:
    L_poles = np.array([0.85, 0.87, 0.89, 0.91, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999])
    L = place_poles(np.eye(dim_x), np.eye(np.eye(dim_x).shape[0]), L_poles).gain_matrix


x_hat = np.zeros((N_sim + 1, 12))  

y_cnt = 0

for t in tqdm(range(N_sim), desc="Simulating MPC"):
    if t % 50 == 0 and t > 0:
        y_cnt += 1
    
    if disturbances:
        dk = np.random.normal(0, 0.05, size=12)  
    else:
        dk = np.zeros(12)

    x_ref_horizon = x_ref[t:t + N + 1, :] if t + N + 1 <= x_ref.shape[0] else x_ref[t:, :]
    u_ref_horizon = u_ref[t:t + N, :] if t + N <= u_ref.shape[0] else u_ref[t:, :]

    current_horizon = x_ref_horizon.shape[0] - 1  # Number of steps in the current horizon

    if current_horizon <= 0:
        print("Reached the end of the reference trajectory.")
        break

    if disturbances:
        x = x_hat[t,:]
    else:
        x = x_hist[t,:]


    # Solve the MPC problem
    u_0, x_1, x_traj, u_seq = mpc(Ad, Bd, current_horizon, x, x_ref_horizon, u_ref_horizon, y_ref[y_cnt], Q, R, Pl, dim_x, dim_u, dk[:3])
    u_hist[t, :] = u_0

    if u_0 is None:
        break  # Stop simulation if there's no valid control input


    if disturbances:
        # Forward simulate the system
        x_hist[t + 1, :] = Ad @ x_hat[t, :] + Bd @ u_0 + dk 

        y_k = x_hist[t + 1, :]  # Actual output (noisy)
        x_hat[t+1,:] = Ad @ x_hat[t,:] + Bd @ u_0 + L @ (y_k - x_hat[t,:]) 
        d_est = x_hist[t + 1, :] - x_hat[t+1,:]  
    else:
        x_hist[t + 1, :] = Ad @ x_hist[t, :] + Bd @ u_0 



x_pos = x_hist[:, 0]
y_pos = x_hist[:, 1]
z_pos = x_hist[:, 2]


if disturbances:

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_hist[:, 0][:-1], x_hist[:, 1][:-1], x_hist[:, 2][:-1], label="Actual Trajectory (w/o observer)", color='b')

    ax.plot(x_hat[:, 0][:-1], x_hat[:, 1][:-1], x_hat[:, 2][:-1], label="Estimated Trajectory (w/ observer)", color='r')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Trajectory Comparison: Actual vs Estimated')

    ax.legend()

    plt.show()
else:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x_pos[:-1], y_pos[:-1], z_pos[:-1], label="Trajectory", color="b")


# Add orientation vectors
for i in range(0, x_hist.shape[0], 3):  # Every third step
    # Extract orientation angles (phi, theta, psi)
    phi = x_hist[i, 3]
    theta = x_hist[i, 4]
    psi = x_hist[i, 5]

    # Compute rotation matrix from Euler angles
    R = np.array(
        [
            [
                np.cos(psi) * np.cos(theta),
                np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi),
                np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi),
            ],
            [
                np.sin(psi) * np.cos(theta),
                np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi),
                np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi),
            ],
            [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)],
        ]
    )

    # Define unit vectors in body frame
    x_body = R[0, :]  # x-axis
    y_body = R[1, :]  # y-axis
    z_body = R[2, :]  # z-axis

    # # Plot the orientation vectors
    # ax.quiver(
    #     x_pos[i],
    #     y_pos[i],
    #     z_pos[i],  # Starting point
    #     x_body[0],
    #     x_body[1],
    #     x_body[2],
    #     color="r",
    #     length=0.5,
    #     normalize=True,
    #     label="x-body" if i == 0 else "",
    # )
    # ax.quiver(
    #     x_pos[i],
    #     y_pos[i],
    #     z_pos[i],  # Starting point
    #     y_body[0],
    #     y_body[1],
    #     y_body[2],
    #     color="g",
    #     length=0.5,
    #     normalize=True,
    #     label="y-body" if i == 0 else "",
    # )
    # ax.quiver(
    #     x_pos[i],
    #     y_pos[i],
    #     z_pos[i],  # Starting point
    #     z_body[0],
    #     z_body[1],
    #     z_body[2],
    #     color="b",
    #     length=0.5,
    #     normalize=True,
    #     label="z-body" if i == 0 else "",
    # )

# Set equal scaling for all axes
max_range = (
    np.array(
        [
            x_pos.max() - x_pos.min(),
            y_pos.max() - y_pos.min(),
            z_pos.max() - z_pos.min(),
        ]
    ).max()
    / 2.0
)

mid_x = (x_pos.max() + x_pos.min()) * 0.5
mid_y = (y_pos.max() + y_pos.min()) * 0.5
mid_z = (z_pos.max() + z_pos.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)


ax.set_xlabel("x (Position)")
ax.set_ylabel("y (Position)")
ax.set_zlabel("z (Position)")

ax.set_title("3D Position Trajectory")

plt.show()

# Plot position of the drone
plt.plot(x_hist[:, 0][:-1], label="x")
plt.plot(x_hist[:, 1][:-1], label="y")
plt.plot(x_hist[:, 2][:-1], label="z")
plt.legend()
plt.show()

# Plot tilt of the drone
plt.plot(x_hist[:, 6], label="u")
plt.plot(x_hist[:, 7], label="v")
plt.plot(x_hist[:, 8], label="w")
plt.legend()
plt.show()

# Plot the control inputs
plt.plot(u_hist[:, 0], label="u1")
plt.plot(u_hist[:, 2], label="u2")
plt.plot(u_hist[:, 2], label="u3")
plt.plot(u_hist[:, 3], label="u4")

plt.legend()
plt.show()
