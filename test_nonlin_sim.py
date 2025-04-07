# dynamics from the paper DroneMPC Papers/Quadcopter Modeling with LQR.pdf - Quadcopter Modeling and Linear Quadratic Regulator Design Using Simulink
import time
import numpy as np
import cvxpy as cp
from scipy.linalg import expm, eigvals, solve_discrete_are, eigh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from quadrotor_dynamics import nonlin_dynamics, rot_mat
from lqr_control import LQRControl
from control import dlqr


def is_stabilizable(A, B):
    eigs = eigvals(A)
    n = A.shape[0]
    for eig in eigs:
        if np.abs(eig) >= 1:
            # Check if [eig*I - A | B] has full row rank
            test_matrix = np.hstack([eig * np.eye(n) - A, B])
            if np.linalg.matrix_rank(test_matrix) < n:
                print(f"Uncontrollable unstable mode at eig = {eig}")
                return False
    return True

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

def setup_mpc(dim_x, dim_u, N):
    # Parameters
    xinit_param = cp.Parameter(dim_x)
    x_ref_param = cp.Parameter((dim_x,N + 1))
    u_ref_param = cp.Parameter((dim_u,N))
    A_param = cp.Parameter((dim_x, dim_x))
    B_param = cp.Parameter((dim_x, dim_u))
    Q_param = cp.Parameter((dim_x, dim_x), PSD=True)
    R_param = cp.Parameter((dim_u, dim_u), PSD=True)
    P_param = cp.Parameter((dim_x, dim_x), PSD=True)

    # Variables
    x = cp.Variable((dim_x, N + 1))
    u = cp.Variable((dim_u, N))

    max_thrust = 2 * m * g
    # Cost and constraints
    cost = 0
    constraints = []

    for k in range(N):
        cost += cp.quad_form(x[:, k] - x_ref_param[:, k], Q_param)
        cost += cp.quad_form(u[:, k] - u_ref_param[:, k], R_param)
        constraints += [(x[:, k + 1] - x_ref_param[:, k])== 
                        A_param @ (x[:, k] - x_ref_param[:, k])
                        + B_param @ (u[:, k]- u_ref_param[:, k])]
        constraints += [u[:, k] >= 0.0, u[:, k] <= max_thrust]
        constraints += [cp.abs(x[6, k]) <= 0.5]
        constraints += [cp.abs(x[7, k]) <= 0.5]

    constraints += [x[:, 0] == xinit_param]
    cost += cp.quad_form(x[:, N] - x_ref_param[:, N], P_param)

    problem = cp.Problem(cp.Minimize(cost), constraints)

    #print("Is DCP? ", problem.is_dcp(dpp=False))
    #print("Is DPP? ", problem.is_dcp(dpp=True))

    return {
        "problem": problem,
        "x": x,
        "u": u,
        "xinit_param": xinit_param,
        "x_ref_param": x_ref_param,
        "u_ref_param": u_ref_param,
        "A_param": A_param,
        "B_param": B_param,
        "Q_param": Q_param,
        "R_param": R_param,
        "P_param": P_param
    }

def rk4_simulate(f, x0, u, dt, T):
    """
    Simulate a nonlinear system using RK4.
    
    Parameters:
        f  : function that takes (x, u) and returns dx/dt
        x0 : initial state (numpy array)
        u  : control input (constant or function of time)
        dt : time step
        T  : total simulation time
    
    Returns:
        t_array : array of time points
        x_array : array of state vectors at each time point
    """
    t_array = np.arange(0, T + dt, dt)
    u_array = np.zeros((len(t_array), 4))
    x_array = np.zeros((len(t_array), len(x0)))
    x = x0.copy()
    
    for i, t in enumerate(t_array):
        x_array[i] = x.flatten()
        # Get current control input (allow u to be a function of time)
        u_current = u(t) if callable(u) else u
        
        u_array[i] = u_current.flatten()
        k1 = f(x, u_current)
        k2 = f(x + 0.5 * dt * k1, u_current)
        k3 = f(x + 0.5 * dt * k2, u_current)
        k4 = f(x + dt * k3, u_current)
        
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return t_array, x_array, u_array

def fd_rk4(dynamics,xk, uk, dt):
    f1 = dynamics(xk, uk)
    f2 = dynamics(xk + 0.5 * dt * f1, uk)
    f3 = dynamics(xk + 0.5 * dt * f2, uk)
    f4 = dynamics(xk + dt * f3, uk)
    return xk + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

def plot_axes(ax,x):
    """
    Plot the axes of the quadrotor in 3D space.
    
    Parameters:
        ax: Axes3D object to plot on
        x: state vector containing position and orientation
    """
    pos = x[0:3]  # Position
    phi, theta, psi = x[6:9]  # Roll, pitch, yaw

    # Rotation matrix from body frame to world frame
    R = rot_mat(phi, theta, psi)

    # Define unit vectors in body frame
    x_body = R[:, 0]  # x-axis
    y_body = R[:, 1]  # y-axis
    z_body = R[:, 2]  # z-axis

    ax.quiver(pos[0], pos[1], pos[2], x_body[0], x_body[1], x_body[2], color='r', length=1, normalize=False)
    ax.quiver(pos[0], pos[1], pos[2], y_body[0], y_body[1], y_body[2], color='g', length=1, normalize=False)
    ax.quiver(pos[0], pos[1], pos[2], z_body[0], z_body[1], z_body[2], color='b', length=1, normalize=False)

def plot_3d_trajectory(x_array):
    """
    Plot 3D trajectory from state array.
    
    Parameters:
        x_array: numpy array of shape (N, 3), where each row is a state [x, y, z]
    """
    if x_array.shape[1] != 3:
        raise ValueError("State must have 3 dimensions for 3D plot.")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x_array[:, 0], x_array[:, 1], x_array[:, 2], lw=2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('3D State Trajectory')
    plt.tight_layout()
    plt.show()


# Load parameters from quadrotor.yaml
with open("quadrotor.yaml", "r") as file:
    params = yaml.safe_load(file)

g = params["g"]  # m/s^2
Ix = params["Ix"]  # kg m^2
Iy = params["Iy"]  # kg m^2
Iz = params["Iz"]  # kg m^2
m = params["m"]  # kg
L = params["L"] #m
kF = params["kF"] #N RPM^-2
kM = params["kM"] #Nm RPM^-2
gamma = params["gamma"] 

# State vector
# [𝑥 𝑦 𝑧] Linear position in Earth reference frame
# [x_dot y_dot z_dot] Linear velocity in Earth reference frame
# [𝜙 𝜃 𝜓] Angular position in Earth reference frame
# [𝑝 𝑞 𝑟] Angular velocity in body reference frame

# Define Equilibrium state
x_ref = 0.0 # x position
y_ref = 0.0 # y position
z_ref = 0.0 # z position
psi_ref = 0.0 # yaw

x0 = np.array([ [x_ref], [y_ref], [z_ref],
                [0.0], [0.0], [0.0],
                [0.0], [0.0], [psi_ref],
                [0.0], [0.0], [0.0]   ])

'''
Addition: Dynamic Linearization around new point / yaw with sympy
'''

# Linearized continuous-time error dynamics
dim_x = 12
A = np.zeros((dim_x, dim_x))
A[0, 3] = 1
A[1, 4] = 1
A[2, 5] = 1
A[3, 6] = g*np.sin(x0[8, 0])
A[3, 7] = g*np.cos(x0[8, 0])
A[4, 6] = -g*np.cos(x0[8, 0])
A[4, 7] = g*np.sin(x0[8, 0])

A[6, 9] = 1
A[7, 10] = 1
A[8, 11] = 1

# Control vector
# f1 = 𝑘𝐹*𝜔1^2 (Force rotor 1)
# f2 = 𝑘𝐹*𝜔2^2 (Force rotor 2)
# f3 = 𝑘𝐹*𝜔3^2 (Force rotor 3)
# f4 = 𝑘𝐹*𝜔4^2 (Force rotor 4)

# Define Equilibrium Input
u0 = m*g/4 * np.array([ [1.0], [1.0], [1.0], [1.0] ])

dim_u = 4
B = np.zeros((dim_x, dim_u))
B[5, 0] = 1 / m
B[5, 1] = 1 / m
B[5, 2] = 1 / m
B[5, 3] = 1 / m
B[9, 1] = L / Ix
B[9, 3] = -L / Ix
B[10, 0] = -L / Iy
B[10, 2] = L / Iy
B[11, 3] = gamma / Iz
B[11, 3] = -gamma / Iz
B[11, 3] = gamma / Iz
B[11, 3] = -gamma / Iz

# Linearized system
# \dot{x_error} = A*x_error + B*u_error
# x_error = x - x_0
# u_error = u - u_0

# Discretize Linearizes continuous time dynamics
Ts = 0.05 # Sampling time
Ad, Bd = discretize(A, B, Ts)

# x_error(k+1) = A*x_error(k) + B*u_error(k)
# x_error(k) = x(k) - x_0
# u_error(k) = u(k) - u_0


# MPC Setup
Q = np.diag([
    100, 100, 100,     # x, y, z
    1,   1,   1,       # vx, vy, vz
    10,  10,  1,       # roll, pitch, yaw (low weight for yaw)
    1,   1,   1        # angular rates
])
R = np.eye(Bd.shape[1]) * 0.1

P = solve_discrete_are(Ad, Bd, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

terminal_set(P, K, 0.01)

print("(A,B) is stabilizable:", is_stabilizable(A, B))



# Simulate
#t_array, x_array, u_array = rk4_simulate(nonlin_dynamics, x0, u, dt=0.01, T=10.0)

# Initial state
x_init = np.zeros((dim_x,1))
#x_init[0:3] = 0.001

#x_ref = np.load("trajectories/xr_opt.npy", allow_pickle=True)
#u_ref = np.load("trajectories/ur_opt.npy", allow_pickle=True)

# Simulation
N_sim = 100
x_hist = np.zeros((N_sim + 1, dim_x))
u_hist = np.zeros((N_sim, dim_u))
x_hist[0, :] = x_init.flatten()
N = 20  # Prediction horizon

# Set up the solver ONCE, instead at every time step:
mpc_data = setup_mpc(dim_x=12, dim_u=4, N=N)


# Live trajectory update:
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Storage for trajectory points
trajectory = []
objective_vals = []
state_error = []

try:
    for t in tqdm(range(N_sim), desc="Simulating MPC"):
        start_time = time.perf_counter()
        # x_error(k+1) = A*x_error(k) + B*u_error(k)
        # x_error(k) = x(k) - x_0
        # u_error(k) = u(k) - u_0
        '''
        # Solve the MPC problem
        u_0_error, x_1_error, _, _ = mpc(Ad, Bd, N, x_hist[t, :], x0.flatten(), u0.flatten(), Q, R, P)
        '''

        # Update MPC parameters
        mpc_data["xinit_param"].value = x_hist[t, :].flatten()
        mpc_data["x_ref_param"].value = np.tile(x0.flatten().reshape(-1, 1), (1, N + 1))
        mpc_data["u_ref_param"].value = np.tile(u0.flatten().reshape(-1, 1), (1, N))
        mpc_data["A_param"].value = Ad
        mpc_data["B_param"].value = Bd
        mpc_data["Q_param"].value = Q
        mpc_data["R_param"].value = R
        mpc_data["P_param"].value = P
        
        # Warm-start from reference
        mpc_data["x"].value = mpc_data["x_ref_param"].value.copy()
        mpc_data["u"].value = mpc_data["u_ref_param"].value.copy()

        # Solve
        mpc_data["problem"].solve(solver=cp.OSQP, 
                                  warm_start=True,
                                  max_iter=20000,
                                  eps_abs=1e-3,
                                  eps_rel=1e-3,
                                  verbose=True)
        if mpc_data["problem"].status != "optimal":
            raise RuntimeError(f"MPC failed at step {t}")

        # Retrieve control input and next state
        u_0 = mpc_data["u"][:, 0].value
        x_1 = mpc_data["x"][:, 1].value

        state_error.append((x_1-x0.flatten()))

        u_hist[t, :] = u_0
        objective_vals.append(mpc_data["problem"].value)

        # Forward simulate the system
        x_next = fd_rk4(nonlin_dynamics, 
                        x_hist[t, :].reshape(12,1), 
                        u_0.reshape(4,1), 
                        Ts).flatten()
        x_hist[t + 1, :] = x_next



        # ---- 3D PLOTTING ----
        pos = x_next[0:3]  # Assuming x[0], x[1], x[2] = x, y, z
        trajectory.append(pos)
        # Real time
            # --- Wait for the rest of the sampling period ---
        elapsed = time.perf_counter() - start_time
        sleep_time = Ts - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


        # Plotting
        if len(trajectory) > 1:

            traj_arr = np.array(trajectory)

            ax.cla()  # Clear axes
            ax.plot3D(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2], 'b-')  # Full trajectory
            ax.scatter3D(pos[0], pos[1], pos[2], color='red')  # Current position
            plot_axes(ax,x_next)

            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])
            ax.set_zlim([-max(0.5,abs(pos[2])), max(0.5,abs(pos[2]))])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"Step {t}")

            plt.draw()
            plt.pause(0.01)

except Exception as e:
    print(f"Simulation crashed: {e}")
finally:
    plt.ioff()
    plt.show()

plot = True

if plot == True:
    plot_3d_trajectory(x_hist[:, :3])  # take first 3 states if you have more

    # Plot position of the drone
    plt.plot(objective_vals, label="obj")
    plt.legend()
    plt.title("Objective Value")
    plt.show()

    # Plot position of the drone
    labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r"]
    plt.plot(state_error, label=labels)
    plt.legend()
    plt.title("State Error")
    plt.show()

    # Plot position of the drone
    plt.plot(x_hist[:, 0], label="x")
    plt.plot(x_hist[:, 1], label="y")
    plt.plot(x_hist[:, 2], label="z")
    plt.title("Position")
    plt.legend()
    plt.show()

    # Plot tilt of the drone
    plt.plot(x_hist[:, 6], label="roll")
    plt.plot(x_hist[:, 7], label="pitch")
    plt.plot(x_hist[:, 8], label="yaw")
    plt.title("Tilt")
    plt.legend()
    plt.show()

    # Plot the control inputs
    plt.plot(u_hist[:, 0], label="u1")
    plt.plot(u_hist[:, 1], label="u2")
    plt.plot(u_hist[:, 2], label="u3")
    plt.plot(u_hist[:, 3], label="u4")
    plt.title("Control Inputs")
    plt.ylim(0.5, 3.0)
    plt.legend()
    plt.show()

