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
import easygui
from numpy.linalg import matrix_rank


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

def terminal_set(P, K, c, upper_bounds, lower_bounds):
    stable = True
    max_values = np.zeros(P.shape[0])
    eigenvalues, eigenvectors = eigh(P)
    axes_length = np.sqrt(c / eigenvalues)
    sign_combinations = np.array(np.meshgrid(*[[-1, 1]] * len(axes_length))).T.reshape(-1, len(axes_length))
    corners = sign_combinations * axes_length
    corners = corners @ eigenvectors.T

    for corner in corners:
        max_values = np.maximum(max_values, np.abs(corner))
        if not np.linalg.norm(K @ corner) <= 0.5:
            stable = False

    for i in range(max_values.shape[0]):
        if max_values[i] < upper_bounds[i] and max_values[i] > lower_bounds[i]:
            continue
        else:
            stable = False
            break

    if stable:
        print("System is Stable")
    else:
        print("System is Unstable")

    print("Max values across corners:", max_values)
    return max_values




    
def mpc(A, B, N, x0, x_ref, u_ref,yref, Q, R, P,dim_x,dim_u,d,upper_bounds,lower_bounds):



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

        if k>0:
            for i in range(12):
                constraints += [x[i, k] <= upper_bounds[i], x[i, k] >= lower_bounds[i]]

    constraints += [x[:, 0] == x0]
    # Terminal cost
    cost += 1*cp.quad_form(x[:, N] - x_ref[N, :], P)

    cost *= 1e-3 


    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP,verbose=False,warm_start = True,eps_abs=1e-5, eps_rel=1e-5, max_iter=10000)

    print("📊 Solver status:", problem.status)
    print("✅ u.value is None?", u.value is None)
    print("✅ x.value is None?", x.value is None)


    if u.value is None or x.value is None:
        print("❌ Solver returned no solution.")
        return None, None, None, None
    else:
        print("✅ MPC solution obtained (even if marked inaccurate).")



    return u[:, 0].value, x[:, 1].value, x[:, :].value, u[:, :].value


disturbances = False
new_trajectory = True

N = 20

point = [[0,0,0,0,0,0]]
radius = 5
height = 5
circle_values = [[5,5]]

choice = easygui.choicebox("Select trajectory type:", "Trajectory type", ["Pre loaded", "Single point", "Circle","Tu Delft","Bread"])

if choice == "Pre loaded":
    new_trajectory = False
    traj = "nothing"
elif choice == "Single point":

    fields = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    defaults = ["5", "5", "5", "0","0","0"]

    traj = "p"
    values = easygui.multenterbox("Enter X Y Z Coordinates:", "Coordinate Input", fields, defaults)

    point = np.array([[float(v.strip()) for v in values]])
elif choice == "Circle":
    traj = "circle"
    values = easygui.multenterbox("Enter Radius and Height:", "Coordinate Input", ["Radius", "Height"], ["5","5"])
    circle_values = np.array([[float(v.strip()) for v in values]])
elif choice == "Tu Delft":
    traj = "tudelft"
elif choice == "Bread":
    traj = "bread"

disturbances = easygui.ynbox("Do you want disturbances?", "Confirm")



with open("quadrotor.yaml", "r") as file:
    params = yaml.safe_load(file)

g = params["g"]  # m/s^2
Ix = params["Ix"]  # kg m^2
Iy = params["Iy"]  # kg m^2
Iz = params["Iz"]  # kg m^2
m = params["m"]  # kg

lower_bounds = [-1000,-1000,0.0,-0.17,-0.17,-2*np.pi,-5,-5,-3,-2,-3,-3]
upper_bounds = [1000,1000,1000,0.17,0.17,2*np.pi,5,5,3,2,3,3]

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


terminal_set(Pl,K,0.25,upper_bounds,lower_bounds)


x0 = np.zeros(12)

C = np.zeros((6, 12))
C[0, 0] = 1  # x
C[1, 1] = 1  # y
C[2, 2] = 1  # z
C[3, 3] = 1  # roll
C[4, 4] = 1  # pitch
C[5, 5] = 1  # yaw




A_aug = np.block([
    [Ad, np.zeros((12, 1))],
    [np.zeros((1, 12)), np.eye(1)]
])
B_aug = np.vstack([Bd, np.zeros((1, 4))])

# Only z position is affected by the disturbance
C_d = np.zeros((6, 1))
C_d = np.zeros((6, 1))
C_d[0, 0] = 1.1  # x
C_d[1, 0] = 1.1  # y
C_d[2, 0] = 1.0  # z
C_d[3, 0] = 1.0
C_d[4, 0] = 1.0
C_d[5, 0] = 1.0
C_aug = np.hstack([C, C_d])  # now shape (6, 13)
O = np.vstack([C_aug @ np.linalg.matrix_power(A_aug, i) for i in range(13)])
print("Observability matrix rank:", matrix_rank(O))




target_selector = OptimalTargetSelection.OptimalTargetSelection(Ad, Bd, C, Q, R,m,traj,point[0][0],point[0][1],point[0][2],point[0][3],point[0][4],point[0][5],circle_values[0][0],circle_values[0][1])
if new_trajectory:
    target_selector.trajectory_gen()



x_ref = np.load("trajectories/xr_opt.npy", allow_pickle=True)
u_ref = np.load("trajectories/ur_opt.npy", allow_pickle=True)
y_ref = np.load("trajectories/yref.npy",allow_pickle=True)



# Simulation
N_sim = x_ref.shape[0]
x_hist = np.zeros((N_sim + 1, 12))
u_hist = np.zeros((N_sim, 4))
x_hist[0, :] = x0

if disturbances:
    # Add a slower pole for the disturbance manually
    L_poles = np.concatenate([np.linspace(0.95, 0.99, 12), [0.98]])

    L_aug = place_poles(A_aug.T, C_aug.T, L_poles).gain_matrix.T  # might still work!



y_cnt = 0

x_hat = np.zeros((N_sim + 1, 12))
d_hat = np.zeros((N_sim + 1, 1))  # now a 1D disturbance



d_est_now = np.zeros(3) 


if disturbances:
    # dk = np.random.normal(0, 0.05, size=12)
    dk = np.zeros(12) 
    dk[2] = 1e-2
else:
    dk = np.zeros(12)

for t in tqdm(range(N_sim), desc="Simulating MPC"):
    if t % 50 == 0 and t > 0:
        y_cnt += 1
    

    y_cnt = y_cnt % len(y_ref)


    x_ref_horizon = x_ref[t:t + N + 1, :] if t + N + 1 <= x_ref.shape[0] else x_ref[t:, :]
    u_ref_horizon = u_ref[t:t + N, :] if t + N <= u_ref.shape[0] else u_ref[t:, :]

    current_horizon = x_ref_horizon.shape[0] - 1

    if current_horizon <= 0:
        print("Reached the end of the reference trajectory.")
        break

    if disturbances:
        x = x_hat[t, :]
    else:
        x = x_hist[t, :]

    if disturbances:
        d_est_now = np.array([0, 0, d_hat[t, 0]])  # use only z-disturbance
    else:
        d_est_now = np.zeros(3)

    u_0, x_1, x_traj, u_seq = mpc(Ad, Bd, current_horizon, x, x_ref_horizon, u_ref_horizon, y_ref[y_cnt], Q, R, Pl, dim_x, dim_u, d_est_now,upper_bounds,lower_bounds)
    u_hist[t, :] = u_0
    if u_0 is None:
        print(f"❌ Solver failed at timestep {t}")
        break

    if disturbances:
        # True next state (simulate plant)
        x_hist[t + 1, :] = Ad @ x_hat[t, :] + Bd @ u_0 + dk
        y_k = x_hist[t + 1, 0:6]  # only position is measured (x, y, z)

        tilde_x_hat = np.hstack([x_hat[t], d_hat[t]])
        y_hat = C_aug @ tilde_x_hat

        tilde_x_hat = A_aug @ tilde_x_hat + B_aug @ u_0 + L_aug @ (y_k - y_hat)

        x_hat[t + 1] = tilde_x_hat[:12] 
        d_hat[t + 1] = tilde_x_hat[12:]


    else:
        # No disturbance/observer, use true state evolution
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

    # Plot the orientation vectors
    ax.quiver(
        x_pos[i],
        y_pos[i],
        z_pos[i],  # Starting point
        x_body[0],
        x_body[1],
        x_body[2],
        color="r",
        length=0.5,
        normalize=True,
        label="x-body" if i == 0 else "",
    )
    ax.quiver(
        x_pos[i],
        y_pos[i],
        z_pos[i],  # Starting point
        y_body[0],
        y_body[1],
        y_body[2],
        color="g",
        length=0.5,
        normalize=True,
        label="y-body" if i == 0 else "",
    )
    ax.quiver(
        x_pos[i],
        y_pos[i],
        z_pos[i],  # Starting point
        z_body[0],
        z_body[1],
        z_body[2],
        color="b",
        length=0.5,
        normalize=True,
        label="z-body" if i == 0 else "",
    )

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

plt.plot(x_hist[:, 3][:-1], label="roll")
plt.plot(x_hist[:, 4][:-1], label="pitch")
plt.plot(x_hist[:, 5][:-1], label="yaw")
plt.legend()
plt.show()

plt.plot(x_hist[:, 6], label="vx")
plt.plot(x_hist[:, 7], label="vy")
plt.plot(x_hist[:, 8], label="vz")
plt.legend()
plt.show()

plt.plot(x_hist[:, 9], label="omega x")
plt.plot(x_hist[:, 10], label="omega y")
plt.plot(x_hist[:, 11], label="omega z")
plt.legend()
plt.show()

# Plot the control inputs
plt.plot(u_hist[:, 0], label="u1")
plt.plot(u_hist[:, 1], label="u2")
plt.plot(u_hist[:, 2], label="u3")
plt.plot(u_hist[:, 3], label="u4")

plt.legend()
plt.show()


if disturbances:
    # Plot estimation error for first 3 states
    state_error = x_hist[:N_sim] - x_hat[:N_sim]
    for i in range(3):
        plt.plot(state_error[:, i], label=f"State {i} error")
    plt.legend()
    plt.title("State Estimation Errors (x, y, z)")
    plt.xlabel("Timestep")
    plt.ylabel("Error")
    plt.grid(True)
    plt.show()

    # Plot estimated vs true disturbances for first 3 states
    for i in range(3):
        disturbance_est = [x_hist[t + 1, i] - x_hat[t + 1, i] for t in range(N_sim - 1)]
        plt.plot(disturbance_est, label=f"Estimated disturbance {i}")
        plt.axhline(dk[i], linestyle='--', color='gray', label=f"True disturbance {i}" if i == 0 else "")
    plt.legend()
    plt.title("Estimated vs. True Disturbances (x, y, z)")
    plt.xlabel("Timestep")
    plt.ylabel("Disturbance Value")
    plt.grid(True)
    plt.show()
