import numpy as np


def rot_mat(phi,theta,psi):
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
    return R

def ang_rot_mat(psi,theta,phi):
    # Compute angular rotation matrix from Euler angles
    T = np.array(
        [
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
        ]
    )
    return T

# State vector
# [𝑥 𝑦 𝑧] Linear position in Earth reference frame
# [x_dot y_dot z_dot] Linear velocity in Earth reference frame
# [𝜙 𝜃 𝜓] Angular position in Earth reference frame
# [𝑝 𝑞 𝑟] Angular velocity in body reference frame

# Input vector
# [f1, f2, f3, f4] Motor thrust in body reference frame
# f_i = thrust from motor i (i=1,2,3,4), f_i = kF * (ω_i)^2, where k is a constant and ω_i is the angular velocity of motor i

# Parameters
m = 0.547 # Quadcopter mass 𝑚 0.547 [𝑘𝑔]
g= 9.81 # Acceleration of gravity 𝑔 9.81 [𝑚/𝑠2]
L = 0.17 # Arm length 𝐿 0.17 [𝑚]
kF = 1.5e-7 # Thrust coefficient 𝑘𝐹 1.5 · 10−7 [𝑁 · 𝑅𝑃𝑀 −2]
kM = 3.75e-7 # Torque coefficient 𝑘 𝑀 3.75 · 10−7 [𝑁𝑚 · 𝑅𝑃𝑀 −2]
gamma = kM/kF
# Air resistance coefficient 𝐶𝑑 1.0 [-]
Ix = 3.3e-3 # Inertia for 𝑥𝑏 𝐼𝑥 3.3 · 10−3 [𝑘𝑔 · 𝑚2]
Iy = 3.3e-3 # Inertia for 𝑦𝑏 𝐼𝑦 3.3 · 10−3 [𝑘𝑔 · 𝑚2]
Iz = 5.8e-3 # Inertia for 𝑧𝑏 𝐼𝑧 5.8 · 10−3 [𝑘𝑔 · 𝑚2]
# Cross-sectional area for 𝑥𝑏 𝐴𝑥 0.011 [𝑚2]
# Cross-sectional area for 𝑦𝑏 𝐴𝑦 0.011 [𝑚2]
# Cross-sectional area for 𝑧𝑏 𝐴𝑧 0.022 [𝑚2]

def nonlin_dynamics(x,u):

    assert x.shape == (12, 1), "State vector x must be of shape (12, 1)"
    assert u.shape == (4, 1), "Input vector u must be of shape (4, 1)"


    F = np.array([[0], [0], [np.sum(u)]])
    M = np.array([[0,L,0,-L],[-L,0,L,0],[gamma,-gamma,gamma,-gamma]]) @ u
    J = np.array([[Ix, 0, 0], [0, Iy, 0], [0, 0, Iz]])
    J_inv = np.linalg.inv(J)

    phi = x[6, 0]  # Roll angle
    theta = x[7, 0]  # Pitch angle
    psi = x[8, 0]  # Yaw angle

    # Normalize angles to the range [-pi, pi]
    x[6, 0] = (phi + np.pi) % (2 * np.pi) - np.pi
    x[7, 0] = (theta + np.pi) % (2 * np.pi) - np.pi
    x[8, 0] = (psi + np.pi) % (2 * np.pi) - np.pi

    phi = x[6, 0]  # Roll angle
    theta = x[7, 0]  # Pitch angle
    psi = x[8, 0]  # Yaw angle

    x_dot = np.zeros((12, 1))

    x_dot[0:3] = x[3:6]  # Linear velocity in Earth reference frame
    x_dot[3:6] = 1/m * (np.array([0, 0, -m*g]).reshape(3, 1) + rot_mat(phi,theta,psi)@F) # Gravity in the z-velocity
    x_dot[6:9] = ang_rot_mat(psi,theta,phi) @ x[9:12]  # Angular velocity in Earth reference frame

    x_dot[9:12] = J_inv @ (M - np.reshape(np.cross(x[9:12].flatten(), (J @ x[9:12]).flatten()),(3,1)))  # Angular acceleration in body reference frame

    return x_dot