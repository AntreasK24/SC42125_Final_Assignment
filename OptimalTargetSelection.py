import numpy as np
import cvxpy as cp
import pandas as pd
import os

class OptimalTargetSelection:

    def __init__(self,Ad,Bd,C,Q,R,m,trajectory = "circle"):
        self.Ad = Ad
        self.Bd = Bd
        self.C = C
        self.Q = Q
        self.R = R
        self.m = m

        self.trajectory = trajectory

        g = 9,81 #m/s^2

    def trajectory_gen(self):
        I = np.eye(self.Ad.shape[0])

        N = 50
        xr = cp.Variable((N,12))
        ur = cp.Variable((N,4))

        if self.trajectory == "circle":
            yrefs = self.circular_trajectory(radius=5)
        elif self.trajectory == "tudelft":
            yrefs = self.tudelft_trajectory()

        xr_combined = []
        ur_combined = []

        for yref in yrefs:
            xr = cp.Variable((N, 12))
            ur = cp.Variable((N, 4))

            # Set up optimization problem
            cost = 0
            constraints = []
            for k in range(N-1):
                cost += cp.quad_form(xr[k], self.Q) + cp.quad_form(ur[k], self.R)
                constraints += [xr[k+1] == self.Ad @ xr[k] + self.Bd @ ur[k]]
                constraints += [self.C @ xr[k] == yref]
            cost += cp.quad_form(xr[N-1], self.Q)

            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.OSQP)

            xr_combined.append(np.array(xr.value))
            ur_combined.append(np.array(ur.value))

        xr_combined = np.vstack(xr_combined)
        ur_combined = np.vstack(ur_combined)
        os.makedirs("trajectories", exist_ok=True)

        np.save("trajectories/xr_opt.npy", xr_combined)
        np.save("trajectories/ur_opt.npy", ur_combined)


    def circular_trajectory(self,radius = 5):
        theta = np.linspace(0,2*np.pi,10)
        yrefs =  [np.array([radius * np.cos(t), radius * np.sin(t), 5]) for t in theta]

        return yrefs
    
    def tudelft_trajectory(self):
        data = pd.read_csv("tudelft.csv")
        x_points = data.iloc[:, 0].values
        y_points = data.iloc[:, 1].values
        z_constant = 5 

        yrefs = [np.array([x, y, z_constant]) for x, y in zip(x_points, y_points)]

        return yrefs



