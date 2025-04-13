import numpy as np
import cvxpy as cp
import pandas as pd
import os
from tqdm import tqdm


class OptimalTargetSelection:

    def __init__(self,Ad,Bd,C,Q,R,m,debug,trajectory = "circle",x=2,y=2,z=2,roll=0,pitch=0,yaw=0,radius=5,height=5):
        self.Ad = Ad
        self.Bd = Bd
        self.C = C
        self.Q = Q
        self.R = R
        self.m = m

        self.x = x
        self.y = y
        self.z = z

        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

        self.radius = radius
        self.height = height

        self.trajectory = trajectory

        self.debug = debug

        g = 9,81 #m/s^2

    def trajectory_gen(self):
        I = np.eye(self.Ad.shape[0])


        N = 50
        xr = cp.Variable((N,12))
        ur = cp.Variable((N,4))

        if self.trajectory == "circle":
            yrefs = self.circular_trajectory(radius=self.radius,height=self.height)
        elif self.trajectory == "eight":
            yrefs = self.figure_eight_trajectory()
        elif self.trajectory == "tudelft":
            yrefs = self.tudelft_trajectory()
        elif self.trajectory == "cyprus":
            yrefs = self.cyprus_trajectory()
        elif self.trajectory == "bread":
            yrefs = self.bread_trajectory()
        else:
            yrefs = np.array([[self.x,self.y,self.z,self.roll,self.pitch,self.yaw]])



        xr_combined = []
        ur_combined = []

        for yref in tqdm(yrefs,desc="Generating Trajectory"): 
            # Set up optimization problem
            xr = cp.Variable((N, 12))
            ur = cp.Variable((N, 4))

            cost = 0
            constraints = []

            yref_full = np.zeros(12)
            yref_full[:6] = yref  

            for k in range(N-1):

                cost += cp.quad_form(xr[k], self.Q) + cp.quad_form(ur[k], self.R)
                constraints += [xr[k+1] == self.Ad @ xr[k] + self.Bd @ ur[k]]
                constraints += [self.C @ xr[k] == yref_full]
                constraints += [ur[k][0] >= self.m * (-9.81)]
            cost += cp.quad_form(xr[N-1], self.Q)

            # Solve the problem
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.SCS, max_iters=10000,verbose=False)

            # Store the results
            xr_combined.append(np.array(xr.value))
            ur_combined.append(np.array(ur.value))


            xr_combined.append(np.array(xr.value))
            ur_combined.append(np.array(ur.value))

        xr_combined = np.vstack(xr_combined)
        ur_combined = np.vstack(ur_combined)
        os.makedirs("trajectories", exist_ok=True)

        np.save("trajectories/xr_opt.npy", xr_combined)
        np.save("trajectories/ur_opt.npy", ur_combined)
        np.save("trajectories/yref",yrefs)

    def trajectory_gen_with_disturbances(self,d,yref):

        N = 50

        xr = cp.Variable((N,12))
        ur = cp.Variable((N,4))

        cost = 0
        constraints = []
        yref = np.concatenate([yref, np.zeros(6)])

        

        for k in range(N - 1):
            A_aug = np.block([
                [np.eye(12) - self.Ad, -self.Bd],
                [self.C,               np.zeros((12, 4))]
            ])

            xu = cp.vstack([cp.reshape(xr[k, :], (12, 1)), cp.reshape(ur[k, :], (4, 1))])
            # Apply disturbance correction
            rhs = np.vstack([
                np.zeros((12, 1)),
                (yref.reshape(-1, 1) - d.reshape(-1, 1))  
            ])

            eps = cp.Variable((12, 1))  

            constraints += [A_aug @ xu == rhs + cp.vstack([np.zeros((12, 1)), eps])]
            cost += cp.quad_form(eps, np.eye(12)) * 1e4  

            cost += cp.quad_form(xr[k], self.Q) + cp.quad_form(ur[k], self.R)

        cost += cp.quad_form(xr[N-1], self.Q)
        
        problem = cp.Problem(cp.Minimize(cost), constraints)
        if self.debug:
            print("📌 yref[:3] =", yref[:3])
            print("📌 d[:3] =", d[:3])
            print("📌 target output =", yref[:3] - d[:3])
        problem.solve(solver=cp.SCS, max_iters=100000,verbose=False)

        return xr.value, ur.value


    def circular_trajectory(self, radius=5, height=5):
        theta = np.linspace(0, 2*np.pi, 20)
        yrefs = [np.array([radius * np.cos(t), radius * np.sin(t), 5]) for t in theta]
        
        yrefs = [np.concatenate((yref, [0, 0, 0])) for yref in yrefs]
        
        return yrefs

    def tudelft_trajectory(self):
        data = pd.read_csv("tudelft.csv")
        x_points = data.iloc[:, 0].values
        y_points = data.iloc[:, 1].values
        z_constant = 5 

        yrefs = [np.array([x, y, z_constant]) for x, y in zip(x_points, y_points)]
        
        yrefs = [np.concatenate((yref, [0, 0, 0])) for yref in yrefs]

        return yrefs

    def cyprus_trajectory(self):
        data = pd.read_csv("cyprus.csv")
        x_points = data.iloc[:, 0].values
        y_points = data.iloc[:, 1].values
        z_constant = 5 

        yrefs = [np.array([x, y, z_constant]) for x, y in zip(x_points, y_points)]
        
        yrefs = [np.concatenate((yref, [0, 0, 0])) for yref in yrefs]

        return yrefs

    def bread_trajectory(self):
        data = pd.read_csv("bread.csv")
        x_points = data.iloc[:, 0].values
        y_points = data.iloc[:, 1].values
        z_constant = 5 

        yrefs = [np.array([x, y, z_constant]) for x, y in zip(x_points, y_points)]
        
        yrefs = [np.concatenate((yref, [0, 0, 0])) for yref in yrefs]

        return yrefs

    def figure_eight_trajectory(self, radius=10):
        theta = np.linspace(0, 2 * np.pi, 30)
        yrefs = [np.array([radius * np.sin(t), radius * np.sin(t) * np.cos(t), 5]) for t in theta]
        
        yrefs = [np.concatenate((yref, [0, 0, 0])) for yref in yrefs]

        return yrefs



