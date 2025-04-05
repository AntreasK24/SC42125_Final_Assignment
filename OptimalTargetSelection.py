import numpy as np
import cvxpy as cp
import pandas as pd
import os

class OptimalTargetSelection:

    def __init__(self,Ad,Bd,C,Q,R,m,trajectory = "circle",x=5,y=5,z=5,roll=0,pitch=0,yaw=0,radius=5,height=5):
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

        for yref in yrefs:
            # Set up optimization problem
            xr = cp.Variable((N, 12))
            ur = cp.Variable((N, 4))

            cost = 0
            constraints = []

            for k in range(N-1):
                cost += cp.quad_form(xr[k], self.Q) + cp.quad_form(ur[k], self.R)
                constraints += [xr[k+1] == self.Ad @ xr[k] + self.Bd @ ur[k]]
                constraints += [self.C @ xr[k] == yref]
                constraints += [ur[k][0] >= self.m * (-9.81)]
            cost += cp.quad_form(xr[N-1], self.Q)

            # Solve the problem
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.SCS, max_iters=10000,verbose=True)

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

        

        for k in range(N-1):
            
            cost += cp.quad_form(xr[k], self.Q) + cp.quad_form(ur[k], self.R)
            constraints += [xr[k+1] == self.Ad @ xr[k] + self.Bd @ ur[k]]
            constraints += [self.C @ xr[k] == (yref - d)]
            constraints += [ur[k][0] >= self.m * (-9.81)]

        cost += cp.quad_form(xr[N-1], self.Q)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.SCS, max_iters=10000,verbose=True)

        return xr.value, ur.value


    def circular_trajectory(self, radius=5, height=5):
        theta = np.linspace(0, 2*np.pi, 20)
        yrefs = [np.array([radius * np.cos(t), radius * np.sin(t), 5]) for t in theta]
        
        # Append [0, 0, 0] to each vector
        yrefs = [np.concatenate((yref, [0, 0, 0])) for yref in yrefs]

        return yrefs

    def tudelft_trajectory(self):
        data = pd.read_csv("tudelft.csv")
        x_points = data.iloc[:, 0].values
        y_points = data.iloc[:, 1].values
        z_constant = 5 

        yrefs = [np.array([x, y, z_constant]) for x, y in zip(x_points, y_points)]
        
        # Append [0, 0, 0] to each vector
        yrefs = [np.concatenate((yref, [0, 0, 0])) for yref in yrefs]

        return yrefs

    def cyprus_trajectory(self):
        data = pd.read_csv("cyprus.csv")
        x_points = data.iloc[:, 0].values
        y_points = data.iloc[:, 1].values
        z_constant = 5 

        yrefs = [np.array([x, y, z_constant]) for x, y in zip(x_points, y_points)]
        
        # Append [0, 0, 0] to each vector
        yrefs = [np.concatenate((yref, [0, 0, 0])) for yref in yrefs]

        return yrefs

    def bread_trajectory(self):
        data = pd.read_csv("bread.csv")
        x_points = data.iloc[:, 0].values
        y_points = data.iloc[:, 1].values
        z_constant = 5 

        yrefs = [np.array([x, y, z_constant]) for x, y in zip(x_points, y_points)]
        
        # Append [0, 0, 0] to each vector
        yrefs = [np.concatenate((yref, [0, 0, 0])) for yref in yrefs]

        return yrefs

    def figure_eight_trajectory(self, radius=10):
        theta = np.linspace(0, 2 * np.pi, 30)
        yrefs = [np.array([radius * np.sin(t), radius * np.sin(t) * np.cos(t), 5]) for t in theta]
        
        # Append [0, 0, 0] to each vector
        yrefs = [np.concatenate((yref, [0, 0, 0])) for yref in yrefs]

        return yrefs



