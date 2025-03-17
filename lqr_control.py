import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from shapely.geometry import Polygon
import geopandas as gpd
from scipy.optimize import linprog
from utils import proj_input
import scipy as sp


class LQRControl:

    def __init__(self,A,B,K,lb_x,ub_x,lb_u,ub_u):
        self.A = A
        self.B = B
        self.K = K
        self.lb_x = lb_x
        self.ub_x = ub_x
        self.lb_u = lb_u
        self.ub_u = ub_u
        


    def remove_redundant_constraints(self,A, b, x0=None, tol=None):
        """
        Removes redundant constraints for the polyhedron Ax <= b.

        """
        # A = np.asarray(A)
        # b = np.asarray(b).flatten()
        
        if A.shape[0] != b.shape[0]:
            raise ValueError("A and b must have the same number of rows!")
        
        if tol is None:
            tol = 1e-8 * max(1, np.linalg.norm(b) / len(b))
        elif tol <= 0:
            raise ValueError("tol must be strictly positive!")
        
        # Remove zero rows in A
        Anorms = np.max(np.abs(A), axis=1)
        badrows = (Anorms == 0)
        if np.any(b[badrows] < 0):
            raise ValueError("A has infeasible trivial rows.")
            
        A = A[~badrows, :]
        b = b[~badrows]
        goodrows = np.concatenate(([0], np.where(~badrows)[0]))
            
        # Find an interior point if not supplied
        if x0 is None:
            if np.all(b > 0):
                x0 = np.zeros(A.shape[1])
            else:
                raise ValueError("Must supply an interior point!")
        else:
            x0 = np.asarray(x0).flatten()
            if x0.shape[0] != A.shape[1]:
                raise ValueError("x0 must have as many entries as A has columns.")
            if np.any(A @ x0 >= b - tol):
                raise ValueError("x0 is not in the strict interior of Ax <= b!")
                
        # Compute convex hull after projection
        btilde = b - A @ x0
        if np.any(btilde <= 0):
            print("Warning: Shifted b is not strictly positive. Convex hull may fail.")
        
        Atilde = np.vstack((np.zeros((1, A.shape[1])), A / btilde[:, np.newaxis]))
        
        hull = ConvexHull(Atilde)    
        u = np.unique(hull.vertices)    
        nr = goodrows[u]    
        h = goodrows[hull.simplices]
        
        # if nr[0] == 0:
        #     nr = nr[1:]
            
        Anr = A[nr, :]
        bnr = b[nr]
            
        return nr, Anr, bnr, h, x0


    def plot_polygon(self,A, b, color='blue', ax=None):
        '''
        Visualize the polytope defined by A x <= b with a specified color.
        '''
        # Create the halfspaces (A x <= b is equivalent to Ax - b <= 0)
        halfspaces = np.hstack((A, -b[:, np.newaxis]))

        # Find a feasible interior point (can be replaced by a valid method if needed)
        feasible_point = np.zeros(A.shape[1])

        # Compute intersections and the polygon
        hs = HalfspaceIntersection(halfspaces, feasible_point)
        polygon = Polygon(hs.intersections).convex_hull

        # Plot the polygon with the specified color
        polygon_gpd = gpd.GeoSeries(polygon)
        polygon_gpd.plot(alpha=0.3, color=color, ax=ax)

        # Plot the boundary of the polygon with the specified color
        ax.plot(*polygon.exterior.xy, color=color, marker='o', markersize=5)

        ax.axis('equal')
        ax.grid()

    def box_constraints(self,lb, ub):
        num_con = 2 * len(lb)
        A = np.kron(np.eye(len(lb)), [[1], [-1]])

        b = np.zeros(num_con)
        for i in range(num_con):
            b[i] = ub[i // 2] if i % 2 == 0 else -lb[i // 2]

        goodrows = np.logical_and(~np.isinf(b), ~np.isnan(b))
        A = A[goodrows]
        b = b[goodrows]
        
        return A, b


    def compute_maximal_admissible_set(self,F, A, b, max_iter=100):
        dim_con = A.shape[0]
        A_inf_hist = []
        b_inf_hist = []

        Ft = F
        A_inf = A
        b_inf = b
        A_inf_hist.append(A_inf)
        b_inf_hist.append(b_inf)

        for t in range(max_iter):
            f_obj = A @ Ft
            stop_flag = True
            for i in range(dim_con):
                x = linprog(-f_obj[i], A_ub=A_inf, b_ub=b_inf, method="highs")["x"]
                # x = solve_qp(np.zeros((2, 2)), -f_obj[i], A_inf, b_inf, solver="") # Actually, this is not a QP, but a LP. It is better to use a LP solver.
                if f_obj[i] @ x > b[i]:
                    stop_flag = False
                    break

            if stop_flag:
                break
            
            A_inf = np.vstack((A_inf, A @ Ft))
            b_inf = np.hstack((b_inf, b))
            Ft = F @ Ft
            A_inf_hist.append(A_inf)
            b_inf_hist.append(b_inf)

        return A_inf_hist, b_inf_hist



    def find_lqr_invariant_set(self,A, B, K, lb_x, ub_x, lb_u, ub_u):
        A_x, b_x = self.box_constraints(lb_x, ub_x)
        A_u, b_u = self.box_constraints(lb_u, ub_u)

        A_lqr = A_u @ K
        b_lqr = b_u

        A_con = np.vstack((A_lqr, A_x))
        b_con = np.hstack((b_lqr, b_x))

        F = A + B @ K
        
        A_inf_hist, b_inf_hist = self.compute_maximal_admissible_set(F, A_con, b_con)

        return A_inf_hist, b_inf_hist



    def computeX1(self,G, H, psi, Ad, Bd, P, gamma): # TODO: Add support for the point constraint 
        '''
        Computes the feasible set X_1 for the system x^+ = Ax + Bu subject to constraints Gx + Hu <= psi and x^+ \in Xf.
        '''
        dim_u = Bd.shape[1]
        G_ = np.vstack((G, P @ Ad))
        H_ = np.vstack((H, P @ Bd))
        psi_ = np.hstack((psi, -gamma))
        
        psi_ = np.expand_dims(psi_, axis=1)
        
        A, b = proj_input(G_, H_, psi_, 1, dim_u)
        b = -b.squeeze()
        
        return A, b

    def computeXn(self,A, B, K, N, lb_x, ub_x, lb_u, ub_u):
        '''
        
        '''
        A_x, b_x = self.box_constraints(lb_x, ub_x)
        A_u, b_u = self.box_constraints(lb_u, ub_u)

        A_lqr = A_u @ K
        b_lqr = b_u

        A_con = np.vstack((A_lqr, A_x))
        b_con = np.hstack((b_lqr, b_x))

        F = A + B @ K
        
        A_inf_hist, b_inf_hist = self.compute_maximal_admissible_set(F, A_con, b_con)
        _, A_inf, b_inf, _, _ = self.remove_redundant_constraints(A_inf_hist[-1], b_inf_hist[-1])
    
        GH = sp.linalg.block_diag(A_x, A_u)
        G = GH[:, :self.dim_x]
        H = GH[:, self.dim_x:]
        psi = -np.hstack((b_x, b_u))
    
        # Xns = [(A_inf_hist[-1], b_inf_hist[-1])]   
        Xns = [(A_inf, b_inf)] 
        
        for _ in range(N):
            P, gamma = Xns[-1]
            P, gamma = self.computeX1(G, H, psi, A, B, P, gamma)        
            _, P, gamma, _, _ = self.remove_redundant_constraints(P, gamma)
            Xns.append((P, gamma))

        return Xns




