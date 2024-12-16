# Fill in the respective function to implement the LQR/EKF SLAM controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *
from ekf_slam import EKF_SLAM

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        
        self.counter = 0
        np.random.seed(99)
        
        # PID parameters for longitudinal controller
        self.Kp_long = 300
        self.Ki_long = 10
        self.Kd_long = 30
        
        # Initializing error variables
        self.sum_lat_err = 0
        self.prev_lat_err = 0
        self.sum_long_err = 0
        self.prev_long_err = 0
        self.desired_speed = 10 
        
        self.A = np.array([
            [0, 1, 0, 0],
            [0, -4*self.Ca/(self.m*self.desired_speed), 4*self.Ca/self.m, -2*self.Ca*(self.lf-self.lr)/(self.m*self.desired_speed)],
            [0, 0, 0, 1],
            [0, -2*self.Ca*(self.lf-self.lr)/(self.Iz*self.desired_speed), 2*self.Ca*(self.lf-self.lr)/self.Iz, 
                -2*self.Ca*(self.lf**2+self.lr**2)/(self.Iz*self.desired_speed)]
        ])
        
        self.B = np.array([
            [0],
            [2*self.Ca/self.m],
            [0],
            [2*self.Ca*self.lf/self.Iz]
        ])
        
        
        self.Q = np.diagflat([1, 0.1, 0.1, 0.01])
        self.R = np.array([50]).reshape(1,1)

        # Add additional member variables according to your need here.

    def getStates(self, timestep, use_slam=False):

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Initialize the EKF SLAM estimation
        if self.counter == 0:
            # Load the map
            minX, maxX, minY, maxY = -120., 450., -500., 50.
            map_x = np.linspace(minX, maxX, 7)
            map_y = np.linspace(minY, maxY, 7)
            map_X, map_Y = np.meshgrid(map_x, map_y)
            map_X = map_X.reshape(-1,1)
            map_Y = map_Y.reshape(-1,1)
            self.map = np.hstack((map_X, map_Y)).reshape((-1))
            
            # Parameters for EKF SLAM
            self.n = int(len(self.map)/2)             
            X_est = X + 0.5
            Y_est = Y - 0.5
            psi_est = psi - 0.02
            mu_est = np.zeros(3+2*self.n)
            mu_est[0:3] = np.array([X_est, Y_est, psi_est])
            mu_est[3:] = np.array(self.map)
            init_P = 1*np.eye(3+2*self.n)
            W = np.zeros((3+2*self.n, 3+2*self.n))
            W[0:3, 0:3] = delT**2 * 0.1 * np.eye(3)
            V = 0.1*np.eye(2*self.n)
            V[self.n:, self.n:] = 0.01*np.eye(self.n)
            # V[self.n:] = 0.01
            print(V)
            
            # Create a SLAM
            self.slam = EKF_SLAM(mu_est, init_P, delT, W, V, self.n)
            self.counter += 1
        else:
            mu = np.zeros(3+2*self.n)
            mu[0:3] = np.array([X, 
                                Y, 
                                psi])
            mu[3:] = self.map
            y = self._compute_measurements(X, Y, psi)
            mu_est, _ = self.slam.predict_and_correct(y, self.previous_u)

        self.previous_u = np.array([xdot, ydot, psidot])

        print("True      X, Y, psi:", X, Y, psi)
        print("Estimated X, Y, psi:", mu_est[0], mu_est[1], mu_est[2])
        print("-------------------------------------------------------")
        
        if use_slam == True:
            return delT, mu_est[0], mu_est[1], xdot, ydot, mu_est[2], psidot
        else:
            return delT, X, Y, xdot, ydot, psi, psidot

    def _compute_measurements(self, X, Y, psi):
        x = np.zeros(3+2*self.n)
        x[0:3] = np.array([X, Y, psi])
        x[3:] = self.map
        
        p = x[0:2]
        psi = x[2]
        m = x[3:].reshape((-1,2))

        y = np.zeros(2*self.n)

        for i in range(self.n):
            y[i] = np.linalg.norm(m[i, :] - p)
            y[self.n+i] = wrapToPi(np.arctan2(m[i,1]-p[1], m[i,0]-p[0]) - psi)
            
        y = y + np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V)
        # print(np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V))
        return y

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the newly defined getStates method
        delT, X, Y, xdot, ydot, psi, psidot = self.getStates(timestep, use_slam=True)
        # You must not use true_X, true_Y and true_psi since they are for plotting purpose
        _, true_X, true_Y, _, _, true_psi, _ = self.getStates(timestep, use_slam=False)

        # You are free to reuse or refine your code from P3 in the spaces below.

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        _, node = closestNode(X,Y,trajectory)
        forwardIndex = min(node + 150, len(trajectory) - 1)
        psiDesired = np.arctan2(trajectory[forwardIndex,1]-trajectory[node,1],trajectory[forwardIndex,0]-trajectory[node,0])
        e1 = (Y - trajectory[forwardIndex,1])*np.cos(psiDesired) -(X - trajectory[forwardIndex,0])*np.sin(psiDesired)  
        e1dot = ydot + xdot*wrapToPi(psi - psiDesired)
        e2 = wrapToPi(psi - psiDesired)
        e2dot = psidot
        states = np.array([e1,e1dot,e2,e2dot])
        
        Ad,Bd, _, _, _ = signal.cont2discrete((self.A,self.B, np.eye(4), np.zeros((4,1))), delT)
        S = np.matrix(linalg.solve_discrete_are(Ad, Bd, self.Q, self.R))
        K = -np.matrix(linalg.inv(self.R + (Bd.T @ S @ Bd)) @ (Bd.T @ S @ Ad))
      
        delta = float(K@states)       
        delta = clamp(delta, -np.pi/6, np.pi/6)

        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        """
        desired_speed = 11
        speed_error = desired_speed - xdot
        # PID controler
        self.sum_long_err += speed_error * delT
        long_err_derivative = (speed_error - self.prev_long_err) / delT
        F = (self.Kp_long * speed_error + self.Ki_long * self.sum_long_err + self.Kd_long * long_err_derivative)
        self.prev_long_err = speed_error        # Changing prev to current for next iteration
        F = clamp(F, 0, 15736)
        # Return all states and calculated control inputs (F, delta)
        return true_X, true_Y, xdot, ydot, true_psi, psidot, F, delta
