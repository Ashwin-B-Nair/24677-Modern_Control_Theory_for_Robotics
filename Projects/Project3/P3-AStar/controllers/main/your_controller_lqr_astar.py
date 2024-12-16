# Fill in the respective functions to implement the LQR optimal controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

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

        # Add additional member variables according to your need here.
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
        
        
        self.Q = np.diagflat([1, 5, 10/np.pi, 5])
        self.R = np.array([1/(np.pi/6)]).reshape(1,1)
        
        
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot, obstacleX, obstacleY = super().getStates(timestep)

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        _, node = closestNode(X,Y,trajectory)
        forwardIndex = min(node + 120, len(trajectory) - 1)
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
        speed_error = self.desired_speed - xdot
        # PID controler
        self.sum_long_err += speed_error * delT
        long_err_derivative = (speed_error - self.prev_long_err) / delT
        F = (self.Kp_long * speed_error + self.Ki_long * self.sum_long_err + self.Kd_long * long_err_derivative)
        self.prev_long_err = speed_error        # Changing prev to current for next iteration
        F = clamp(F, 0, 15736)

        # Return all states and calculated control inputs (F, delta) and obstacle position
        return X, Y, xdot, ydot, psi, psidot, F, delta, obstacleX, obstacleY
