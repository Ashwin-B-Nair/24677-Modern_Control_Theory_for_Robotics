    # Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)
        
         # PID parameters for lateral controller
        self.Kp_lateral = 1
        self.Ki_lateral = 0
        self.Kd_lateral = 0
        
        # PID parameters for longitudinal controller
        self.Kp_long = 250
        self.Ki_long = 20
        self.Kd_long = 3
        
        # Initializing error variables
        self.sum_lat_err = 0
        self.prev_lat_err = 0
        self.sum_long_err = 0
        self.prev_long_err = 0
        self.desired_speed = 5 
        
        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        # Add additional member variables according to your need here.
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
        
        self.desired_poles = [-0.001, -100, -5, -0.01]
        # self.desired_poles = [ -0.2, -0.1, -0.001, -0.0001]
        self.K = signal.place_poles(self.A, self.B, self.desired_poles).gain_matrix
        

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        # Finding the closest point on the trajectory
        mindist, closest_idx = closestNode(X, Y, trajectory)
        ahead_idx = min(closest_idx + 50, len(trajectory) - 1)   
        target_x, target_y = trajectory[ahead_idx]                     
        desired_psi = np.arctan2(target_y - Y, target_x - X)     
        
        vec_to_vehicle = np.array([X, Y]) - trajectory[ahead_idx]
        traj_direction = trajectory[ahead_idx] - trajectory [closest_idx]
        traj_heading = np.arctan2(traj_direction[1], traj_direction[0])
        relative_heading = psi - traj_heading
        
        e1 = np.linalg.norm(vec_to_vehicle)*np.sin(desired_psi-traj_heading)
        e1_dot = xdot * np.sin(relative_heading) + ydot * np.cos(relative_heading)
          
        e2 = wrapToPi(desired_psi - psi)
        e2_dot = psidot
        state = np.array([e1, e1_dot, e2, e2_dot])
        delta = -np.dot(self.K, state)[0]
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
        
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
