# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import wrapToPi, closestNode, clamp

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
        
        
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        # Add additional member variables according to your need here.

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
        _, closest_idx = closestNode(X, Y, trajectory)
        ahead_idx = min(closest_idx + 50, len(trajectory) - 1)   # Finding the next point in the trajectory
        target_x, target_y = trajectory[ahead_idx]               # Getting the target points      
        desired_psi = np.arctan2(target_y - Y, target_x - X)     # Calculating heading
        lat_err = wrapToPi(desired_psi - psi)                    # Calculating lateral error (steering angle)
        
        # PID controller
        self.sum_lat_err += lat_err * delT
        lat_err_derivative = (lat_err - self.prev_lat_err) / delT
        delta = (self.Kp_lateral * lat_err + self.Ki_lateral * self.sum_lat_err + self.Kd_lateral * lat_err_derivative)
        self.prev_lat_err = lat_err               # Changing prev to current for next iteration
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
