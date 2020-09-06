import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley 

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    '''


    def __init__(self, gain_constant=0.4, damping_constant=0.05):

        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0
        self.car_position = np.array([48, 0])

    def stanley(self, waypoints, speed):
        '''
        ##### TODO #####
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            speed (float)
        '''
        # derive orientation error as the angle of the first path segment to the car orientation
        # orientation_error = np.arctan((waypoints[1][1] - waypoints[1][0])/ (waypoints[0][1] - waypoints[0][0]))    #(y2-y1)/(x2-x1)
        # # derive cross track error as distance between desired waypoint at spline parameter equal zero or the car position
        # # print(f'Hello : {self.car_position.shape, waypoints.shape}')
        # cross_error = waypoints[0] [0] - self.car_position[0]
        # # cross_error = np.min(np.sum((self.car_position - waypoints[:,:1])**2,axis = 1))
        # # derive stanley control law
        # eps = 1e-6
        # temp = orientation_error + np.arctan((self.gain_constant * cross_error)/ (speed+ eps))
        # # prevent division by zero by adding as small epsilon
        
        # # derive damping term
                 
        # steering_angle = temp - self.damping_constant *(temp - self.previous_steering_angle)
        
        # self.previous_steering_angle = steering_angle
        v = waypoints[:,1] - waypoints[:,0]
        v/=np.linalg.norm(v)
        # print(f'the value of v is {v}')
        u = np.array([0,1])
        angle1 = np.arctan2(v[1],v[0])
        angle2 = np.pi/2
        or_error = angle2 - angle1
        # print(f'angles 1 and 2 {angle1,angle2}')
        # exit()
        
        
        # derive cross track error as distance between desired waypoint at spline parameter equal zero or the car position
        p = waypoints[:,0]
        d = p - self.car_position
        num = -d[0]*v[0] - d[1]*v[1]
        denom = v[0]*v[0] + v[1]*v[1]
        t = num/denom
        p1 = p + t*v
        cross_tack_error = np.sqrt(np.sum((self.car_position - p1)**2))
        epsilon = 1e-6
        term2 = np.arctan((self.gain_constant * cross_tack_error) /(speed+ epsilon))
        temp = or_error + term2
        # derive damping term
        damp_term = temp - self.damping_constant * (temp - self.previous_steering_angle)
        steering_angle = damp_term
        # print(f'Steeering {temp}')
        self.previous_steering_angle = np.clip(steering_angle, -0.4, 0.4) / 0.4
        
        
        
        
        # clip to the maximum stering angle (0.4) and rescale the steering action space
        return np.clip(steering_angle, -0.4, 0.4) / 0.4






