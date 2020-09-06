import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


def normalize(v):
    norm = np.linalg.norm(v,axis=0) + 0.00001
    return v / norm.reshape(1, v.shape[1])

def curvature(waypoints):
    '''
    ##### TODO #####
    Curvature as  the sum of the normalized dot product between the way elements
    Implement second term of the smoothin objective.

    args: 
        waypoints [2, num_waypoints] !!!!!
    '''
    diff = np.diff(waypoints)
    norm_diff = normalize(diff)
    
    curvature = 0
    for i in range(diff.shape[1]-1):
        # print(f'hey there {norm_diff[:,i],norm_diff[:,i+1]}')
        curvature += np.dot(norm_diff[:,i],norm_diff[:,i+1])
    
    return curvature


def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    '''
    Objective for path smoothing

    args:
        waypoints [2 * num_waypoints] !!!!!
        waypoints_center [2 * num_waypoints] !!!!!
        weight_curvature (default=40)
    '''
    # mean least square error between waypoint and way point center
    ls_tocenter = np.mean((waypoints_center - waypoints)**2)

    # derive curvature
    curv = curvature(waypoints.reshape(2,-1))

    return -1 * weight_curvature * curv + ls_tocenter


def waypoint_prediction(roadside1_spline, roadside2_spline, num_waypoints=6, way_type = "smooth"):
    '''
    ##### TODO #####
    Predict waypoint via two different methods:
    - center
    - smooth 

    args:
        roadside1_spline
        roadside2_spline
        num_waypoints (default=6)
        parameter_bound_waypoints (default=1)
        waytype (default="smoothed")
    '''
    if way_type == "center":
        ##### TODO #####
     
        # create spline arguments

        # derive roadside points from spline

        # derive center between corresponding roadside points

        # output way_points with shape(2 x Num_waypoints)
        
        t = np.linspace(0, 1, num_waypoints)
        lane_boundary1_points_points = np.array(splev(t, roadside1_spline))
        lane_boundary2_points_points = np.array(splev(t, roadside2_spline))
        
        
        tempx = 0.5*(lane_boundary1_points_points[0] + lane_boundary2_points_points[0])
        tempy = 0.5*(lane_boundary1_points_points[1] + lane_boundary2_points_points[1])
        
        way_point_center = np.vstack((tempx,tempy))
        # print(way_point_center.shape)
        return way_points
    
    elif way_type == "smooth":
        ##### TODO #####

        # create spline arguments

        # derive roadside points from spline

        # derive center between corresponding roadside points
        
                

        t = np.linspace(0, 1, num_waypoints)
        lane_boundary1_points_points = np.array(splev(t, roadside1_spline))
        lane_boundary2_points_points = np.array(splev(t, roadside2_spline))
        
        tempx = 0.5*(lane_boundary1_points_points[0] + lane_boundary2_points_points[0])
        tempy = 0.5*(lane_boundary1_points_points[1] + lane_boundary2_points_points[1])
        
        way_points_center = np.hstack((tempx,tempy))
        
        # print(tempx.shape,tempx)
        # print(tempy.shape,tempy)
        # print(f'waypoints_center shape{way_points_center.shape,way_points_center}')
        # optimization
        way_points = minimize(smoothing_objective, 
                      (way_points_center), 
                      args=way_points_center)["x"]

        return way_points.reshape(2,-1)


def target_speed_prediction(waypoints, num_waypoints_used=5,
                            max_speed=60, exp_constant=4.5, offset_speed=30):
    '''
    ##### TODO #####
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)
    
    output:
        target_speed (float)
    '''
    curv = curvature(waypoints)
    temp = np.exp(-exp_constant * abs(num_waypoints_used - 2 - curv)) 
    target_speed = (max_speed - offset_speed) * temp + offset_speed
    # print(f'Hello : {target_speed}')
    return target_speed