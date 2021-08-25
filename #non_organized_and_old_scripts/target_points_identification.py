# -*- coding: utf-8 -*-
"""
Define target points


Do not continue on this path


Using the list of stations which satisfy the time threshold, define a set of points
in which the model will predict the water table depth.

The points will need to be unique, and in a representative distribution.
To identify the target points, it is needed to:
    - Define the minimum number of stations that will have to "cover" each point
    - Define the maximum distance between the point and the station
    - Define the minimum distance between the target points

@author: colompa
"""

def find_target_points(stations, min_stations, max_distance_st, min_distance_tp):
    #Create a buffer around each station, using max_distance_st
    #Overlay the buffers
    #Count the overlaying buffers
    #Select areas where the count is >= min_stations
    #In these areas, create points which are ditant at least min_distance_tp
    
    print(0)

