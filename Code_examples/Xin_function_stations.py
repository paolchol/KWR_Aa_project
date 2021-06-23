# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:27:12 2021

@author: colompa
"""

import pandas as pd
import pyproj
import numpy as np

def find_nearest_station(stn_name, para_name, org_md, tg_md, tg_db):
    """
    org: original
    tg: target
    Return:
    a dict containing timeseries and its information
    """

    # define the system
    geod = pyproj.Geod(ellps='WGS84')

    # get a temporary df with lat and lon
    df_tem = org_md[org_md['id_src'] == stn_name][['id', 'id_src', 'Lat', 'Lon']]
    lat0, lon0 = df_tem.Lat, df_tem.Lon
    # get the distance
    lst = []
    for lat1, lon1 in zip(tg_md['Lat'], tg_md['Lon']):
        _, _, distance = geod.inv(lon0, lat0, lon1, lat1)
        lst.append(distance/1e3)
        df_dist = pd.DataFrame(lst, columns=['dist'])
        idx_min = np.argmin(df_dist)

    dict_stn = {
        'distance' : df_dist.iloc[idx_min].values,
        'name' : tg_md.loc[idx_min, 'label'],
        'id_src': tg_md.loc[idx_min, 'id_src'],
        'id': tg_md.loc[idx_min, 'id'],
        'timeseries' : tg_db.iloc[:, tg_db.columns == nam_min]
        }
    return dict_stn


# ============================================================================

import math








#Found the problem: the coordinates in the two files are in a different format
#Fix the format before calling the function
point = log_coord
# def find_nearest_point(point, tgpoints, namelat = 'Lat', namelon = 'Lon', namepoint = 'label'):
#     # This function finds the nearest point to a specified point (point) from
#     # a set of coordinates (tgpoints)
#     #Define the system
#     geod = pyproj.Geod(ellps='WGS84')
#     #Create a dataframe containing the set points
#     df = pd.DataFrame(point)
#     df.insert(len(df.columns), "target_name", 'xxx', True)
#     df.insert(len(df.columns), "target_lat", 0, True)
#     df.insert(len(df.columns), "target_lon", 0, True)
#     df.insert(len(df.columns), "distance", 0, True)
    
#     if len(point.index) > 1:
#         for i in range(len(point.index)):
#             #df.iloc[i,:]
#             see = find_nearest_point(point.iloc[i,:], tgpoints, namelat, namelon, namepoint)
#             print(see)
#     else:
#         lat0 = point[namelat]; lon0 = point[namelon]
#         lst = []
#         for lat1, lon1 in zip(tgpoints[namelat], tgpoints[namelon]):
#             _, _, distance = geod.inv(lon0, lat0, lon1, lat1)
#             lst.append(distance/1e3)
#             df_dist = pd.DataFrame(lst, columns=['dist'])
#             idx_min = np.argmin(df_dist)
        
#         df.loc['target_name'] = tgpoints.loc[idx_min, namepoint]
#         df.loc['target_lat'] = tgpoints.loc[idx_min, namelat]
#         df.loc['target_lon'] = tgpoints.loc[idx_min, namelon]
#         df.loc['distance'] = df_dist.iloc[idx_min].values
#     return df.T



nearest_stations.transpose()

row = nearest_stations.T
row['target_name']

