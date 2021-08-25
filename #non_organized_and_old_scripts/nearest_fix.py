# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 17:14:01 2021

@author: colompa
"""

def find_nearest_point(point, tgpoints, namelat = 'Lat', namelon = 'Lon', namepoint = 'label'):
    # This function finds the nearest point to a specified point (point) from
    # a set of coordinates (tgpoints)
    #Define the system
    geod = pyproj.Geod(ellps='WGS84')
    #Create a dataframe containing the set points
    df = pd.DataFrame({'station': point.index, 'Lat': point[namelat], 'Lon': point[namelon]})
    lat0 = point[namelat]; lon0 = point[namelon]
    lst = []
    for lat1, lon1 in zip(tgpoints[namelat], tgpoints[namelon]):
        _, _, distance = geod.inv(lon0, lat0, lon1, lat1)
        lst.append(distance/1e3)
    df_dist = pd.DataFrame(lst, columns=['dist'])
    idx_min = np.argmin(df_dist)
        
    df.loc['target_name'] = tgpoints.loc[idx_min, namepoint]
    df.loc['target_lat'] = tgpoints.loc[idx_min, namelat]
    df.loc['target_lon'] = tgpoints.loc[idx_min, namelon]
    df.loc['distance'] = df_dist.iloc[idx_min].values
    return df.T


path = r'D:\Users\colompa\Documents\KWR_Internship\Data\logger_dataset\log_coord_gg.txt'
log_coord.to_csv(path_or_buf = path, sep = "\t", index = True)