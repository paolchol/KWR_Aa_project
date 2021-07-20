# -*- coding: utf-8 -*-
"""
Weather stations data

Operations to perform:
    - Cut the series between 2007 and 2020
    - Select the ones with less than 10% of NAs in this period
    - Select, for each GW station, the closest precipitation station
    


@author: colompa
"""
import os
import pandas as pd
import datetime
import time
import numpy as np

#Load the datasets
import pickle

path = r'D:\Users\colompa\Documents\Data\metadata.pkl'
with open(path, 'rb') as handle:
            md_weather, _, _, _, _= pickle.load(handle)

path = r'D:\Users\colompa\Documents\Data\data.pkl'
with open(path, 'rb') as handle:
            _, _, df_prec, df_evap, _, _, _, _= pickle.load(handle)

#Just to be sure to have them next time
# path = r'D:\Users\colompa\Documents\Data\metadata.csv'
# md_weather.to_csv(path, sep = ',', index = False)
# path = r'D:\Users\colompa\Documents\Data\df_prec.csv'
# df_prec.to_csv(path, sep = ',', index = True)
# path = r'D:\Users\colompa\Documents\Data\df_evap.csv'
# df_evap.to_csv(path, sep = ',', index = True)


# Cut the datasets between 2007 and 2020
## Cut the precipitation dataset
new_index = []
for timestamp in df_prec.index:
    new_index.append(timestamp.date())

df_prec.index = new_index
start = datetime.datetime(2007, 1, 1)
end = datetime.datetime(2020, 11, 11) #the loggers data end at 11/11/2020
date = pd.date_range(start = start, end = end)
df_prec_cut = df_prec[date[0] : date[len(date)-1]]

## Cut the evaporation dataset
new_index = []
for timestamp in df_evap.index:
    new_index.append(timestamp.date())

df_prec.index = new_index
start = datetime.datetime(2007, 1, 1)
end = datetime.datetime(2020, 11, 11) #the loggers data end at 11/11/2020
date = pd.date_range(start = start, end = end)
df_evap_cut = df_evap[date[0] : date[len(date)-1]]

# Check the NA percentages in the datasets and remove the stations with more
# than 10% of missing data

def check_NAs(df):
    count = 0
    for column in df.columns:
        NA_perc = round(df[column].isna().sum()/len(df[column]),3)*100
        print(f'The column "{column}" has this percentage of NAs:\n{NA_perc}')
        if NA_perc > 10:
            count = count + 1
    print(f'There are {count} columns which have more than 10% of missing values')

def filter_NAs(df, NA_threshold):
    #Compute the NA percentage in each df's column
    #If it is higher than the threshold, it removes the column from the dataframe
    #NA_threshold has to be put in terms of percentage (e.g. 10)
    #Returns:
    #   - df without columns with NAs percentage higher than  NA_threshold
    #   - (dataframe containing the NA percentage for each column and if it has been removed or not)
    NAs = pd.DataFrame(index = df.columns, columns = ['NA_perc', 'Removed'])
    for i, column in enumerate(df.columns, start = 0):
        #Percentage of NAs over the period
        NAs.iloc[i,0] = round(df[column].isna().sum()/len(df[column]), 3)*100        
        if NAs.iloc[i,0] > NA_threshold:
            df.drop(columns = column, inplace = True)
            NAs.iloc[i,1] = True    
        else:
            NAs.iloc[i,1] = False
    return df, NAs

check_NAs(df_prec_cut)
check_NAs(df_evap_cut)

df_prec_cut, NA_prec = filter_NAs(df_prec_cut, 10)
df_evap_cut, NA_evap = filter_NAs(df_evap_cut, 10)

# Get the positions of the selected precipitation and evaporation stations
prec_coord = md_weather[md_weather['label'].isin(df_prec_cut.columns)]
prec_coord = prec_coord[~prec_coord.loc[:,'label'].duplicated()]
#Nieuw Beerta has dc_id code equal to 18, and also its time series is odd
#Consider removing it if it results as the closest to one of the GW stations

evap_coord = md_weather[md_weather['label'].isin(df_evap_cut.columns)]
dc_id_condition = evap_coord == 18
evap_coord = evap_coord.loc[dc_id_condition.values,:]

# Save them as .txt files
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\after_data_prep\prec_coord.txt'
prec_coord.to_csv(path_or_buf = path, sep = "\t", index = False)
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\after_data_prep\evap_coord.txt'
evap_coord.to_csv(path_or_buf = path, sep = "\t", index = False)

# Select the closest weather station to each GW station
#I have two sets of coordinates: loog_coord and prec_coord. I want to find the
#point from the second set which are the closest to the first set. It is basically a
#Nearest Neighbour

path = r'D:\Users\colompa\Documents\KWR_Internship\Data\after_data_prep\log_coord_final.txt'
log_coord = pd.read_table(path)
log_coord.rename(columns = {log_coord.columns[1]: 'Lat', log_coord.columns[2]: 'Lon'}, inplace = True)
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\after_data_prep\prec_coord.txt'
prec_coord = pd.read_table(path)
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\after_data_prep\evap_coord.txt'
evap_coord = pd.read_table(path)

import pyproj

def find_nearest_point(point, tgpoints, namelat = 'Lat', namelon = 'Lon', namepoint = 'label'):
    # This function finds the nearest point to a specified point (point) from
    # a set of coordinates (tgpoints)
    #Define the system
    geod = pyproj.Geod(ellps='WGS84')
    #Create a dataframe containing the set points
    df = pd.DataFrame(point)
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

#Find the nearest precipitation stations
for i in range(len(log_coord.index)):
    if i == 0:
        nearest_stations_p = find_nearest_point(log_coord.iloc[i,:], prec_coord)
    else:
        nearest_stations_p = nearest_stations_p.append(find_nearest_point(log_coord.iloc[i,:], prec_coord))

#Find the nearest evaporation stations
for i in range(len(log_coord.index)):
    if i == 0:
        nearest_stations_e = find_nearest_point(log_coord.iloc[i,:], evap_coord)
    else:
        nearest_stations_e = nearest_stations_e.append(find_nearest_point(log_coord.iloc[i,:], evap_coord))


# Get only the wether stations we need from the full dataframe
df_prec_cut = df_prec_cut.loc[:,df_prec_cut.columns.isin(nearest_stations_p['target_name'])]
df_evap_cut = df_evap_cut.loc[:,df_evap_cut.columns.isin(nearest_stations_e['target_name'])]

# Save just to be sure
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\after_data_prep\df_prec_cut.csv'
df_prec_cut.to_csv(path, sep = ',', index = True)
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\after_data_prep\df_evap_cut.csv'
df_evap_cut.to_csv(path, sep = ',', index = True)



