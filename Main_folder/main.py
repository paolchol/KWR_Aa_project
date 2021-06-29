# -*- coding: utf-8 -*-
"""
Project:
    Development of a deep learning model in order to predict the water table
    depth in the Drenthe province, The Netherlands

Data available:
    - Water table depths
    - Precipitation
    - Evaporation
    - River discharge

Development:
    # Data preprocessing: #
        - Water table depth data is in hourly and sub-hourly configuration,
            it needs to be resampled in a daily configuration
        - The water table depth stations with a good amount of consecutive
            recent data have to be selected and extracted
        - These extracted stations have to be cleaned from outliers and
            completed if missing data is present 
        - The nearest weather station to each of the selected water table depth
            station has to be selected
        ** Result **
            The result of this phase will be a dataframe for each water table 
            depth point. It will contain the water table depth time series, the
            precipitation time series and the evaporation time series.
            
    # SSA: #
        - To improve the model performance, the water table depth has to be
            divided in three components:
            overall trend, yearly periodicity and noise.
            They will be given to the model separately, and then reconstructed
        - The SSA is performed for each station, and the selection of the
            components will be done manually
        ** Result **
            The result of this phase will be a dataframe containing the index
            of the elementary components of each water table depth station
            identifying trend, periodicity and noise
    
    # Model construction: #
        - Separate script
    # Model testing: #
        - Separate script
    # Model utilization: #
        - 


@author: colompa
"""

# %% Libraries

#General libraries
import os
import pandas as pd
import numpy as np
import datetime
import time
import pickle as pkl
import pyproj

#Custom libraries
import data_preprocessing as dp

# %% Data Preprocessing: Water table Depth - Dataframe construction

# LOGGERS Dataframe creation
## Arrange the paths to the files in the cloud
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\GW'
log_GW_files = dp.create_list_files(path, '.csv')
#Extract their names
station_names = []
for file in log_GW_files:
    station_names.append(os.path.basename(file)[1:-4])

## LOGGERS metadata
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\Metadata_loggers_OW_GW.xlsx'
log_meta = pd.read_excel(path)
#In the metadata the coordinates of the stations are associated with the names of the columns
#columns 5 and 6 contains the coordinates of the stations

### Select only the metadata for the stations whom have a file in the folder
log_meta = log_meta.loc[log_meta['LOCATION'].isin(station_names),:]

### Select only the files for which we have metadata
index_station = pd.DataFrame({'names': station_names})
indexes = np.array(index_station['names'].isin(log_meta['LOCATION']))
indexes = np.where(indexes == True)[0]
log_GW_files_new = []; station_names_new = []
for i in range(len(indexes)):
    log_GW_files_new.append(log_GW_files[indexes[i]])
    station_names_new.append(station_names[indexes[i]])
log_GW_files = log_GW_files_new; del log_GW_files_new
station_names = station_names_new; del station_names_new

## Extract starting and ending dates
log_dates = log_meta[['LOCATION','START','EIND']]
log_dates.loc[:,'EIND'].fillna(pd.to_datetime('11-11-2020'), inplace = True)
log_dates.loc[:,'START'] = pd.to_datetime(log_dates.loc[:,'START'])

#Create a dataframe containing only the staring and ending year in numbers
start_y = [0]*len(log_dates.iloc[:,1])
end_y = [0]*len(log_dates.iloc[:,2])
for i in range(len(log_dates.iloc[:,1])):
        start_y[i] = log_dates.iloc[i,1].year
        end_y[i] = log_dates.iloc[i,2].year
start_end = pd.DataFrame({"start_y": start_y, "end_y": end_y})

# Generate the loggers dataframe between two dates
start = time.time()
loggers_0720 = dp.get_loggers_two_dates(log_GW_files, 2007, 2020, start_end)
end = time.time()
print(f'Loggers GW dataframe creation\tElapsed time: {round((end - start)/60)} minutes')
print(end - start)

## Select only the columns which have less than 10% of missing values
#10% of 13 years = 1 year and 4 months
log_clean_0720, NAs_0720 = dp.filter_NAs(loggers_0720, 10)

# Keep only the 20 best stations (best percentages of NAs)
log_clean_0720, max_na = dp.keep_columns(log_clean_0720)

# Get the positions of the selected GW stations
log_coord = log_meta.loc[:,['LOCATION', 'X', 'Y']]
log_coord.rename(columns = {log_coord.columns[0]: 'station'}, inplace = True)

#Transform the coordinates into WGS1984
from pyproj import Proj, transform

inProj = Proj('epsg:28992') #Amersfoort, dont know if it is the correct one
outProj = Proj('epsg:4326')
for i in range(len(log_coord)):
    log_coord.iloc[i,1:] = transform(inProj,outProj,log_coord.iloc[i,1],log_coord.iloc[i,2])
print(log_coord)
#For some reason X and Y are inverted, so here I just change their columns
log_coord.rename(columns = {log_coord.columns[1]: 'Y', log_coord.columns[2]: 'X'}, inplace = True)

#Select only the positions of the selected GW stations
log_coord = log_coord.loc[log_coord['station'].isin(log_clean_0720.columns),:]
log_coord.index = log_coord['station']
log_coord.drop('station', 1, inplace = True)

# %% Data Preprocessing: Water table Depth - Outlier rejection and NA filling

#Outlier removal
log_ready = dp.remove_outliers(log_clean_0720)
dp.check_outliers(log_ready)

#NA filling
log_ready = log_ready.interpolate('linear', limit = 365)

# %% DP of Water Table Depth: save resulting data and metadata

pkl.dump(log_ready, open('loggers_dataframe.p', 'wb'))
pkl.dump(log_coord, open('log_coordinates.p', 'wb'))

# %% Data Preprocessing: Select the nearest Weather Stations to the
#   Water Table Depth stations

# Load the weather stations
path = r'D:\Users\colompa\Documents\Data\metadata.pkl'
with open(path, 'rb') as handle:
            md_weather, _, _, _, _= pkl.load(handle)
path = r'D:\Users\colompa\Documents\Data\data.pkl'
with open(path, 'rb') as handle:
            _, _, df_prec, df_evap, _, _, _, _= pkl.load(handle)

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
dp.check_NAs(df_prec_cut)
dp.check_NAs(df_evap_cut)

df_prec_cut, NA_prec = dp.filter_NAs(df_prec_cut, 10)
df_evap_cut, NA_evap = dp.filter_NAs(df_evap_cut, 10)

# Get the positions of the selected precipitation and evaporation stations
prec_coord = md_weather[md_weather['label'].isin(df_prec_cut.columns)]
prec_coord = prec_coord[~prec_coord.loc[:,'label'].duplicated()]
#Nieuw Beerta has dc_id code equal to 18, and also its time series is odd
#Consider removing it if it results as the closest to one of the GW stations

evap_coord = md_weather[md_weather['label'].isin(df_evap_cut.columns)]
dc_id_condition = evap_coord == 18
evap_coord = evap_coord.loc[dc_id_condition.values,:]

# Select the closest weather station to each GW station
#I have two sets of coordinates: loog_coord and prec_coord. I want to find the
#point from the second set which are the closest to the first set. It is basically a
#Nearest Neighbour

log_coord.rename(columns = {log_coord.columns[1]: 'Lat', log_coord.columns[2]: 'Lon'}, inplace = True)

#Find the nearest precipitation stations
for i in range(len(log_coord.index)):
    if i == 0:
        nearest_stations_p = dp.find_nearest_point(log_coord.iloc[i,:], prec_coord)
    else:
        nearest_stations_p = nearest_stations_p.append(dp.find_nearest_point(log_coord.iloc[i,:], prec_coord))

#Find the nearest evaporation stations
for i in range(len(log_coord.index)):
    if i == 0:
        nearest_stations_e = dp.find_nearest_point(log_coord.iloc[i,:], evap_coord)
    else:
        nearest_stations_e = nearest_stations_e.append(dp.find_nearest_point(log_coord.iloc[i,:], evap_coord))

# Get only the weather stations we need from the full dataframe
df_prec_cut = df_prec_cut.loc[:,df_prec_cut.columns.isin(nearest_stations_p['target_name'])]
df_evap_cut = df_evap_cut.loc[:,df_evap_cut.columns.isin(nearest_stations_e['target_name'])]


