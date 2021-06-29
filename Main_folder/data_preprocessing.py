# -*- coding: utf-8 -*-
"""
Module containing functions useful for the data-preprocessing

List of the functions contained:
    
    # General tasks #
    - daily_mean/daily_mean_index:
        Computes the daily mean of a sub-daily observations dataframe
    - create_list_files:
        Creates a list containing the paths of the fiels to import
    - get_loggers_two_dates:
        Returns a dataframe containing the LOGGERS stations with a time series
        ending at 'end_year' on the columns
    
    # NA operations #
    - check_NAs:
        Returns 
    - filter_NAs:
        Computes the NA percentage in each df's column. If it is higher than
        the threshold, it removes the column from the dataframe
    - keep_columns:
        Keeps only a specified amount ofthe dataframe's columns based on the NA
        percentage of the columns
        
    # Outlier rejection #
    - check_outliers:
        Returns the number of outliers in each column of a dataframe, by the
        means of the IQR method
    - remove_outliers:
        Removes the outliers in each column of a dataframe, by the means of 
        the IQR method
    
    # Spatial analysis #
    - find_nearest_point:
        Finds the nearest point to a specified point (point) from a set of
        coordinates (tgpoints)

@author: colompa
"""

#%% Libraries
import os
import pandas as pd
import datetime
import numpy as np
import pyproj

#%% General tasks

def daily_mean(df, date_pos = 0):
    #Output: daily dates as index, daily mean as values 
    df.rename(columns = {df.columns[date_pos]: "date"}, inplace = True)
    df.loc[:,'date'] = pd.to_datetime(df.loc[:,'date'], format = '%d-%m-%Y %H:%M:%S')
    df.index = df.loc[:,'date']
    df.drop('date', axis=1, inplace=True)
    df_daily_avg = df.resample('1D').mean()
    #other method: df.groupby(df.index.date).mean()
    return df_daily_avg

def daily_mean_index(df):
    #Output: daily dates as index, daily mean as values 
    df.index = pd.to_datetime(df.index, format = '%d-%m-%Y %H:%M:%S')
    df_daily_avg = df.resample('1D').mean()
    #other method: df.groupby(df.index.date).mean()
    return df_daily_avg

def create_list_files(path, extention):
    #Creates a list containing the paths of the file to import
    listfiles = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extention):
                listfiles.append(root + '\\' + file)
    return listfiles

def get_loggers_two_dates(list_files, start_year, end_year, dates):
    #In the original file
    #First column: date and time
    #Second column: measurement
    #The output presents:
    #   - Rows: daily dates 
    #   - Columns: stations' daily observations
    start = datetime.datetime(start_year, 1, 1)
    end = datetime.datetime(end_year, 11, 11) #the loggers data end at 11/11/2020
    date = pd.date_range(start = start, end = end)
    date = date.strftime('%Y-%m-%d')
    df = pd.DataFrame({"date": date})
    fields = ['Tijd', 'DIVER (m tov NAP)']
    for i, file in enumerate(list_files, start=0):
        if dates.iloc[i,1] == end_year:
            #Only stations which nominally end in 2020 are selected
            load = pd.read_csv(file, sep = ";", usecols=fields)
            daily = daily_mean(load, 0)
            x = daily.index.strftime('%Y-%m-%d')
            y = daily.values
            name = os.path.basename(file)[1:-4]
            df[name] = np.nan
            df.loc[df['date'].isin(x), name] = y[x.isin(df.iloc[:,0])]
            print(f'Iteration n.: {i+1}\tStations added: {len(df.columns)-1}')
    return df

#%% NA operations

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

def keep_columns(df, how_many = 20):
    #Keeps only a specified amount ofthe dataframe's columns based on the NA
    #percentage of the columns
    NAs = pd.DataFrame(index = df.columns, columns = ['NA_perc'])
    for i, column in enumerate(df.columns, start = 0):
        #Percentage of NAs over the period
        NAs.iloc[i,0] = round(df[column].isna().sum()/len(df[column]), 3)*100
    keep = sorted(NAs.iloc[:,0])[:how_many+1] #+1 because the first row is date
    keep_stations = NAs[NAs.iloc[:,0].isin(keep)].index
    df = df.loc[:,df.columns.isin(keep_stations)]
    #Put date as the index and remove the column date
    df.index = df.iloc[:,0]; df.index.names = ['date']
    df.drop(columns = 'date', inplace = True)
    return df, max(keep)

#%% Outlier rejection

def check_outliers(df):
    for column in df.columns:
        Q1 = np.nanpercentile(df[column], 25)
        Q3 = np.nanpercentile(df[column], 75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5*IQR
        lower_limit = Q1 - 1.5*IQR
        print(f'Column: {column}')
        print(f'Number of upper outliers: {sum(df[column] > upper_limit)}')
        print(f'Number of lower outliers: {sum(df[column] < lower_limit)}')
        print(f'Percentage of outliers: {(sum(df[column] > upper_limit) + sum(df[column] < lower_limit))/len(df[column])}')

def remove_outliers(df, fill = np.nan):
    for column in df.columns:
        Q1 = np.nanpercentile(df[column], 25)
        Q3 = np.nanpercentile(df[column], 75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5*IQR
        lower_limit = Q1 - 1.5*IQR
        df.loc[df[column] > upper_limit,column] = fill
        df.loc[df[column] < lower_limit,column] = fill
        
        #add a method to return the "positions" of the outliers,
        #the index basically, and their values
        
    return df

#%% Spatial analysis

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
