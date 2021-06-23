# -*- coding: utf-8 -*-
"""
This is a new script to select the loggers data in another way than what it's
doen in Data_import_cloud.py

    - Only the loggers stations whose data is present also in 2020 are selected
    - Specifying a starting year, we can define the length of the time series
        we want
    - A dataframe containing loggers station data from the starting year to
        2020 is created
    - The stations which present more than 10% of missing data in the selected
        period are removed from the dataframe

The resulting dataframe will then have as rows 

Following steps:
    - Take only the first 20 (?) stations with the best NA percentage
    - Remove outliers
    - Fill the NAs in these stations

@author: colompa
"""

#Libraries
import os
import pandas as pd
import datetime
import time
import numpy as np

#Functions
def daily_mean(df, date_pos = 0):
    #Output: daily dates as index, daily mean as values 
    df.rename(columns = {df.columns[date_pos]: "date"}, inplace = True)
    df.loc[:,'date'] = pd.to_datetime(df.loc[:,'date'], format = '%d-%m-%Y %H:%M:%S')
    df.index = df.loc[:,'date']
    df.drop('date', axis=1, inplace=True)
    df_daily_avg = df.resample('1D').mean()
    #other method: df.groupby(df.index.date).mean()
    return df_daily_avg

def create_list_files(listfiles, path, extention):
    #Creates a list containing the paths of the file to import
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extention):
                listfiles.append(root + '\\' + file)
    return listfiles

## New function to get loggers data between two specified years
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

# LOGGERS Dataframe creation
## Arrange the paths to the files in the cloud
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\GW'
log_GW_files = []
log_GW_files = create_list_files(log_GW_files, path, '.csv')
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

# ================================================
# Generate the loggers dataframe between two dates
# ================================================
start = time.time()
loggers_0720 = get_loggers_two_dates(log_GW_files, 2007, 2020, start_end)
end = time.time()
print(f'Loggers GW dataframe creation\tElapsed time: {round((end - start)/60)} minutes')
print(end - start)

## Save the dataframe
# path = r'D:\Users\colompa\Documents\KWR_Internship\Data\logger_dataset\logger_GW_20072020.csv'
# loggers_0720.to_csv(path_or_buf = path, sep=",", index = False)

## Select only the columns which have less than 10% of missing values
#10% of 20 years = 6 months

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

# path = r'D:\Users\colompa\Documents\KWR_Internship\Data\logger_dataset\logger_GW_20072020.csv'
# loggers_0720 = pd.read_csv(path, sep = ",")

# log_clean_0020, NAs_0020 = filter_NAs(loggers_0020, 10)
log_clean_0720, NAs_0720 = filter_NAs(loggers_0720, 10)

#Keep only the 20 best stations (best percentages of NAs)

def keep_columns(df, how_many = 20):
    #Keeps only a certain amount ofthe dataframe's columns based on the NA
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

log_clean_0720, max_na = keep_columns(log_clean_0720)

path = r'D:\Users\colompa\Documents\KWR_Internship\Data\logger_dataset\logger_GW_20072020_20col.csv'
log_clean_0720.to_csv(path, sep = ',', index = True)

# Get the positions of the selected GW stations
#from the dataframe columns take the names and then select the rows in log_coord

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

#Export as a .txt file
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\logger_dataset\log_coord_final.txt'
log_coord.to_csv(path_or_buf = path, sep = "\t", index = False)

# # Visualization
# import plotly.express as ple
# from plotly.offline import plot

# fig = ple.line(log_clean_0720.iloc[:,1:6], x = log_clean_0720.iloc[:,0], y = log_clean_0720.columns[1:6])
# plot(fig)













