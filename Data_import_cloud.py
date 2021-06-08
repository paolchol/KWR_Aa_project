☺# -*- coding: utf-8 -*-
"""
Data import from KWR's cloud

- LOGGERS: groundwater heads and surface water levels from loggers
- TELEMETRIE: surface water level from automatic loggers
- D2extraction: smaller extractions by farmers, building locations, ecc.

GW: groundwater
OW: surface water

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
    df.loc[:,'date'] = pd.to_datetime(df.loc[:,'date'])
    df.loc[:,'date'] = df.loc[:,'date'].apply( lambda df: datetime.datetime(year=df.year, month=df.month, day=df.day))
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

def create_logger_dataframe(list_files, df, threshold, NAs):
    #In the original file
    #First column: date and time
    #Second column: measurement
    #The output presents:
    #   - Rows: daily dates 
    #   - Columns: stations' daily observations
    fields = ['Tijd', 'DIVER (m tov NAP)']
    for i, file in enumerate(list_files, start=0):
        if (NAs.iloc[i,0]/100)*NAs.iloc[i,1] >= threshold:
            #The stations are selected based on the length of their
            #observations' time series
            load = pd.read_csv(file, sep = ";", usecols=fields)
            daily = daily_mean(load, 0)
            x = daily.index.strftime('%Y-%m-%d')
            y = daily.values
            name = os.path.basename(file)[1:-4]
            df[name] = np.nan
            df.loc[df['date'].isin(x), name] = y[x.isin(df.iloc[:,0])]
            print(f'Iteration n.: {i+1}\tStations added: {len(df. columns)-1}')
    return df

def count_NAs(list_files):
    #In the original files are not present NAs, when a data is not
    #available the corresponding row is missing. So, before counting the missing
    #data, they have to be grouped by day. This is done by the function daily_mean
    #After creating the daily aggregation of the data, the missing values are counted
    #The output presents:
    #   - Index: name of the file/station
    #   - NA_daily_prec: percentage of NAs in the station's daily time series
    #   - length: length of the daily time series
    fields = ['Tijd', 'DIVER (m tov NAP)']
    file_index = []
    for file in list_files:
        file_index.append(os.path.basename(file)[1:-4])
    NAs = pd.DataFrame(index = file_index, columns = ['NA_daily_perc','lenght'])
    for i, file in enumerate(list_files, start = 0):
        print(i, file)
        load = pd.read_csv(file, sep = ";", usecols=fields)
        daily = daily_mean(load, 0)
        NAs.iloc[i,0] = round(daily.isna().sum().iloc[0]/len(daily), 3)*100
        NAs.iloc[i,1] = len(daily)
    return NAs


# LOGGERS Dataframe creation
## Arrange the paths to the files in the cloud
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\GW'
log_GW_files = []
log_GW_files = create_list_files(log_GW_files, path, '.csv')
#Extract their names
station_names = []
for file in log_GW_files:
    station_names.append(os.path.basename(file)[1:-4])

## Count the NAs in each station's time series
# start = time.time()
# NA_count = count_NAs(log_GW_files)
# end = time.time()
# print('Elapsed time  - NAs count')
# print(end - start)
# #It took 3000 seconds
# #Save as a .txt
# path = r'D:\Users\colompa\Documents\KWR_Internship\Data\NAs_count.txt'
# NA_count.to_csv(path_or_buf = path, sep = "\t", index = True)

#Load the NAs' count saved before
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\NAs_count.txt'
NA_count_up = pd.read_table(path)
NA_count_up.index = NA_count_up.iloc[:,0]; NA_count_up.index.names = ['Station']
NA_count_up.drop('Unnamed: 0', axis=1, inplace=True)

years_data = round(NA_count_up.iloc[:,0]/100*NA_count_up.iloc[:,1]/365)
      
## LOGGERS metadata
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\Metadata_loggers_OW_GW.xlsx'
log_meta = pd.read_excel(path)
#In the metadata the coordinates of the stations are associated with the names of the columns
#columns 5 and 6 contains the coordinates of the stations

### Select only the metadata for the stations whom have a file in the folder
log_meta = log_meta.loc[log_meta['LOCATION'].isin(station_names),:]

### Select only the files for which we have metadata
indexes = np.array(NA_count_up.index.isin(log_meta['LOCATION']))
indexes = np.where(indexes == True)[0]
log_GW_files_new = []; station_names_new = []
for i in range(len(indexes)):
    log_GW_files_new.append(log_GW_files[indexes[i]])
    station_names_new.append(station_names[indexes[i]])
NA_count_up = NA_count_up[NA_count_up.index.isin(log_meta['LOCATION'])]
log_GW_files = log_GW_files_new; del log_GW_files_new
station_names = station_names_new; del station_names_new

## Define the time threshold
#Extract starting and ending dates
log_dates = log_meta[['LOCATION','START','EIND']]
log_dates.loc[:,'EIND'].fillna(pd.to_datetime('11-11-2020'), inplace = True)
log_dates.loc[:,'START'] = pd.to_datetime(log_dates.loc[:,'START'])

round(max(log_dates.loc[:,'EIND']-log_dates.loc[:,'START']).days/365) #42 years (not cleaned)
s = round(len(log_dates)*10/100) #number of stations to extract to calculate the threshold
delta = log_dates.loc[:,'EIND']-log_dates.loc[:,'START']
#"Clean" the deltas by considering the missing data percentage
delta_clean = delta.dt.days*(NA_count_up.iloc[:,0].values/100)
mean_delta = sum(sorted(delta_clean)[-s:])/s
# threshold = round(mean_delta) #7882 days
threshold = 20*365 #20 years threshold
threshold = 13*365 #14 years threshold
print(f'Threshold: {threshold} days, {round(threshold/365)} \
years\nNumber of stations which satisfy the threshold: \
{len(delta_clean[delta_clean >= threshold])}')
#Threshold: 7882 days, 22 years
#Number of stations which satisfy the threshold: 8

#We can decrease the threshold in order to have more stations, for example:
#Threshold: 7300 days, 20 years
#Number of stations which satisfy the threshold: 12

## Dataframe construction
### Date range setting
#1980-2020 to start, then we can change the range
end_year = 2020
start = datetime.datetime(end_year-int(threshold/365), 1, 1)
end = datetime.datetime(2020, 11, 11) #the loggers data end at 11/11/2020
date = pd.date_range(start = start, end = end)
date = date.strftime('%Y-%m-%d')

### Actual dataframe construction
loggers_GW = pd.DataFrame({"date": date})
start = time.time()
loggers_GW = create_logger_dataframe(log_GW_files, loggers_GW, threshold, NA_count_up)
end = time.time()
print('Elapsed time  - Loggers GW dataframe creation')
print(end - start)
#In create_loggers_dataframe, the threshold on the minimum series length is used
#to select only station who have at least that number of measurements.
#It took 38 seconds to create a dataframe with a 22 years threshold
#It took 79 seconds to create a dataframe with a 20 years threshold

#### Save the dataframe as a csv
#And also try to save it as a pickle
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\logger_GW_13y.csv'
loggers_GW.to_csv(path_or_buf = path, sep=",", index = False)

## Extract the coordinates and visualize the stations
log_coord = log_meta.loc[:,['LOCATION', 'X', 'Y']]
log_coord.rename(columns = {log_coord.columns[0]: 'station'}, inplace = True)
#The name of the station is located in the LOCATION column of log_coord

#Transform the coordinates into WGS1984
from pyproj import Proj, transform

inProj = Proj('epsg:28992') #Amersfoort, dont know if it is the correct one
outProj = Proj('epsg:4326')
for i in range(len(log_coord)):
    log_coord.iloc[i,1:] = transform(inProj,outProj,log_coord.iloc[i,1],log_coord.iloc[i,2])
print(log_coord)
log_coord.rename(columns = {log_coord.columns[1]: 'Y', log_coord.columns[2]: 'X'}, inplace = True)

### Extract the stations that satisfy the threshold
log_coord_thre = log_coord[delta_clean >= threshold]

# Add a column to both log_coord with the actual length of the time series
#(NA*length, in years)

#Export as .txt files
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\log_coord.txt'
log_coord.to_csv(path_or_buf = path, sep = "\t", index = False)
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\log_coord_thre_13y.txt'
log_coord_thre.to_csv(path_or_buf = path, sep = "\t", index = False)


# Extraction data
## Arrange the information in the file
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\D2extraction\Singlepoint\ExtractionsSinglePointWithCoords.csv'
extraction_file = pd.read_csv(path, sep = ";")

#Extract values and positions
columns = ['Sublocation_ID', 'EXTR_m3y_2010', 'EXTR_m3y_2011', 'EXTR_m3y_2012', 'EXTR_m3y_2013', 'EXTR_m3y_2014', 'EXTR_m3y_2015', 'EXTR_m3y_2016', 'EXTR_m3y_2017', 'EXTR_m3y_2018', 'EXTR_m3y_2019', 'EXTR_m3y_2020']
extraction_values = extraction_file.loc[:,columns]
columns = ['Sublocation_ID', 'X', 'Y']
extraction_positions = extraction_file.loc[:,columns]

#Put date in rows and station in columns
extraction_values = extraction_values.transpose()
extraction_values.columns = extraction_values.iloc[0,:]
extraction_values.drop(index = 'Sublocation_ID', inplace = True)
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2021, 1, 1) #the loggers data end at 11/11/2020
extraction_values.index =  pd.date_range(start = start, end = end, freq = 'Y')

## Obtain daily data by dividing the yearly data by the length of that year

def daily_extraction(station, name = 'data'):
    import calendar as cl
    station = pd.to_numeric(station, errors='coerce')
    for i in range(len(station)):
        num_days = 365
        if cl.isleap(station.index[i].year):
            num_days = 366
        start = datetime.datetime(station.index[i].year, 1, 1)
        date = pd.date_range(start = start, end = station.index[i])
        new_year = pd.DataFrame({"date": date})
        if station.iloc[i] is None:
            new_year[name] = nan
        else:
            daily = station.iloc[i]/num_days
            new_year[name] = daily
        if i == 0: df = new_year
        else: df = df.append(new_year)
    return df

for i, station in enumerate(extraction_values, start = 0):
    if i > 0:
        dd = daily_extraction(extraction_values.iloc[:,i], str(extraction_values.columns[i]))
        extr_daily[station] = dd.iloc[:,1]
    else:
        extr_daily = daily_extraction(extraction_values.iloc[:,i], str(extraction_values.columns[i]))

## Save the dataframe
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\extr_daily.csv'
extr_daily.to_csv(path_or_buf = path, sep = "\t", index = False)

## Select the extraction points near the target points


