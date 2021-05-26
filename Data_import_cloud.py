# -*- coding: utf-8 -*-
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

def create_logger_dataframe(list_files, df):
    for file in list_files:
        load = pd.read_csv(file, sep = ';', header = 0)
        load = load.iloc[:,:2]
        daily = daily_mean(load, 0)
        x = daily.index.strftime('%Y-%m-%d')
        y = daily.values
        name = os.path.basename(file)[1:-4]   
        df[name] = np.nan
        df.loc[df['date'].isin(x), name] = y[x.isin(df.iloc[:,0])]
    return df

# Arrange the paths to all the files in the cloud
## Files from LOGGERS
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\GW'
log_GW_files = []
log_GW_files = create_list_files(log_GW_files, path, '.csv')

path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\OW'
log_OW_files = []
log_OW_files = create_list_files(log_OW_files, path, '.csv')

## Files from TELEMETRIE
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\TELEMETRIE\TELEMETRIE'
tel_OW_files = []
tel_OW_files = create_list_files(tel_OW_files, path, '.csv')

## File from D2extraction
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\D2extraction\Singlepoint'
extr_file = []
extr_file = create_list_files(extr_file, path, '.csv')

# Dataframes creation
## Date range setting
#   1980-2020 to start, then we can change the range
start = datetime.datetime(1980, 1, 1)
end = datetime.datetime(2020, 12, 31)

date = pd.date_range(start = start, end = end)
date = date.strftime('%Y-%m-%d')

## LOGGERS Dataframe
loggers_GW = pd.DataFrame({"date": date})
loggers_OW = pd.DataFrame({"date": date})

start = time.time()
loggers_GW = create_logger_dataframe(log_GW_files, loggers_GW)
end = time.time()
print('Elapsed time  - For-loop Loggers GW creation')
print(end - start)

loggers_OW = create_logger_dataframe(log_OW_files, loggers_OW)

### LOGGERS metadata
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\Metadata_loggers_OW_GW.xlsx'
log_meta = pd.read_excel(path)
# In the metadata the coordinates of the stations are associated with the names of the columns
# Columns 5 and 6 contains the coordinates of the stations



## TELEMETRIE Dataframe
telem_OW = pd.DataFrame({"date": date})

start = time.time()
for file in tel_OW_files:
    load = pd.read_csv(file, sep = ';', header = 0)
    #Remove the first row and transform the data columns into float
    load = load.iloc[1:,:]
    for i in range(1,len(load.columns)):
        load.iloc[:,i] = pd.to_numeric(load.iloc[1:,i].str.replace(',', '.'))
    
    #Take the daily mean of the data for each column in input
    daily_data = daily_mean(load, 0)
    daily_data.index
    for station in daily_data.columns:
        if station in telem_OW.columns:
            #Add the data in the same column based on the date
            x = daily_data.index.strftime('%Y-%m-%d') #Extract the date and change date format
            y = daily_data.loc[:,station].values   #Extract the data
            
            telem_OW.loc[telem_OW['date'].isin(x), station] = y[x.isin(telem_OW.iloc[:,0])]
        else:
            #Create a new column and add the data based on the date
            x = daily_data.index.strftime('%Y-%m-%d')
            y = daily_data.loc[:,station].values
            
            telem_OW[station] = np.nan   #Create a new column with the name of the station
            telem_OW.loc[telem_OW['date'].isin(x), station] = y[x.isin(telem_OW.iloc[:,0])]

end = time.time()
print('Elapsed time  - For-loop TELEMETRIE creation')
print(end - start)
#Memory error, but the code works
        
### TELEMETRIE metadata
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\TELEMETRIE\TELEMETRIE\Metadata_OW_telemetrie.xlsx'
tel_meta = pd.read_excel(path, sheet_name = 0)
#The file has two sheets, the first one who has more information
# Columns 2 and 3 contains the coordinates of the stations


