# # -*- coding: utf-8 -*-
# """
# Data import from KWR's cloud

# - LOGGERS: groundwater heads and surface water levels from loggers
# - TELEMETRIE: surface water level from automatic loggers
# - D2extraction: smaller extractions by farmers, building locations, ecc.

# GW: groundwater
# OW: surface water

# @author: colompa
# """
# #try

# #Libraries
# import os
# import pandas as pd

# # Create lists containing the paths of the file to import

# def create_list_files(listfiles, path, extention):
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if file.endswith(extention):
#                 listfiles.append(root + '\\' + file)
#     return listfiles

# #Files from LOGGERS
# path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\GW'
# log_GW_files = []
# log_GW_files = create_list_files(log_GW_files, path, '.csv')

# path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\OW'
# log_OW_files = []
# log_OW_files = create_list_files(log_OW_files, path, '.csv')

# #Files from TELEMETRIE
# path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\TELEMETRIE\TELEMETRIE'
# tel_OW_files = []
# tel_OW_files = create_list_files(tel_OW_files, path, '.csv')

# #File from D2extraction
# path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\D2extraction\Singlepoint'
# extr_file = []
# extr_file = create_list_files(extr_file, path, '.csv')

# # Dataframes creation

# ## LOGGERS metadata
# #   Columns 5 and 6 contains the coordinates of the stations
# path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\Metadata_loggers_OW_GW.xlsx'
# log_meta = pd.read_excel(path)

# import datetime

# start = datetime.datetime(1980, 1, 1)
# end = datetime.datetime(2020, 12, 31)
# #1980-2020 to start, then we can change the range

# date = pd.date_range(start = start, end = end)
# date = date.strftime('%Y-%m-%d')

# loggers_GW = pd.DataFrame({"date": date})
# loggers_OW = pd.DataFrame({"date": date})

# import time
# import numpy as np

# def create_logger_dataframe(list_files, dataframe):
#     for file in list_files:
#         load = pd.read_csv(file, sep = ';', header = 0)  #Load the .csv
#         x = pd.DatetimeIndex(load.iloc[:,0]).strftime('%Y-%m-%d') #Extract the date and change date format
#         y = load.iloc[:,1]   #Extract the data
        
#         #Here arrange the sub-daily observations
        
#         name = os.path.basename(file)[1:-4]   
#         dataframe[name] = np.nan   #Create a new column with the name of the station
#         dataframe.loc[dataframe['date'].isin(x), name] = y[x.isin(dataframe.iloc[:,0])].tolist()
#     #different lengths: due to the fact that the csv has sub-daily observations
#     #introduce a way to take the mean of the observations which occurs on the same day
#     return dataframe

# start = time.time()

# loggers_GW = create_logger_dataframe(log_GW_files, loggers_GW)

# end = time.time()
# print('Elapsed time  - For-loop Loggers GW creation')
# print(end - start)

# loggers_OW = create_logger_dataframe(log_OW_files, loggers_OW)

# # In the metadata the coordinates of the stations are associated with the names of the columns

# ## TELEMETRIE metadata
# #   Columns 5 and 6 contains the coordinates of the stations
# path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\TELEMETRIE\TELEMETRIE\Metadata_OW_telemetrie.xlsx'
# tel_meta = pd.read_excel(path, sheet_name=0)
# #The file has two sheets, the first one who has more information

# telem_OW = pd.DataFrame({"date": date})

# import datetime

# df = load
# df = df.iloc[first_row:,:]
# df.rename(columns = {df.columns[date_pos]: "date"}, inplace = True)

# df.loc[:,'date'] = pd.to_datetime(df.loc[:,'date'])
# df.loc[:,'date'] = df.loc[:,'date'].apply( lambda df: datetime.datetime(year=df.year, month=df.month, day=df.day))
# df.index = df.loc[:,'date']
# df.drop('date', axis=1, inplace=True)
# df_daily_avg = df.iloc[:,1].resample('1D').mean()
# #problem: the columns are dtype: object, change it into float

# df[:,1].astype('float')
# #df.iloc[:,date_pos] = pd.DatetimeIndex(df.iloc[:,date_pos])
# #df.index = pd.DatetimeIndex(df.iloc[:,date_pos])

# df = df.iloc[:,date_pos+1:]
# df.groupby(df.index.date).mean()

# df.index.date


# def daily_mean(df, date_pos = 0, first_row = 0):
#     #Select only the rows containing data
#     df = df.iloc[first_row:,:]
    
#     #Create the daily dataframe
#     df.iloc[:,date_pos] = pd.DatetimeIndex(df.iloc[:,date_pos]).strftime('%Y-%m-%d')
#     df.rename(columns = {df.columns[date_pos]: "date"}, inplace = True)
#     x = df.drop_duplicates(subset = ['date'], keep = 'first', ignore_index = True).iloc[:,date_pos]
#     daily_data = pd.DataFrame({"date": x})
    
#     col=df.columns[1]
#     for col in df.columns[1:]: #generalize
#         #Create daily data column
#         df.loc[:,col]
        
#         #Attach it to the daily dataframe
#         daily_data[]
    
 

# for file in tel_OW_files:
#     load = pd.read_csv(file, sep = ';', header = 0)
#     load.iloc[:,1].replace(',', '.')
#     #Fuction to take the daily mean of the data for each column in input
#     #Output: first column date, other columns daily mean of the data
    
#     for station in load.columns[1:]:
#         if station in telem_OW.columns:
#             #Add the data in the same column based on the date
                
#         else:
#             #Create a new column and add the data based on the date
#             x = pd.DatetimeIndex(load.iloc[:,0]).strftime('%Y-%m-%d') #Extract the date and change date format
#             y = load.loc[:,station]   #Extract the data
            
            
#             dataframe[station] = np.nan   #Create a new column with the name of the station
#             dataframe.loc[dataframe['date'].isin(x), station] = y[x.isin(dataframe.iloc[:,0])].tolist()


        
        
# #telem_OW = telem_OW.append(load)


# load = pd.read_csv(tel_OW_files[1], sep = ';', header = 0)

# #Insert the data for each station at the correct date


# dataframe[name] = np.nan   #Create a new column with the name of the station
# dataframe.loc[dataframe['date'].isin(x), name] = y[x.isin(dataframe.iloc[:,0])].tolist()
# #The data are already subdivided by columns with the name of the station for each column
# #Just put them together to create a unique dataframe

# #Notes
# #This lets me extract the file name from the path
# os.path.basename(extr_file[0])

# #This is the basic operation to load the csv
# df = pd.read_csv(extr_file[0], sep = ';', header = 0)
# print (df)