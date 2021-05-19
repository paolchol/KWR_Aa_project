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

# Create lists containing the paths of the file to import

def create_list_files(listfiles, path, extention):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extention):
                listfiles.append(root + '\\' + file)
    return listfiles

#Files from LOGGERS
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\GW'
log_GW_files = []
log_GW_files = create_list_files(log_GW_files, path, '.csv')

path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\OW'
log_OW_files = []
log_OW_files = create_list_files(log_OW_files, path, '.csv')

#Files from TELEMETRIE
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\TELEMETRIE\TELEMETRIE'
tel_OW_files = []
tel_OW_files = create_list_files(tel_OW_files, path, '.csv')

#File from D2extraction
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\D2extraction\Singlepoint'
extr_file = []
extr_file = create_list_files(extr_file, path, '.csv')

# Dataframes creation

## LOGGERS metadata
#   Columns 5 and 6 contains the coordinates of the stations
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\Metadata_loggers_OW_GW.xlsx'
log_meta = pd.read_excel(path)

#Create a blank dataframe, with the first column filled by dates
#1980-2020 to start, then we can change the range

#Load the station's data using the list of paths
#Take the useful columns: dates and data
#Check the format of the dates and if it is different from the new dataframe's one,
#   change it accordingly
#Add the data column to the new dataframe created, where the dates are the same
#Give the name of the file to the columns name

## TELEMETRIE metadata
#   Columns 5 and 6 contains the coordinates of the stations
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\TELEMETRIE\TELEMETRIE\Metadata_OW_telemetrie.xlsx'
tel_meta = pd.read_excel(path)
#The file has two sheets, take only the first one who has more informations

#The data are already subdivided by columns with the name of the station for each column
#Just put them together to create a unique dataframe

#Notes
#This lets me extract the file name from the path
os.path.basename(extr_file[0])

#This is the basic operation to load the csv
df = pd.read_csv(extr_file[0], sep = ';', header = 0)
print (df)