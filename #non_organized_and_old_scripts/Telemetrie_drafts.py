# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:38:38 2021

@author: colompa
"""

#Below
#Previous drafts/discarded parts

# Arrange the paths to the files in the cloud
## Files from TELEMETRIE
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\TELEMETRIE\TELEMETRIE'
tel_OW_files = []
tel_OW_files = create_list_files(tel_OW_files, path, '.csv')

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

tel_coord = tel_meta.iloc[:,[0,1,2]]
# Transform the coordinates into WGS1984?

from pyproj import Proj, transform

inProj = Proj(init='epsg:28992') #Amersfoort, dont know if it is the correct one
outProj = Proj(init='epsg:4326')
for i in range(len(log_coord)):
    tel_coord.iloc[i,1:] = transform(inProj,outProj,tel_coord.iloc[i,1],tel_coord.iloc[i,2])
print(tel_coord)

# Arrange the paths to the files in the cloud
## Files from LOGGERS OW
path = r'\\nwg.local\dfs\projectdata\P402045_238\ReceivedData\DataNZV\LOGGERS\LOGGERS\OW'
log_OW_files = []
log_OW_files = create_list_files(log_OW_files, path, '.csv')

loggers_OW = create_logger_dataframe(log_OW_files, loggers_OW, round(mean))
