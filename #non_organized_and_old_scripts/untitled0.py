# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:46:00 2021

@author: colompa
"""

def create_logger_dataframe(list_files, df, threshold, NAs):
    #In the original file
    #First column: date and time
    #Second column: measurement
    #The output presents:
    #   - Rows: daily dates 
    #   - Columns: stations' daily observations
    fields = ['Tijd', 'DIVER (m tov NAP)']
    for i, file in enumerate(list_files, start=0):
        print(i, file)
        if NAs[i,0]*NAs[i,1] >= threshold:
            #The stations are selected based on the length of their
            #observations' time series
            load = pd.read_csv(file, sep = ";", usecols=fields)
            daily = daily_mean(load, 0)
            x = daily.index.strftime('%Y-%m-%d')
            y = daily.values
            name = os.path.basename(file)[1:-4]
            df[name] = np.nan
            df.loc[df['date'].isin(x), name] = y[x.isin(df.iloc[:,0])]
    return df

loggers_GW = pd.DataFrame({"date": date})
start = time.time()
loggers_GW = create_logger_dataframe(log_GW_files, loggers_GW, round(mean), NA_count)
end = time.time()
print('Elapsed time  - For-loop Loggers GW creation')
print(end - start)

#The function can be updated by putting the dates in the index in df instead
#of the first column
