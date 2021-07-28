# -*- coding: utf-8 -*-
"""
Module containing functions useful for the data-preprocessing

List of the functions contained:
    
    # Dataframe operations #
    - daily_mean/daily_mean_index:
        Computes the daily mean of a sub-daily observations dataframe
    - create_list_files:
        Creates a list containing the paths of the fiels to import
    - get_loggers_two_dates:
        Returns a dataframe containing the LOGGERS stations with a time series
        ending at 'end_year' on the columns
    - create_dataframe:
        Creates a dataframe containing the variable to predict and two
        exogeneous variables
    
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
    
    # Spatial operations #
    - find_nearest_point:
        Finds the nearest point to a specified point (point) from a set of
        coordinates (tgpoints)
    
    # SSA analysis and operations #
    - plot_Wcorr_Wzomm:
        Plots the W-Correlation matrix already with the selected zoom and a
        standard title
    - plot_SSA_results:
        
    - components_SSA:

@author: colompa
"""

#%% Libraries
import os
import pandas as pd
import datetime
import numpy as np
import pyproj
import matplotlib.pyplot as plt

#%% Dataframe operations

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

def month_to_daily(df, value, col, m, y):
    df.loc[(df.index.month == m) & (df.index.year == y), col] = \
        round(value/len(df.loc[(df.index.month == m) & (df.index.year == y), col]), 3)
    return df

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

def create_dataframe(var_, ex_df1, ex_df2, close1, close2, name1 = 'prec', name2 = 'evap'):
    #var_: variable to predict
    #ex_df: exogeneous variables dataframes (1: precipitation, 2: evaporation)
    #close: closest precipitation station's information
    
    def get_st(var_, ex_df, close):
        ex_st_name = close.loc[var_.name].loc['target_name']
        ex_st = ex_df[ex_st_name]
        return ex_st
    
    def get_xy(ex_df, var_, start, end):
        t = pd.to_datetime(ex_df.index) == var_.index[start]
        x = [x for x, i in enumerate(t) if i]
        t = pd.to_datetime(ex_df.index) == var_.index[end - 1]
        y = [y+1 for y, i in enumerate(t) if i]
        return x, y
    
    ex_st1 = get_st(var_, ex_df1, close1)
    ex_st2 = get_st(var_, ex_df2, close2)
    #Get the data from starting to ending non-missing values
    start = var_.index.get_loc(var_.first_valid_index())
    end = var_.index.get_loc(var_.last_valid_index()) + 1
    model_df = pd.DataFrame()
    model_df[var_.name] = var_[start:end]
    x, y = get_xy(ex_st1, var_, start, end)
    model_df[name1] = ex_st1[x[0]:y[0]].values
    x, y = get_xy(ex_st2, var_, start, end)
    model_df[name2] = ex_st2[x[0]:y[0]].values
    return model_df

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
        print(f'Percentage of outliers: {((sum(df[column] > upper_limit) + sum(df[column] < lower_limit))/len(df[column]))*100}')

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

#%% Spatial operations

def find_nearest_point(point, tgpoints, namelat = 'Lat', namelon = 'Lon', namepoint = 'label'):
    # This function finds the nearest point to a specified point (point) from
    # a set of coordinates (tgpoints)
    #Define the system
    geod = pyproj.Geod(ellps='WGS84')
    #Create a dataframe containing the set points
    #df = pd.DataFrame(point)
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

#%% SSA Analysis and operations

def plot_Wcorr_Wzomm(SSA_object, name, num = 0):
    if(num == 0):
        SSA_object.plot_wcorr()
        plt.title(f"W-Correlation for {name} - 365 days window")
    else:
        SSA_object.plot_wcorr(max = num)
        plt.title(f"W-Correlation for {name} - 365 days window \n- Zoomed at the first {num + 1} components")

 
def plot_SSA_results(SSA_object, Fs, noise = 0, label = 'SSA results', file = 'temp.html', final = False,
                     xlab = "Date", ylab = "Water table level [MAMSL]"):
    import plotly.express as px
    from plotly.offline import plot
    #import plotly.graph_objs as go
    
    df = pd.DataFrame({'Original': SSA_object.orig_TS.values})
    df.index = SSA_object.orig_TS.index
    if(noise != 0):
        df['Noise'] = SSA_object.reconstruct(noise).values
    if(final):
        names = ['Trend', 'Periodicity']
        for i, F in enumerate(Fs):
            name = names[i]
            df[name] = SSA_object.reconstruct(F).values
    else:
        for i, F in enumerate(Fs):
            name = f'F{i}'
            df[name] = SSA_object.reconstruct(F).values
    #Would be nicer to have the original series with a bit of transparency
    #Also to add labels to the axis and a title
    # plt.title(label)
    # plt.xlabel("Date")
    # plt.ylabel("Level [MASL - Meters Above Sea Level]")
    # Also save it with a name instead of temp-plot, and in a specified path
    figure = px.line(df)
    figure.update_layout(
        xaxis_title = xlab,
        yaxis_title = ylab,
        legend_title = "Variables"
        )
    plot(figure, filename = file)
    return(df)

class components_SSA():
    def __init__(self, num, name, trend, Yper, Mper, Nstart, Nend):
        self.num = num
        self.name = name
        self.trend = trend
        self.Yper = Yper
        self.Mper = Mper
        self.Nstart = Nstart
        self.Nend = Nend

def extract_attributes(SSA_object, SSA_info, attribute):
    if(attribute == 't'):
        #Trend
        out = SSA_object.reconstruct(SSA_info.trend).values
    elif(attribute == 'yp'):
        #Yearly periodicity
        out = SSA_object.reconstruct(SSA_info.Yper).values
    elif(attribute == 'mp'):
        #Monthly periodicity
        out = SSA_object.reconstruct(SSA_info.Mper).values
    elif(attribute == 'n'):
        #Noise
        out = SSA_object.reconstruct(slice(SSA_info.Nstart, SSA_info.Nend)).values
    else:
        print(f'The inserted attribute {attribute} is not correct')
        return
    #Take the name from SSA_info and place it as the series name
    out = pd.Series(out)
    out.name = SSA_info.name
    out.index = pd.to_datetime(SSA_object.orig_TS.index).strftime('%Y-%m-%d')
    return out

# %% Visualization

def fast_df_visualization(df):
    from matplotlib import pyplot
    pyplot.figure()
    for i, column in enumerate(df.columns, start = 1):
    	pyplot.subplot(len(df.columns), 1, i)
    	pyplot.plot(df[column].values)
    	pyplot.title(column, y = 0.5, loc = 'right')
    pyplot.show()

def interactive_df_visualization(df, xlab = 'X', ylab = 'Y', file = 'temp.html'):
    import plotly.express as px
    from plotly.offline import plot
    figure = px.line(df)
    figure.update_layout(
        xaxis_title = xlab,
        yaxis_title = ylab,
        legend_title = "Variables"
        )
    plot(figure, filename = file)
