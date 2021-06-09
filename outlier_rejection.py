# -*- coding: utf-8 -*-
"""
Outlier rejection from loggers dataset

    - Load loggers dataset 2007-2020
    - Remove the outliers
        - Keep the outliers information separately


@author: colompa
"""

#Libraries
import os
import pandas as pd
import datetime
import time
import numpy as np

# Load the dataframe to clean

path = r'D:\Users\colompa\Documents\KWR_Internship\Data\logger_GW_20072020_clean.csv'
loggers = pd.read_csv(path, sep = ',', index_col=0)

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

check_outliers(loggers)

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

log_out = remove_outliers(loggers)