# -*- coding: utf-8 -*-
"""
NA filling in a dataframe

Workflow:
    - Load the dataframe
    - Fill the NA
    - Export the filled dataframe

@author: colompa
"""

#Libraries
import os
import pandas as pd
from datetime import datetime
import time
import numpy as np

#Functions
# Load the dataframe to fill
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\logger_GW_nooutliers.csv'
loggers = pd.read_csv(path, sep = ',', index_col = 0)

# Compare different interpolation methods with the function DataFrame.interpolate()
methods = ['linear', 'quadratic', 'cubic']
means_trial = pd.DataFrame({'name': loggers.columns, 'original': loggers.mean().values})
medians_trial = pd.DataFrame({'name': loggers.columns, 'original': loggers.median().values})
std_trial = pd.DataFrame({'name': loggers.columns, 'original': loggers.std().values})
for method in methods:
    trial = loggers.interpolate(method, limit = 30) #max 30 consecutive days interpolated
    means_trial[method] = trial.mean().values
    medians_trial[method] = trial.median().values
    std_trial[method] = trial.std().values

## Compare mean, median and sd of the original dataframe and on the interpolated dataframes
for i in range(2,len(medians_trial.columns)):
    delta = medians_trial.iloc[:,1]-medians_trial.iloc[:,i]
    print(f'Mean MEDIANS delta between the original an the {medians_trial.columns[i]} interpolation method:\n{delta.abs().mean()}')
for i in range(2,len(medians_trial.columns)):
    delta = std_trial.iloc[:,1]-std_trial.iloc[:,i]
    print(f'Mean STD delta between the original an the {medians_trial.columns[i]} interpolation method:\n{delta.abs().mean()}')

#Based on these result, the linear interpolation seems to perform better than
#the other two methods, since the outcome of the interpolation presents the most
#similar statistics to the original dataset

# Create the new dataframe
loggers_nona = loggers.interpolate('linear', limit = 30)

# Export the dataframe
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\logger_GW_noNA.csv'
loggers_nona.to_csv(path_or_buf = path, sep = ",", index = True)
