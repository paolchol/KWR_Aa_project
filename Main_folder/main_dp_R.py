# -*- coding: utf-8 -*-
"""
Data pre-processing for the river data

# Dataset characteristics: #
    - Variables included:
        Level, flow, precipitation, evaporation, temperature,
        extraction    
    - Time range:
        Level, flow, precipitation, evaporation, temperature: 1995 - 2020
        Extraction: 2003 - 2020
    - Time step:
        Level, flow, precipitation, evaporation, temperature: daily
        Extraction: monthly

# Operations: #
    - 
    

@author: colompa
"""

# %% Libraries and values definition

import os
import pandas as pd
import numpy as np
import datetime
import time
import pickle as pkl
import pyproj

#Custom libraries and classes
import functions_dp as dp
from functions_dp import components_SSA
from class_SSA import SSA

#Path to the plot folder
path_plot = r'D:\Users\colompa\Documents\KWR_project\Spyder_project\plots'

# %% Load the river dataframe

df_river = pkl.load(open('River.pkl', 'rb'))

#Visualize the dataframe
dp.fast_df_visualization(df_river)

#Keep the columns needed and rename them
#Keep only the flow
keep = [1, 2, 3, 6, 7, 8]
df_river = df_river.iloc[:,keep]
df_river.columns = ['flow', 'prec', 'evap', 'extr1', 'extr2', 'extr3']

#Obtain a daily value for extraction from the monthly values
date_notna = df_river.loc[df_river['extr1'] > 0,:].index
extr1_notna = df_river.loc[df_river['extr1'] > 0, 'extr1'].values
extr2_notna = df_river.loc[df_river['extr2'] > 0, 'extr2'].values
extr3_notna = df_river.loc[df_river['extr3'] > 0, 'extr3'].values

for i in range(len(date_notna)):
    m = date_notna[i].month
    y = date_notna[i].year
    df_river = dp.month_to_daily(df_river, extr1_notna[i], 'extr1', m, y)
    df_river = dp.month_to_daily(df_river, extr2_notna[i], 'extr2', m, y)
    df_river = dp.month_to_daily(df_river, extr3_notna[i], 'extr3', m, y)

#Keep only the complete dataset (from 2003 to 2020)
pos = df_river.index.strftime('%Y-%m-%d').get_loc(datetime.datetime(2003, 1, 1).strftime('%Y-%m-%d'))
df_river = df_river.iloc[pos:, :]

# %% Outlier rejection

#Visualize again the dataset
dp.fast_df_visualization(df_river)
dp.interactive_df_visualization(df_river, file = 'plot_df_river.html')

#Check the presence of outliers
dp.check_outliers(df_river)

#Remove the outliers from flow, precipitation and evaporation
#The outliers rejection in this case is removing values that do not behave as outliers, so it is not performed
# col = [0, 1, 2]
# df_river.iloc[:, col] = dp.remove_outliers(df_river.iloc[:, col])
# dp.interactive_df_visualization(df_river, file='after_outlier.html')

# %% NA removal

#Check the presence of NAs
dp.check_NAs(df_river)

#Fill the NAs
df_river = df_river.interpolate('linear', limit = 365)

# %% Save the df

pkl.dump(df_river, open('df_river_clean.p', 'wb'))
# df_river = pkl.load(open('df_river_clean.p', 'rb'))

# %% SSA on the flow

#SSA components
L = 365
process = df_river.iloc[:, 0][df_river.iloc[:, 0].notna()]
SSA_flow = SSA(process, L)
pkl.dump(SSA_flow, open('SSA_flow.p', 'wb'))

#Obtain trend, periodicities and noise
dp.plot_Wcorr_Wzomm(SSA_flow, 'River flow')
dp.plot_Wcorr_Wzomm(SSA_flow, 'River flow', 49)
dp.plot_Wcorr_Wzomm(SSA_flow, 'River flow', 9)

#Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4]
F3 = [5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
dp.plot_SSA_results(SSA_flow, Fs, label = 'River flow - SSA', file = f'{path_plot}\\river_SSA.html')

#F0 is the overall trend
#F1 the yearly periodicity
#F2 the 6-month periodicity
#The rest will be taken as the noise

#Save these information
SSA_flow_info = components_SSA(0, 'river', F0, F1, F2, 5, 365)
pkl.dump(SSA_flow_info, open('SSA_flow_info.p', 'wb'))

