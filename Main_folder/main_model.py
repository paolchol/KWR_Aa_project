# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:32:10 2021

@author: colompa
"""

# %% Modules

#General modules
import pandas as pd
import numpy as np
import pickle as pkl

#Visualization
from matplotlib import pyplot
import plotly.express as px
from plotly.offline import plot

#Tensor Flow
import tensorflow as tf

#Custom modules
import functions_dp as dp
from functions_dp import components_SSA
from class_SSA import SSA
import functions_model as md
from functions_model import model_par

# %% Load the necessary data

#Load the SSA components and the information on which one to use for each station
SSA_components = pkl.load(open('SSA_components.p', 'rb'))
SSA_information = pkl.load(open('SSA_information.p', 'rb'))

#Load the exogeneous data
df_prec = pkl.load(open('df_prec.p', 'rb'))
df_evap = pkl.load(open('df_evap.p', 'rb'))
near_p = pkl.load(open('near_prec.p', 'rb'))
near_e = pkl.load(open('near_evap.p', 'rb'))

#Remove outliers precipitation data
df_prec = dp.remove_outliers(df_prec, 0)

# %% Model launch

i = 0
for name, _ in SSA_components.items():
    trend = dp.extract_attributes(SSA_components[name], SSA_information[i], 't')
    df_trend = dp.create_dataframe(trend, df_prec, df_evap, near_p, near_e)
    
    #Dataset operations
    
    #Data windowing
    
    pred_t = launch_model(df_trend)
    #launch_model will have to return a series
    
    
    #Same for yper, mper and noise
    yper = dp.extract_attributes(SSA_components[name], SSA_information[i], 'yp')
    mper = dp.extract_attributes(SSA_components[name], SSA_information[i], 'mp')
    noise = dp.extract_attributes(SSA_components[name], SSA_information[i], 'n')
    
    prediction = pred_t + pred_yp + pred_mp + pred_noise
    #Prediction doesn't have the date information anymore like this
    #It can be taken from SSA_components[name].orig_TS.index, and add the two weeks ahead
    #Prediction can be stored in a dictionary, to keep all the 20 runs for later
    
    
    i = i + 1


# %% Visualize
#Visualize the result
# orig = trend+yper+mper+noise
# df = pd.DataFrame({'Original': SSA_components['log_ssa1'].orig_TS.values})
# df.index = SSA_components['log_ssa1'].orig_TS.index
# df['orig'] = orig.values

# figure = px.line(df)
# figure.update_layout(
#     xaxis_title = xlab,
#     yaxis_title = ylab,
#     legend_title = "Variables"
#     )
# plot(figure)

