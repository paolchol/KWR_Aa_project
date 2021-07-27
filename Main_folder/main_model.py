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

#Load the model parameters
m_par = pkl.load(open('model_parameters.p', 'rb'))

# %% Model launch

i = 0
predictions = {}
observations = {}
for name, _ in SSA_components.items():
    #Trend
    trend = dp.extract_attributes(SSA_components[name], SSA_information[i], 't')
    df_trend = dp.create_dataframe(trend, df_prec, df_evap, near_p, near_e)
    t_yhat, t_y, t_fitted = md.launch(df_trend, m_par[i], 't')
    #Yearly periodicity
    yper = dp.extract_attributes(SSA_components[name], SSA_information[i], 'yp')
    df_yper = dp.create_dataframe(yper, df_prec, df_evap, near_p, near_e)
    y_yhat, y_y, y_fitted = md.launch(df_yper, m_par[i], 'yp')
    #Monthly periodicity
    mper = dp.extract_attributes(SSA_components[name], SSA_information[i], 'mp')
    df_mper = dp.create_dataframe(mper, df_prec, df_evap, near_p, near_e)
    m_yhat, m_y, m_fitted = md.launch(df_mper, m_par[i], 'mp')
    #Noise
    noise = dp.extract_attributes(SSA_components[name], SSA_information[i], 'n')
    df_noise = dp.create_dataframe(noise, df_prec, df_evap, near_p, near_e)
    n_yhat, n_y, n_fitted = md.launch(df_noise, m_par[i], 'n')
    
    #Complete prediction
    predictions[i] = t_yhat + y_yhat + m_yhat + n_yhat
    observations[i] = t_y + y_y + m_y + n_y
    
    i = i + 1
    
    #Prediction doesn't have the date information anymore like this
    #It can be taken from SSA_components[name].orig_TS.index, and add the two weeks ahead
    #Prediction can be stored in a dictionary, to keep all the 20 runs for later


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

