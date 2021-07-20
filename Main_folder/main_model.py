# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:32:10 2021

@author: colompa
"""

# %% Modules

#General modules
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

#Custom modules
import functions_dp as dp
import functions_model as model

# %% Load the SSA components and the information on which one to use for each station


#1 Save ALL the components (results of the first for loop in SSA_script)
#2 load the components and the information about the elementary components
#3 use the information to reconstruct the 4 selected components

SSA_components = pkl.load(open('SSA_components.p', 'rb'))
SSA_information = pkl.load(open('SSA_information.p', 'rb'))

#%% Create the input dataframes for the model

df_log = pkl.load(open('loggers_dataframe.p', 'rb'))
df_prec = pkl.load(open('df_prec.p', 'rb'))
df_evap = pkl.load(open('df_evap.p', 'rb'))
near_p = pkl.load(open('near_prec.p', 'rb'))
near_p = pkl.load(open('near_evap.p', 'rb'))

#Instead of df_log, put the SSA components

for i in range(len(df_log.columns)):
    df_model = create_dataframe(df_log.iloc[:,i], df_prec, df_evap, near_p, near_e)
    result = launch_model(df_model)
    #Save the training results for each GW station


