# -*- coding: utf-8 -*-
"""
Model experiments

This script will focus on developing the full modelling procedure for one station
only, in order to define general functions that will work on the full modelling
procedure later

What it needs to be done:
    
    # Functions #
    - matrix_processing:
        Input: matrix of features
        Actions: add date as index, split, feature scaling
        Output: train, validation and test dataframes
    - baseline*:
        Run a linear model to compare the performances
        https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/
    - launch_model:
        Input: matrix, dictionary of parameters (?)

*Should the baseline be computed on the SSA components as well or on the full
original time series?
I would say to compone it on the SSA components as well
The baseline function could be run after the components in the model, by taking
already all the three components as input and by returning baseline_prediction
as output (already summed up) 


@author: colompa
"""

# %% Modules

#General modules
import pandas as pd
import numpy as np
import pickle as pkl

#Visualization
# from matplotlib import pyplot

#Tensor Flow
import tensorflow as tf

#Custom modules
import functions_dp as dp
from functions_dp import components_SSA
from class_SSA import SSA
import functions_model as md
from functions_model import model_par
from functions_model import single_par

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

# %% For one station only

## Attribute extraction ##
trend = dp.extract_attributes(SSA_components['log_ssa1'], SSA_information[0], 't')
yper = dp.extract_attributes(SSA_components['log_ssa1'], SSA_information[0], 'yp')
mper = dp.extract_attributes(SSA_components['log_ssa1'], SSA_information[0], 'mp')
noise = dp.extract_attributes(SSA_components['log_ssa1'], SSA_information[0], 'n')

## Dataframe creation ##
df_trend = dp.create_dataframe(trend, df_prec, df_evap, near_p, near_e)
df_yper = dp.create_dataframe(yper, df_prec, df_evap, near_p, near_e)
df_mper = dp.create_dataframe(mper, df_prec, df_evap, near_p, near_e)
df_noise = dp.create_dataframe(noise, df_prec, df_evap, near_p, near_e)

## Matrix pre-processing ##
# train_X, test_X, train_y, test_y, ts_par = matrix_processing(df_trend, 0.7, 3, 4, 5)
#Final configuration below, now just 4 and 5 to be faster
train_X, test_X, train_y, test_y, ts_par = md.matrix_processing(df_trend, 0.7, 3, lag_in = 30, lag_out = 14)

## Model parameters definition ##
m_par = single_par(SSA_information[0].name, 50, 1000, 72)
#these model_par can already be defined before and put on a dictionary in the
#same way done for SSA_information, so that each station has its parameters
#associated

## Launch the model ##
t_fitted = md.fit_model(train_X, test_X, train_y, test_y, m_par)
t_yhat, t_y, t_rmse = md.model_predict(t_fitted, test_X, test_y, ts_par)

#Yper
train_X, test_X, train_y, test_y, ts_par = md.matrix_processing(df_yper, 0.7, 3, lag_in = 30, lag_out = 14)
m_par = model_par(SSA_information[0].name, 50, 1000, 72)
y_fitted = md.fit_model(train_X, test_X, train_y, test_y, m_par)
y_yhat, y_y, y_rmse = md.model_predict(y_fitted, test_X, test_y, ts_par)

#Mper
train_X, test_X, train_y, test_y, ts_par = md.matrix_processing(df_mper, 0.7, 3, lag_in = 30, lag_out = 14)
m_par = model_par(SSA_information[0].name, 50, 1000, 72)
m_fitted = md.fit_model(train_X, test_X, train_y, test_y, m_par)
m_yhat, m_y, m_rmse = md.model_predict(m_fitted, test_X, test_y, ts_par)

#Noise
train_X, test_X, train_y, test_y, ts_par = md.matrix_processing(df_noise, 0.7, 3, lag_in = 30, lag_out = 14)
n_par = model_par(SSA_information[0].name, 50, 1000, 72)
n_fitted = md.fit_model(train_X, test_X, train_y, test_y, m_par)
n_yhat, n_y, n_rmse = md.model_predict(n_fitted, test_X, test_y, ts_par)

## Regroup the extracted attributes ##
prediction = t_yhat + y_yhat + m_yhat + n_yhat
observation = t_y + y_y + m_y + n_y

#How to show the results?
# Check in time_series.ypinb


# %% Save the models

pkl.dump(fitted, open('fitted_1000.p', 'wb'))

#%% Trials

df_prec.index.get_loc(pd.to_datetime(df_prec.index) == trend.index[0])
sum(trend.index[0] == df_prec.index)

pd.Series(df_prec.index).strftime('%Y-%m-%d')


def feature_scale(df):
    scaled = (df - df.mean()) / df.std()
    return scaled

dp.check_outliers(df_evap)
# fast_df_visualization(pd.DataFrame(m))

x = 2
val = df_trend.values
print(val[:, :x])
print(val[:, -x])

superv = series_to_supervised(df_trend, 3, 10)
superv = superv.values
print(superv[:, -3])
print(superv[:, n_obs])

lag_in = 3
n_features = 3
n_obs = lag_in*n_features
lag_out = 10
n_pred = n_features * lag_out

pos = [n_obs]
for i in range(int(n_pred/n_features)):
    pos.append(pos[i] + n_features)
    if(pos[i + 1] >= (n_pred + n_obs)):
        pos.pop(i + 1)

cc = superv[:, pos]

superv[pos]

val = df_trend.values
val = val.astype('float32')

#Split the dataset in train and test

train=0.7
n = len(val)
train_df = val[0:int(n*train)]
test_df = val[int(n*train):]


