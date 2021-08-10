# -*- coding: utf-8 -*-
"""
Model experiments for river data

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

df_river = pkl.load(open('df_river_clean.p', 'rb'))
df_river.index = df_river.index.strftime('%Y-%m-%d')

SSA_river = pkl.load(open('SSA_flow.p', 'rb'))
SSA_information = pkl.load(open('SSA_flow_info.p', 'rb'))

# %% Extract the components

trend = dp.extract_attributes(SSA_river, SSA_information, 't')
yper = dp.extract_attributes(SSA_river, SSA_information, 'yp')
mper = dp.extract_attributes(SSA_river, SSA_information, 'mp')
noise = dp.extract_attributes(SSA_river, SSA_information, 'n')

#Define the model parameters
t_par = single_par(SSA_information.name, 50, 10, 72)
yp_par = single_par(SSA_information.name, 50, 10, 72)
mp_par = single_par(SSA_information.name, 50, 10, 72)
n_par = single_par(SSA_information.name, 50, 10, 72)
m_par = model_par(SSA_information.name, trend = t_par, yper = yp_par, mper = mp_par, noise = n_par)

# %% Dataframe creation

exogeneous = ['prec','evap', 'extr1', 'extr2', 'extr3']
df_trend = pd.DataFrame(trend).join(df_river[exogeneous])
df_yper = pd.DataFrame(yper).join(df_river[exogeneous])
df_mper = pd.DataFrame(mper).join(df_river[exogeneous])
df_noise = pd.DataFrame(noise).join(df_river[exogeneous])

# %% Fit the models

#Trend
train_X, test_X, train_y, test_y, trend_par = md.matrix_processing(df_trend, 0.7, 6, lag_in = 30, lag_out = 14)
t_fitted = md.fit_model(train_X, test_X, train_y, test_y, m_par.trend)
#Yearly periodicity
train_X, test_X, train_y, test_y, yper_par = md.matrix_processing(df_yper, 0.7, 6, lag_in = 30, lag_out = 14)
y_fitted = md.fit_model(train_X, test_X, train_y, test_y, m_par.yper)
#Six-months periodicity
train_X, test_X, train_y, test_y, mper_par = md.matrix_processing(df_mper, 0.7, 6, lag_in = 30, lag_out = 14)
m_fitted = md.fit_model(train_X, test_X, train_y, test_y, m_par.mper)
#Noise
train_X, test_X, train_y, test_y, noise_par = md.matrix_processing(df_noise, 0.7, 6, lag_in = 30, lag_out = 14)
n_fitted = md.fit_model(train_X, test_X, train_y, test_y, m_par.noise)


# %% Obtain the prediction

def prediction(model, df, par, n_feat = 6, lag_in = 30, lag_out = 1):
    df_scale = df.values
    df_scale = (df_scale - par[0]['mean']) / par[0]['std']
    
    x = lag_in + lag_out
    subset = df_scale[-x:, :]
    obs = md.series_to_supervised(subset, lag_in, lag_out).values
    n_obs = lag_in * n_feat
    
    obs = obs[:, :n_obs]
    obs = obs.reshape(obs.shape[0], 30, 6)
    
    yhat = model.predict(obs)
    yhat = np.transpose(yhat)
    yhat = yhat*par[0]['std'][0] + par[0]['mean'][0]
    return(yhat)

that = prediction(t_fitted, df_trend, trend_par)
yphat = prediction(y_fitted, df_yper, yper_par)
mphat = prediction(m_fitted, df_mper, mper_par)
nhat = prediction(n_fitted, df_noise, noise_par)

y = df_river.values[-14:, 0]
yhat = that + yphat + mphat + nhat

output = pd.DataFrame(yhat, index = df_river.index[-14:])
output['y'] = y
output.rename(columns = {0: 'yhat'}, inplace = True)

dp.fast_df_visualization(output)


# %% Draft 
#Obtain the results for the trend
df_scale = df_trend.values
df_scale = (df_scale - ts_par[0]['mean']) / ts_par[0]['std']

# subset = df_scale[-44:, :]
subset = df_scale[-31:, :]
# obs = md.series_to_supervised(subset, 30, 14).values
obs = md.series_to_supervised(subset, 30, 1).values
n_obs = 30*6
obs_1 = obs[:, :n_obs]
obs_2 = obs_1.reshape(obs_1.shape[0], 30, 6)

yhat = t_fitted.predict(obs_2)

yhat = np.transpose(yhat)
yhat_1 = yhat*ts_par[0]['std'][0] + ts_par[0]['mean'][0]
y = df_scale[-14:, 0].astype('float32')*ts_par[0]['std'][0] + ts_par[0]['mean'][0]




output = pd.DataFrame(yhat_1, index = df_trend.index[-14:])
output['y'] = y
output.rename(columns = {0: 'yhat'}, inplace = True)

dp.fast_df_visualization(output)
dp.fast_df_visualization(df_trend)


# Run for all the components and put them together




