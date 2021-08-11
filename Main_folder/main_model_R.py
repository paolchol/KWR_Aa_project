# -*- coding: utf-8 -*-
"""
Model construction operated fo the river flow

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

#Set the model parameters
t_par = single_par(SSA_information.name, 50, 100, 72)
yp_par = single_par(SSA_information.name, 50, 100, 72)
mp_par = single_par(SSA_information.name, 50, 100, 72)
n_par = single_par(SSA_information.name, 50, 100, 72)
m_par = model_par(SSA_information.name, trend = t_par, yper = yp_par, mper = mp_par, noise = n_par)

#Load the model parameters
# m_par = pkl.load(open('model_parameters_R.p','rb'))

# %% Dataframe creation

exogeneous = ['prec','evap', 'extr1', 'extr2', 'extr3']
df_trend = pd.DataFrame(trend).join(df_river[exogeneous])
df_yper = pd.DataFrame(yper).join(df_river[exogeneous])
df_mper = pd.DataFrame(mper).join(df_river[exogeneous])
df_noise = pd.DataFrame(noise).join(df_river[exogeneous])

# %% Fit the models
# The last 70% of data is used to train the model, while the first 30% is used for validate it

#Trend
train_X, test_X, train_y, test_y, trend_par = md.matrix_processing(df_trend, 0.7, 6, lag_in = 30, lag_out = 14,
                                                                   forward = False)
t_fitted = md.fit_model(train_X, test_X, train_y, test_y, m_par.trend)
#Yearly periodicity
train_X, test_X, train_y, test_y, yper_par = md.matrix_processing(df_yper, 0.7, 6, lag_in = 30, lag_out = 14,
                                                                  forward = False)
y_fitted = md.fit_model(train_X, test_X, train_y, test_y, m_par.yper)
#Six-months periodicity
train_X, test_X, train_y, test_y, mper_par = md.matrix_processing(df_mper, 0.7, 6, lag_in = 30, lag_out = 14,
                                                                  forward = False)
m_fitted = md.fit_model(train_X, test_X, train_y, test_y, m_par.mper)
#Noise
train_X, test_X, train_y, test_y, noise_par = md.matrix_processing(df_noise, 0.7, 6, lag_in = 30, lag_out = 14,
                                                                   forward = False)
n_fitted = md.fit_model(train_X, test_X, train_y, test_y, m_par.noise)

# %% Save the fitted models

path_save = r'D:\Users\colompa\Documents\KWR_project\Spyder_project\Saved_model'
t_fitted.save(f'{path_save}\model_R_trend.h5')
y_fitted.save(f'{path_save}\model_R_yper.h5')
m_fitted.save(f'{path_save}\model_R_mper.h5')
n_fitted.save(f'{path_save}\model_R_noise.h5')

#To load them:
# from keras.models import load_model
# path_load = r'D:\Users\colompa\Documents\KWR_project\Spyder_project\Saved_model'
# t_fitted = load_model(f'{path_load}\model_R_trend.h5')
# y_fitted = load_model(f'{path_load}\model_R_yper.h5')
# m_fitted = load_model(f'{path_load}\model_R_mper.h5')
# n_fitted = load_model(f'{path_load}\model_R_noise.h5')

# %% Obtain the prediction

that = md.prediction(t_fitted, df_trend, trend_par)
yphat = md.prediction(y_fitted, df_yper, yper_par)
mphat = md.prediction(m_fitted, df_mper, mper_par)
nhat = md.prediction(n_fitted, df_noise, noise_par)

y = df_river.values[-14:, 0]
yhat = that + yphat + mphat + nhat

output = pd.DataFrame(yhat, index = df_river.index[-14:])
output['y'] = y
output.rename(columns = {0: 'yhat'}, inplace = True)

dp.fast_df_visualization(output)

#R2 square between the prediction and the observations
from sklearn.metrics import r2_score

r2 = r2_score(output['y'], output['yhat'])
print(f'The R2 score of the 14 days prediction is {r2}')

dp.interactive_df_visualization(output, xlab='Days', ylab='River flow [m3/s]', file = r'D:\Users\colompa\Documents\KWR_project\Spyder_project\plots\river_100.html')

dp.interactive_df_visualization(df_trend)
dp.interactive_df_visualization(df_noise)



