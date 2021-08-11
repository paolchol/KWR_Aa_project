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

# %% Fit the models using the last 70% for the training

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

r2_forward = r2_score(output['y'], output['yhat'])
print('** Forward result **')
print(f'The R2 score of the 14 days prediction is {r2_forward}')

r2_back = r2_score(output['y'], output['yhat'])
print('** Backward result **')
print(f'The R2 score of the 14 days prediction is {r2_back}')

#Better result with the 'backward' calibration

# %% Save the fitted model

from keras.models import load_model

t_fitted.save('model.h5')
prova_load = load_model('model.h5')
prova_load.summary()
prova_load.evaluate(test_X, test_y)
