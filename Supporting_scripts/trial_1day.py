# -*- coding: utf-8 -*-
"""
Trial: 1-day ahead prediction of river's water level

Trial script to perform a 1-day ahead prediction on the whole dataset,
with 30 days of data

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

SSA_river = pkl.load(open('SSA_level.p', 'rb'))
SSA_information = pkl.load(open('SSA_level_info.p', 'rb'))

# %% Extract the components

trend = dp.extract_attributes(SSA_river, SSA_information, 't')
yper = dp.extract_attributes(SSA_river, SSA_information, 'yp')
mper = dp.extract_attributes(SSA_river, SSA_information, 'mp')
noise = dp.extract_attributes(SSA_river, SSA_information, 'n')

#Set the model parameters
t_par = single_par(SSA_information.name, 50, 100, 72)
yp_par = single_par(SSA_information.name, 50, 100, 72)
mp_par = single_par(SSA_information.name, 50, 100, 72)
m_par = model_par(SSA_information.name, trend = t_par, yper = yp_par, mper = mp_par)

#Load the model parameters
#For example, the resulting parameters from the tuning
# m_par = pkl.load(open('model_parameters_R.p','rb'))

# %% Noise classification

val = trend + yper + mper
df_noise = pd.DataFrame({'val': val, 'noise': noise})
ngroups, nbounds = md.noise_group(df_noise)

# %% Dataframe creation

exogeneous = ['flow', 'prec','evap', 'extr1', 'extr2', 'extr3']
df_trend = pd.DataFrame(trend).join(df_river[exogeneous])
df_yper = pd.DataFrame(yper).join(df_river[exogeneous])
df_mper = pd.DataFrame(mper).join(df_river[exogeneous])

# %% Fit the models
# The last 70% of data is used to train the model, while the first 30% is used for validate it

Nf = len(df_river.columns) #number of features
trainp = 0.7    #percentage of the dataset to use a train set
IN = 30         #input data window
OUT = 1         #output data window
forwardTF = False   #condition to take the first trainp as train set (True) or the last (False)
oneday = np.take(np.where(OUT == 1, True, False), 0)

#Trend
_, _, _, _, trend_par = md.matrix_processing(df_trend, trainp, Nf, lag_in = IN, lag_out = OUT,
                                                                   forward = forwardTF)
#Yearly periodicity
_, _, _, _, yper_par = md.matrix_processing(df_yper, trainp, Nf, lag_in = IN, lag_out = OUT,
                                                                  forward = forwardTF)
#Six-months periodicity
_, _, _, _, mper_par = md.matrix_processing(df_mper, trainp, Nf, lag_in = IN, lag_out = OUT,
                                                                  forward = forwardTF)

# %% Load the fitted models

#To load them:
from keras.models import load_model
path_load = r'D:\Users\colompa\Documents\KWR_project\Spyder_project\Saved_model'
t_fitted = load_model(f'{path_load}\model_Rlevel_trend.h5')
y_fitted = load_model(f'{path_load}\model_Rlevel_yper.h5')
m_fitted = load_model(f'{path_load}\model_Rlevel_mper.h5')

# %% Obtain the prediction
#1-day ahead prediction for the whole period

def Xy_1daypred(df, n_features, par, lag_in = 30, lag_out = 1):
    val = df.values
    val = val.astype('float32')
    val = (val - par[0]['mean'])/par[0]['std']
    
    val = md.series_to_supervised(val, lag_in, lag_out)
    val = val.values
    
    n_obs = lag_in * n_features
    n_pred = lag_out * n_features
    pos = [n_obs]
    for i in range(int(n_pred/n_features)):
        pos.append(pos[i] + n_features)
        if(pos[i + 1] >= (n_pred + n_obs)):
            pos.pop(i + 1)
    X, y = val[:, :n_obs], val[:, pos]
    X = X.reshape((X.shape[0], lag_in, n_features))
    return X, y

def rescale(val, par):
    val = val*par[0]['std'][0] + par[0]['mean'][0]
    return val

trend_X, trend_y = Xy_1daypred(df_trend, 7, trend_par, lag_out = 1)
yper_X, yper_y = Xy_1daypred(df_yper, 7, yper_par)
mper_X, mper_y = Xy_1daypred(df_mper, 7, mper_par)

that = t_fitted.predict(trend_X)
that = rescale(that, trend_par)
yphat = y_fitted.predict(yper_X)
yphat = rescale(yphat, yper_par)
mphat = m_fitted.predict(mper_X)
mphat = rescale(mphat, mper_par)

yhat = that + yphat + mphat

# %% Create the output dataframe

y = rescale(trend_y, trend_par) + rescale(yper_y, yper_par) + rescale(mper_y, mper_par)

output = pd.DataFrame(yhat, index = df_river.index[30:])
output.rename(columns = {0: 'yhat'}, inplace = True)
output['y'] = y

noisebands = md.noise_variation(output.yhat, ngroups, nbounds,
                                name = 'yhat')

output['highband'] = noisebands['highband']
output['lowband'] = noisebands['lowband']

output['']

# %% Plot

dp.fast_df_visualization(output)
dp.interactive_df_visualization(output)

dp.interactive_df_visualization(output, xlab='Days', ylab='River water level [m]', file = r'D:\Users\colompa\Documents\KWR_project\Spyder_project\plots\river1dayahead.html')

#With confidence interval
output = output.dropna(subset = ['highband', 'lowband'])

import plotly.graph_objects as go
from plotly.offline import plot

fig = go.Figure()
fig.add_trace(go.Scatter(x=output.index, y = output.highband.values,
              fill = None,
              mode = 'lines',
              line_color = 'indigo',
              ))
fig.add_trace(go.Scatter(x=output.index, y = output.lowband.values,
              fill = 'tonexty',
              mode = 'lines',
              line_color = 'indigo',
              ))
fig.add_trace(go.Scatter(x=output.index, y = output.yhat.values,
              fill = None,
              mode = 'lines',
              line_color = 'red',
              ))
plot(fig)




# %% Evaluate
#R2 square between the prediction and the observations
from sklearn.metrics import r2_score

r2 = r2_score(output['y'], output['yhat'])
print(f'The R2 score of the 14 days prediction is {r2}')


