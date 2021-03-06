# -*- coding: utf-8 -*-
"""
Functions for the model construction and the data preparation

List of classes:
    - model_par:
        
    - single_par:
        

List of functions:
    - fast_df_visualization:
        
    - series_to_supervised:
        Obtained from https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
        
    - matrix_processing:
        
    - fit_model:
        
    - model_predict:
        Depracated
    
    - prediction:
        

@author: colompa
"""

# %% Libraries

#General modules
import pandas as pd
import numpy as np

# %% Classes

class model_par():
    #Complete 'collection' of the parameters for each station
    def __init__(self, name, trend = 0, yper = 0, mper = 0, noise = 0):
        self.name = name
        self.trend = trend
        self.yper = yper
        self.mper = mper
        self.noise = noise

class single_par():
    def __init__(self, name, neurons = 50, epochs = 50, batch_size = 72, learn_rate = 0.001):
        self.name = name
        self.neurons = neurons
        self.epochs = epochs
        self.batch = batch_size
        self.learn_rate = learn_rate

# %% Functions

def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def matrix_processing(df, train, n_features, lag_in = 1, lag_out = 1, forward = True):
    #df: df containing y and the matrix of features
    #train: % of the time series for the train dataset (e.g. 0.7)
    #n_features: number of features in df, including the y variable
    #lag_in: window of input data to consider in the prediction
    #lag_out: window for the prediction
    #forward: True means that the train set is taken from the 
    
    val = df.values
    val = val.astype('float32')
    
    #Split the dataset in train and test
    n = len(val)
    if(forward):
        train_df = val[0:int(n*train), :]
        test_df = val[int(n*train):, :]
    else:
        train_back = 1 - train
        train_df = val[int(n*train_back):, :]
        test_df = val[0:int(n*train_back):, :]
    
    #Feature scaling - Standardization
    train_mean = np.mean(train_df, axis = 0)
    train_std = np.std(train_df, axis = 0)
    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    param = [{'mean': train_mean, 'std': train_std}]
    
    #Frame as supervised learning
    #Number of columns in the df: lag_in*n_features + lag_out*n_features 
    train_df = series_to_supervised(train_df, lag_in, lag_out)
    test_df = series_to_supervised(test_df, lag_in, lag_out)
    train_df = train_df.values
    test_df = test_df.values
    
    #Obtain X and Y dataframes
    #Select only columns we want to use for training and we want to predict
    if(lag_in == 1 and lag_out == 1):
        train_X, train_y = train_df[:, :-n_features], train_df[:, -n_features]
        test_X, test_y = test_df[:, :-n_features], test_df[:, -n_features]
    elif(lag_out == 1):
        n_obs = lag_in * n_features
        train_X, train_y = train_df[:, :n_obs], train_df[:, -n_features]
        test_X, test_y = test_df[:, :n_obs], test_df[:, -n_features]
    else:
        n_obs = lag_in * n_features
        n_pred = lag_out * n_features
        pos = [n_obs]
        for i in range(int(n_pred/n_features)):
            pos.append(pos[i] + n_features)
            if(pos[i + 1] >= (n_pred + n_obs)):
                pos.pop(i + 1)
        train_X, train_y = train_df[:, :n_obs], train_df[:, pos]
        test_X, test_y = test_df[:, :n_obs], test_df[:, pos]
    
    #Reshape input to be 3D
    train_X = train_X.reshape((train_X.shape[0], lag_in, n_features))
    test_X = test_X.reshape((test_X.shape[0], lag_in, n_features))
    #Check if also y has to be reshaped when lag_out > 1
    #https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
    #It doesn't seem like it needs to be reshaped
    return train_X, test_X, train_y, test_y, param

def fit_model(train_X, test_X, train_y, test_y, m_par, oneday = False):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from matplotlib import pyplot
    
    # Design network
    model = Sequential()
    model.add(LSTM(m_par.neurons, input_shape = (train_X.shape[1], train_X.shape[2])))
    if oneday:
        shp = 1
    else: shp = train_y.shape[1]
    model.add(Dense(shp))
    # optimizer = tf.keras.optimizers.Adam(learning_rate = m_par.learn_rate)
    model.compile(loss = 'mae', optimizer = 'adam')
    # Fit network
    history = model.fit(train_X, train_y, epochs = m_par.epochs,
                        batch_size = m_par.batch, validation_data = (test_X, test_y),
                        verbose = 2, shuffle = False)
    # Other method
    #https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
    #The LSTM layer has to be changed introducing batch_input_size
    #  for i in range(m_par.epochs):
    #       model.fit(train_X, train_y, epochs = 1, batch_size = m_par.batch, verbose = 0, shuffle = False)
    # 		model.reset_states()
    # Plot history
    pyplot.plot(history.history['loss'], label = 'train')
    pyplot.plot(history.history['val_loss'], label = 'test')
    pyplot.legend()
    pyplot.show()
    return model

def prediction(model, df, par, n_feat = 6, lag_in = 30, lag_out = 1):
    #model: fitted model
    #df: non-scaled df containing the features
    #par: parameters to perform the scaling (the same used to scale the training and test sets)
    #n_feat: number of features in df, including the y variable
    
    #Take the values and rescale them
    df_scale = df.values
    df_scale = (df_scale - par[0]['mean']) / par[0]['std']
    #Obtain the observations to make the prediction
    x = lag_in + lag_out
    subset = df_scale[-x:, :]
    
    #Check this above
    
    obs = series_to_supervised(subset, lag_in, lag_out).values
    n_obs = lag_in * n_feat
    obs = obs[:, :n_obs]
    obs = obs.reshape(obs.shape[0], lag_in, n_feat)
    #Make the prediction
    yhat = model.predict(obs)
    yhat = np.transpose(yhat)
    yhat = yhat*par[0]['std'][0] + par[0]['mean'][0]
    return yhat

## Attention:
## model_predict is deprecated, use prediction instead

def model_predict(model, test_X, test_y, ts_par):
    #https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    
    # Make prediction
    yhat = model.predict(test_X)
    
    inv_yhat = yhat*ts_par[0]['std'][0] + ts_par[0]['mean'][0]
    inv_y = test_y*ts_par[0]['std'][0] + ts_par[0]['mean'][0]
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    nrmse = round(rmse/(np.amax(inv_y) - np.amin(inv_y)), ndigits = 3)
    print('Test RMSE: %.3f' % rmse)
    print(f'Test NRMSE: {nrmse}')
    return inv_yhat, inv_y, rmse

# %% Noise classification

def noise_group(df, valcol = 0, ngroup = 10):
    #df: pandas dataframe containing the value column and the noise column
    #valcol: position of the value column
    #ngroup: number of groups to be created (number of quantiles to consider)
    
    df.columns = ['val', 'noise'] if valcol == 0 else ['noise', 'val']
    groups = df.groupby(pd.qcut(df.val, ngroup, labels = False))
    bounds = pd.qcut(df.val, ngroup, labels = False, retbins = True)[1]
    return groups, bounds

# single_noise_variation doesn't really work, modify
def single_noise_variation(pred, groups, bounds):
    i = 0
    for bound in bounds:
        if(pred >= bound): classs = i
        i += 1
    gmean = groups.mean()['noise']
    gstd = groups.std()['noise']
    maxnoise = pd.DataFrame(gmean + 3*gstd)
    
    highval = pred + maxnoise.iloc[classs, 0]
    lowval = pred - maxnoise.iloc[classs, 0]
    bands = pd.DataFrame[{'highval': highval, 'lowval': lowval}]   
    return classs, bands

def noise_variation(pred, groups, bounds, single = False):
    #pred: Series containing the prediction
    #groups: variable and noise grouped
    #bounds: boundaries of the groups
    
    if(single): return single_noise_variation(pred, groups, bounds)
    
    classes = pd.DataFrame(pd.cut(pred, bounds, labels = False))
    gmean = groups.mean()['noise']
    gstd = groups.std()['noise']
    maxnoise = pd.DataFrame({'maxnoise': gmean + 3*gstd})
    minnoise = pd.DataFrame({'minnoise': gmean - 3*gstd})
    
    bands = classes.join(maxnoise, on = 'val')
    bands = bands.join(minnoise, on = 'val')
    
    #Is this correct??
    bands['highband'] = pred + bands['maxnoise']
    bands['lowband'] = pred + bands['minnoise']
    bands['val'] = pred
    bands.drop(['maxnoise', 'minnoise'], 1, inplace = True)
    
    return bands

# %% Sum-up function

def launch(df, m_par, code):
    if(code == 't'): par = m_par.trend
    elif(code == 'yp'): par = m_par.yper
    elif(code == 'mp'): par = m_par.mper
    elif(code == 'n'): par = m_par.noise
    else:
        print(f'The inserted code {code} is not correct')
        return
        
    train_X, test_X, train_y, test_y, ts_par = matrix_processing(df, 0.7, 3, lag_in = 30, lag_out = 14)
    fitted = fit_model(train_X, test_X, train_y, test_y, par)
    yhat, y, rmse = model_predict(fitted, test_X, test_y, ts_par)
    
    return yhat, y
