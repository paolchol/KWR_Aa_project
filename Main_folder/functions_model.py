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
    def __init__(self, name, neurons = 50, epochs = 50, batch_size = 72):
        self.name = name
        self.neurons = neurons
        self.epochs = epochs
        self.batch = batch_size

# %% Functions

def fast_df_visualization(df):
    from matplotlib import pyplot
    pyplot.figure()
    for i, column in enumerate(df.columns, start = 1):
    	pyplot.subplot(len(df.columns), 1, i)
    	pyplot.plot(df[column].values)
    	pyplot.title(column, y = 0.5, loc = 'right')
    pyplot.show()

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

def matrix_processing(df, train, n_features, lag_in = 1, lag_out = 1, y_loc = 0):
    #df: df containing y and the matrix of features
    #train: % of the time series for the train dataset (e.g. 0.7)
    #n_features: number of features in df, including the y variable
    #y_loc: position of the variable to predict in df (default: 0)
    
    val = df.values
    val = val.astype('float32')
    
    #Split the dataset in train and test
    n = len(val)
    train_df = val[0:int(n*train), :]
    test_df = val[int(n*train):, :]
    
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

def fit_model(train_X, test_X, train_y, test_y, m_par):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from matplotlib import pyplot
    
    # Design network
    model = Sequential()
    model.add(LSTM(m_par.neurons, input_shape = (train_X.shape[1], train_X.shape[2])))
    model.add(Dense(train_y.shape[1]))
    model.compile(loss = 'mae', optimizer = 'adam')
    #Fit network
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
