# -*- coding: utf-8 -*-
"""
First draft of the model
Lines of code taken from 

@author: colompa
"""

#%% Setup
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

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

#%% Dataset operations

# Load the data
## Groundwater table depth's SSA components
## Weather data

# Create the station's datasets
## Load the closest-station-relationship between GW and Weather stations
#One dataset for each station

df = create_dataframe()

#use create_dataframe on each station
#the code below will be then included in a function, which will need only the
#df returned by create_dataframe as an input

"""
Here ask XIN:
The tutorial puts the 'important' frequencies directly in the features matrix
We instead said to use the model on the components, and not directly on the 
original time series

Perform a FFT on the series, to identify the 'important' frequencies
"""

# Split the data
#70% training, 20% validation, 10% test
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

# Standardize the data
#Subtract the mean and divide by the standard deviation
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

#%% Data windowing
#We want to use the last year (?) data to predict the future two weeks

OUT_STEPS = 14 #14 days = 2 weeks
multi_window = WindowGenerator(input_width = 365,
                               label_width = OUT_STEPS,
                               shift = OUT_STEPS,
                               train_df, val_df, test_df)
multi_window.plot()
multi_window


#%% Baseline creation
#A "single shot" linear model will be used as the baseline to compare the LSTM model result
#A "single shot" model makes the entire sequence prediction in a single step

MAX_EPOCHS = 20

multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

linear_history = compile_and_fit(multi_linear_model, multi_window, MAX_EPOCHS)

IPython.display.clear_output()
multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_linear_model)



#%% LSTM multi-step model


num_features = df.shape[1]
feedback_model = FeedBack(units = 32, out_steps = OUT_STEPS, num_features = num_features)



prediction, state = feedback_model.warmup(multi_window.example[0])
prediction.shape

print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)

history = compile_and_fit(feedback_model, multi_window, MAX_EPOCHS)

IPython.display.clear_output()

multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(feedback_model)


#%% Performance

#%% Validation


