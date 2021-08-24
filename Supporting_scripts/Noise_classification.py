# -*- coding: utf-8 -*-
"""
Script to handle the noise
    - Classification of trend + periodicities: grouping
    - Extraction of the noise distribution for each group

@author: colompa
"""

# %% Setup

#General modules
import pandas as pd
import numpy as np
import pickle as pkl

#Visualization
# from matplotlib import pyplot

#Custom modules
import functions_dp as dp
from functions_dp import components_SSA
from class_SSA import SSA
import functions_model as md
from functions_model import model_par
from functions_model import single_par

# %% Load and generate the dataframe to group

#River SSA results
SSA_river = pkl.load(open('SSA_level.p', 'rb'))
SSA_information = pkl.load(open('SSA_level_info.p', 'rb'))

#Extract the components
trend = dp.extract_attributes(SSA_river, SSA_information, 't')
yper = dp.extract_attributes(SSA_river, SSA_information, 'yp')
mper = dp.extract_attributes(SSA_river, SSA_information, 'mp')
noise = dp.extract_attributes(SSA_river, SSA_information, 'n')

#Dataframe
val = trend + yper + mper
df = pd.DataFrame({'val': val, 'noise': noise})

# %% Group

#Create the groups and extract the boundaries
groups = df.groupby(pd.qcut(df.val, 10 , labels = False))
bounds = pd.qcut(df.val, 10 , labels = False, retbins = True)[1]

#Print the groups obtained
for key, grp in groups:
    print(f'\nGroup: {key}\n{grp}')

#Print the boundaries
# bounds[0] is the minimum of the whole series, while bound[9] is the maximum
print(bounds)

# %% Noise description

#Check correlation
groups.corr()

#Visualize the histograms
groups.hist()

#Extract descriptory values
gmean = groups.mean()['noise']
gmin = groups.min()['noise']
gmax = groups.max()['noise']
gstd = groups.std()['noise']

# %% Functions

def noise_group(df, valcol = 0, ngroup = 10):
    #df: pandas dataframe containing the value column and the noise column
    #valcol: position of the value column
    #ngroup: number of groups to be created (number of quantiles to consider)
    
    df.columns = ['val', 'noise'] if valcol == 0 else ['noise', 'val']
    groups = df.groupby(pd.qcut(df.val, ngroup, labels = False))
    bounds = pd.qcut(df.val, ngroup, labels = False, retbins = True)[1]
    return groups, bounds

#This function doesn't work yet
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

def noise_variation(pred, groups, bounds, coeff = 1, single = False):
    #pred: Series containing the prediction
    #groups: variable and noise grouped
    #bounds: boundaries of the groups
    
    if(single): return single_noise_variation(pred, groups, bounds)
    
    classes = pd.DataFrame(pd.cut(pred, bounds, labels = False))
    gmean = groups.mean()['noise']
    gstd = groups.std()['noise']
    maxnoise = pd.DataFrame({'maxnoise': gmean + coeff*gstd})
    minnoise = pd.DataFrame({'minnoise': gmean - coeff*gstd})
    
    bands = classes.join(maxnoise, on = 'val')
    bands = bands.join(minnoise, on = 'val')
    bands['highband'] = pred + bands['maxnoise']
    bands['lowband'] = pred + bands['minnoise'] #bands['noise']
    bands['val'] = pred
    bands.drop(['maxnoise', 'minnoise'], 1, inplace = True)
    
    return bands

# %% To use the functions

groups, bands = noise_group(df)
pred = df.val
bands = noise_variation(pred, groups, bounds)
dp.interactive_df_visualization(bands)

