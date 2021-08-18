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
bound = pd.qcut(df.val, 10 , labels = False, retbins = True)[1]

#Print the groups obtained
for key, grp in groups:
    print(f'\nGroup: {key}\n{grp}')

#Print the boundaries
# bound[0] is the minimum of the whole series, while bound[9] is the maximum
print(bound)

# %% Noise description

#Check correlation
groups.corr()

#Visualize the histograms
groups.hist()

#Extract descriptory values
gmean = groups.mean()['noise']
gmin = groups.min()['noise']
gmax = groups.max()['noise']
gsd = groups.std()['noise']


