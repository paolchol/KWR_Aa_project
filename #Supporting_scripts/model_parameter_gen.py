# -*- coding: utf-8 -*-
"""
# Generation of 'model_parameters.p' #

@author: colompa
"""

# %% Modules and classes

import pickle as pkl

from functions_model import single_par
from functions_model import model_par

# %% Load necessary information

#Load the SSA components and the information on which one to use for each station
SSA_information = pkl.load(open('SSA_information.p', 'rb'))

# %% First generation
#Just generate the dictionary containing the classes for each station's
#component

parameters = {}
for i in range(len(SSA_information)):
    trend = single_par(SSA_information[i].name, 50, 1000, 72)
    yper = single_par(SSA_information[i].name, 50, 1000, 72)
    mper = single_par(SSA_information[i].name, 50, 1000, 72)
    noise = single_par(SSA_information[i].name, 50, 1000, 72)
    parameters[i] = model_par(SSA_information[i].name, trend, yper, mper, noise)

#The parameters will have to be changed through the model's tuning

# %% Save the model parameters

pkl.dump(parameters, open('model_parameters.p', 'wb'))

#To load:
# m_par = pkl.load(open('model_parameters.p', 'rb'))

# %% How to modify the parameters

parameters[0].trend.batch = 72
# 0: number of the station
# .trend/.yper/.mper/.noise: SSA component of the station
# .name/ecc.: actual parameter of the model

