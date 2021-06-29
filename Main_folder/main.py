# -*- coding: utf-8 -*-
"""
Project:
    Development of a deep learning model in order to predict the water table
    depth in the Drenthe province, The Netherlands

Data available:
    - Water table depths
    - Precipitation
    - Evaporation
    - River discharge

Development:
    # Data preprocessing: #
        - Water table depth data is in hourly and sub-hourly configuration,
            it needs to be resampled in a daily configuration
        - The water table depth stations with a good amount of consecutive
            recent data have to be selected and extracted
        - These extracted stations have to be cleaned from outliers and
            completed if missing data is present 
        - The nearest weather station to each of the selected water table depth
            station has to be selected
        ** Result **
            The result of this phase will be a dataframe for each water table 
            depth point. It will contain the water table depth time series, the
            precipitation time series and the evaporation time series.
            
    # SSA: #
        - To improve the model performance, the water table depth has to be
            divided in three components:
            overall trend, yearly periodicity and noise.
            They will be given to the model separately, and then reconstructed
        - The SSA is performed for each station, and the selection of the
            components will be done manually
        ** Result **
            The result of this phase will be a dataframe containing the index
            of the elementary components of each water table depth station
            identifying trend, periodicity and noise
    
    # Model construction: #
        - Separate script
    # Model testing: #
        - Separate script
    # Model utilization: #
        - 


@author: colompa
"""
