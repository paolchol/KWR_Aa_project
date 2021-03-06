# Project @ KWR, 2021: Deep learning model for hydrological values prediction
# Still in progress
## Author: Paolo Colombo, Supervisor: Xin Tian

Python scripts to develop a deep learning model to forecast hydrological variables in the Drenthe province, The Netherlands.

The model will be used to predict, for 2 weeks ahead:
  - Water table depth in Aa river's region
  - Aa river's discharge

The main code is organized in **Main folder** as written below:
  - *main_GW_dp.py*: data pre-processing procedures to clean the groundwater (GW) data and obtain 20 stations with a good amount of data where to predict the water table depth. Then, SSA time series decomposition to obtain the important components of the time series
  - *main_R_dp.py*: data pre-processing procedures applied to the river (R) data. Then, SSA time series decomposition to obtain the important components of the time series.
  - *main_model.py*: model construction

Also, other folders are present:
  - *Supporting_scripts*: experiments doen that were crucial in defining information and data used in the main scripts
  - *non_organized_and_old_scripts*: various drafts and experiments done throughout the project progresses, kept just in case
  - *Code_examples*: other codes/projects that have been useful in the study phase


*All the sources of pieces of code not written directly by me are cited in the scripts.*
