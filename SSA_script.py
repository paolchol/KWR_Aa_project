# -*- coding: utf-8 -*-
"""
SSA time series decomposition

Goal of this script:
    - Identify the elementary components that compose the overall trend
        and the yearly periodicity of each station
    - Interactively plot trend, periodicity, noise and the original time series
    - Save the information about the grouped elementary components

@author: colompa
"""

#Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import time
from SSA_class import SSA

#Function

def plot_Wcorr_Wzomm(SSA_object, name, num = 0):
    if(num == 0):
        SSA_object.plot_wcorr()
        plt.title(f"W-Correlation for {name} - 365 days window")
    else:
        SSA_object.plot_wcorr(max = num)
        plt.title(f"W-Correlation for {name} - 365 days window \n- Zoomed at the first {num + 1} components")

 
def plot_SSA_results(SSA_object, Fs, noise = 0, label = 'SSA results', file = 'temp.html'):
    import plotly.express as px
    from plotly.offline import plot
    #import plotly.graph_objs as go
    
    df = pd.DataFrame({'Original': SSA_object.orig_TS.values})
    df.index = SSA_object.orig_TS.index
    if(noise != 0):
        df['Noise'] = SSA_object.reconstruct(noise).values
    for i, F in enumerate(Fs):
        name = f'F{i}'
        df[name] = SSA_object.reconstruct(F).values
    #Would be nicer to have the original series with a bit of transparency
    #Also to add labels to the axis and a title
    # plt.title(label)
    # plt.xlabel("Date")
    # plt.ylabel("Level [MASL - Meters Above Sea Level]")
    # Also save it with a name instead of temp-plot, and in a specified path
    figure = px.line(df)
    #figure.
    plot(figure, filename = file)
    return(df)


#Load the stations
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\after_data_prep\logger_GW_noNA.csv'
loggers = pd.read_csv(path, sep = ',', index_col = 0)

#Create the SSA class for each station
start = time.time()
L = 365 #Window length
dSSA = {}
for i in range(len(loggers.columns)):
    logg = loggers.iloc[:,i][loggers.iloc[:,i].notna()]
    dSSA["log_ssa{0}".format(i+1)] = SSA(logg, L)
end = time.time()
print(f'SSA class creation for all the stations\tElapsed time: {round((end - start)/60)} minutes')
print(end - start)

#Path to the plot folder
path_plot = r'D:\Users\colompa\Documents\KWR_project\Spyder_project\plots'

# Station 1
name = 'log_ssa1'
# for num in [0, 49, 9]:
#     plot_Wcorr_Wzomm(dSSA[name], name, num)
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = [0, 1]
F1 = 2
F2 = [3, 4]
F3 = [5, 6]
F4 = [7, 8]
F5 = [9, 10]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}.html')
#Decide which components are the trend and the periodicity, and place the
#remaining ones in the noise
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - SSA - With noise showed', file=f'{name}_noise.html')

figure = px.line(dSSA[name].orig_TS - dSSA[name].reconstruct(F0) - dSSA[name].reconstruct(F1))
plot(figure)

# Station 2
name = 'log_ssa2'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4]
F3 = [5, 6]
F4 = [7, 8, 9]
Fs = [F0, F1, F2, F3, F4]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{name}.html')
#Decide which components are the trend and the periodicity, and place the
#remaining ones in the noise
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs, noise = Noise, label = f'{name} - SSA - With noise showed', file=f'{name}_noise.html')

# Station 3
name = 'log_ssa3'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4]
F3 = [5, 6]
F4 = [7, 8]
F5 = 9
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{name}.html')
#Decide which components are the trend and the periodicity, and place the
#remaining ones in the noise
Fs_sel = [F0 ,F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs, noise = Noise, label = f'{name} - SSA - With noise showed', file=f'{name}_noise.html')

# Station 4
name = 'log_ssa3'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4]
F3 = [5, 6]
F4 = [7, 8]
F5 = 9
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{name}.html')
#Decide which components are the trend and the periodicity, and place the
#remaining ones in the noise
Fs_sel = [F0 ,F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs, noise = Noise, label = f'{name} - SSA - With noise showed', file=f'{name}_noise.html')






