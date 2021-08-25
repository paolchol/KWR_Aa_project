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

#%% Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import time
from class_SSA import SSA

#%% Functions and class

def plot_Wcorr_Wzomm(SSA_object, name, num = 0):
    if(num == 0):
        SSA_object.plot_wcorr()
        plt.title(f"W-Correlation for {name} - 365 days window")
    else:
        SSA_object.plot_wcorr(max = num)
        plt.title(f"W-Correlation for {name} - 365 days window \n- Zoomed at the first {num + 1} components")

 
def plot_SSA_results(SSA_object, Fs, noise = 0, label = 'SSA results', file = 'temp.html', final = False,
                     xlab = "Date", ylab = "Water table level [MAMSL]"):
    import plotly.express as px
    from plotly.offline import plot
    #import plotly.graph_objs as go
    
    df = pd.DataFrame({'Original': SSA_object.orig_TS.values})
    df.index = SSA_object.orig_TS.index
    if(noise != 0):
        df['Noise'] = SSA_object.reconstruct(noise).values
    if(final):
        names = ['Trend', 'Periodicity']
        for i, F in enumerate(Fs):
            name = names[i]
            df[name] = SSA_object.reconstruct(F).values
    else:
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
    figure.update_layout(
        xaxis_title = xlab,
        yaxis_title = ylab,
        legend_title = "Variables"
        )
    plot(figure, filename = file)
    return(df)

class components_SSA():
    def __init__(self, num, name, trend, Yper, Mper, Nstart, Nend):
        self.num = num
        self.name = name
        self.trend = trend
        self.Yper = Yper
        self.Mper = Mper
        self.Nstart = Nstart
        self.Nend = Nend


#%% Setup
#Load the stations
path = r'D:\Users\colompa\Documents\KWR_Internship\Data\logger_dataset\logger_GW_noNA_365.csv'
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
#It takes 14 minutes to generate the SSA for all the stations

#Path to the plot folder
path_plot = r'D:\Users\colompa\Documents\KWR_project\Spyder_project\plots'

#Creation of the results dictionary
SSA_results = {}

#%% Station 1
name = 'log_ssa1'
# for num in [0, 49, 9]:
#     plot_Wcorr_Wzomm(dSSA[name], name, num)
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = [0, 1]
F1 = 2
F2 = 3
F3 = [4, 5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')

#Decide which components are the trend and the periodicity, and place the
#remaining ones in the noise
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F1, F0]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

#Save the results in a dictionary
results = components_SSA(0, loggers.columns[0], 2, [0,1], 3, 4, 365)
SSA_results[results.num] = results

#%% Station 2
name = 'log_ssa2'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4]
F3 = [5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(1, loggers.columns[1], 0, [1,2], [3,4], 5, 365)
SSA_results[results.num] = results

#%% Station 3
name = 'log_ssa3'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = 3
F3 = [4, 5]
F4 = 6
F5 = 7
F6 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5, F6]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(2, loggers.columns[2], 0, [1,2], 3, 4, 365)
SSA_results[results.num] = results

#%% Station 4
name = 'log_ssa4'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4]
F3 = [5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(3, loggers.columns[3], 0, [1,2], [3, 4], 5, 365)
SSA_results[results.num] = results

#%% Station 5
name = 'log_ssa5'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = 3
F3 = [4, 5]
F4 = 6
F5 = 7
F6 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5, F6]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(4, loggers.columns[4], 0, [1,2], 3, 4, 365)
SSA_results[results.num] = results

#%% Station 6
name = 'log_ssa6'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4]
F3 = [5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(5, loggers.columns[5], 0, [1,2], [3, 4], 5, 365)
SSA_results[results.num] = results

#%% Station 7
name = 'log_ssa7'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4]
F3 = [5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(6, loggers.columns[6], 0, [1,2], [3, 4], 5, 365)
SSA_results[results.num] = results

#%% Station 8
name = 'log_ssa8'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4]
F3 = [5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(7, loggers.columns[7], 0, [1,2], [3, 4], 5, 365)
SSA_results[results.num] = results

#%% Station 9
name = 'log_ssa9'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4]
F3 = [5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(8, loggers.columns[8], 0, [1,2], [3, 4], 5, 365)
SSA_results[results.num] = results

#%% Station 10
name = 'log_ssa10'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4, 5, 6]
F3 = 7
F4 = [8, 9]
Fs = [F0, F1, F2, F3, F4]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(9, loggers.columns[9], 0, [1,2], [3, 4, 5, 6], 7, 365)
SSA_results[results.num] = results

#%% Station 11
name = 'log_ssa11'
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
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(10, loggers.columns[10], 0, [1,2], [3, 4], 5, 365)
SSA_results[results.num] = results

#%% Station 12
name = 'log_ssa12'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4]
F3 = [5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(11, loggers.columns[11], 0, [1,2], [3, 4], 5, 365)
SSA_results[results.num] = results

#%% Station 13
name = 'log_ssa13'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4, 5, 6]
F3 = 7
F4 = [8, 9]
Fs = [F0, F1, F2, F3, F4]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(12, loggers.columns[12], 0, [1,2], [3, 4, 5, 6], 7, 365)
SSA_results[results.num] = results

#%% Station 14
name = 'log_ssa14'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4, 5, 6]
F3 = 7
F4 = [8, 9]
Fs = [F0, F1, F2, F3, F4]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(13, loggers.columns[13], 0, [1,2], [3, 4, 5, 6], 7, 365)
SSA_results[results.num] = results

#%% Station 15
name = 'log_ssa15'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4]
F3 = [5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(14, loggers.columns[14], 0, [1,2], [3, 4], 5, 365)
SSA_results[results.num] = results

#%% Station 16
name = 'log_ssa16'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4] 
F3 = [5, 6, 7]
F4 = [8, 9]
Fs = [F0, F1, F2, F3, F4]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(15, loggers.columns[15], 0, [1,2], [3, 4], 5, 365)
SSA_results[results.num] = results

#%% Station 17
name = 'log_ssa17'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4] 
F3 = [5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(16, loggers.columns[16], 0, [1,2], [3, 4], 5, 365)
SSA_results[results.num] = results

#%% Station 18
name = 'log_ssa18'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = 3
F3 = [4, 5]
F4 = [6, 7]
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(17, loggers.columns[17], 0, [1,2], 3, 4, 365)
SSA_results[results.num] = results

#%% Station 19
name = 'log_ssa19'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = [3, 4] 
F3 = [5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(18, loggers.columns[18], 0, [1,2], [3, 4], 5, 365)
SSA_results[results.num] = results

#%% Station 20
name = 'log_ssa20'
plot_Wcorr_Wzomm(dSSA[name], name)
plot_Wcorr_Wzomm(dSSA[name], name, 49)
plot_Wcorr_Wzomm(dSSA[name], name, 9)
## Group the first 10 elementary components
F0 = 0
F1 = [1, 2]
F2 = 3
F3 = [4, 5, 6]
F4 = 7
F5 = [8, 9]
Fs = [F0, F1, F2, F3, F4, F5]
plot_SSA_results(dSSA[name], Fs, label = f'{name} - SSA', file=f'{path_plot}\{name}_comp.html')
#First component in Fs_sel has to be the trend, the second has to be the periodicity
Fs_sel = [F0, F1]
Noise = slice(3, 365)
plot_SSA_results(dSSA[name], Fs_sel, noise = Noise, label = f'{name} - Decomposed',
                 file = f'{path_plot}\{name}_final.html', final = True)

results = components_SSA(19, loggers.columns[19], 0, [1,2], 3, 4, 365)
SSA_results[results.num] = results


#%% Save the results

import pickle as pkl

pkl.dump(SSA_results, open('SSA_components.p', 'wb'))

gg = pkl.load(open('SSA_components.p', 'rb'))

pkl.dump(gg, open('SSA_information.p', 'wb'))



