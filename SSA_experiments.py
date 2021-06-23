# -*- coding: utf-8 -*-
"""
SSA Time series decomposition

    - Load the loggers data
    - Identify trend, periodicity and noise for each station
    - Create dataframes containing them
    - Add the weather data to these dataframes


@author: colompa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SSA_class import SSA

path = r'D:\Users\colompa\Documents\KWR_Internship\Data\after_data_prep\logger_GW_noNA.csv'
loggers = pd.read_csv(path, sep = ',', index_col = 0)

# Remove starting and ending NAs
#Create a series

#This code below will have to be a loop afterwards
logg = loggers.iloc[:,0][loggers.iloc[:,0].notna()]
logg.plot()


#Give the series to the SSA class
L = 365 #1 year window length
log_ssa = SSA(logg, L)

#Plot the w-correlation matrix to help orient ourselves with the separated components
log_ssa.plot_wcorr()
plt.title("W-Correlation for Loggers[,0] - 365 days window")

#Zoom the w-correlation matrix for the first 50 components
log_ssa.plot_wcorr(max = 49)
plt.title("W-Correlation for Loggers[,0] - 365 days window - Zoomed")

# Start grouping the first 7 elementary components, form the following groups
# and plot them
#   - F(0) = F0 + F1
#   - F(1) = F2
#   - F(2) = F3 + F4
#   - F(3) = F5 + F6

log_ssa.reconstruct([0,1]).plot(alpha = 0.5)
log_ssa.reconstruct(2).plot(alpha = 0.5)
log_ssa.reconstruct([3,4]).plot(alpha = 0.3)
log_ssa.reconstruct([5,6]).plot(alpha = 0.3)
log_ssa.orig_TS.plot(alpha = 0.4)
plt.title("Loggers[,0] - First 4 groups")
plt.xlabel("Date")
plt.ylabel("Level [MASL - Meters Above Sea Level]")
legend = [r'$\tilde{{F}}^{{({0})}}$'.format(i) for i in range(4)] + ["Original TS"]
plt.legend(legend)

#Zoom at a 5 year subseries (approximately)
start = 365*2           #around 2009
end = start + 365     #around 2014
log_ssa.reconstruct([0,1]).plot(alpha = 0.5)
log_ssa.reconstruct(2).plot(alpha = 0.5)
log_ssa.reconstruct([3,4]).plot(alpha = 0.3)
log_ssa.reconstruct([5,6]).plot(alpha = 0.3)
log_ssa.orig_TS.plot(alpha = 0.4)
plt.title("Loggers[,0] - First 4 groups - Zoom 1 year")
plt.xlabel("Date")
plt.ylabel("Level [MASL - Meters Above Sea Level]")
plt.xlim(start, end)
legend = [r'$\tilde{{F}}^{{({0})}}$'.format(i) for i in range(4)] + ["Original TS"]
plt.legend(legend)

#We can see that F(1) can be the trend, while F(0) is a dominant periodicity
#and F(2), F(3) are two high frequency periodicities, F(3) having a lower periodicity

#Visualize the first 7 components added together
log_ssa.reconstruct(slice(0,6)).plot(alpha = 0.5)
log_ssa.orig_TS.plot(alpha = 0.3)
plt.title('Firts 7 components')
plt.xlim(start, end)
#We can see the low frequency periodicity (F(0)), which could be the yearly periodicity,
#with higher frequency periodicities influencing it (F(2) and F(3)). F(2) could be the daily
#periodicity (it has an higher frequency than F(3)) and F(3) could be the monthly one.

#Let's try going ahead and group other elementary components and plot the groups
#   - F(4) = F7 + F8
#   - F(5) = F9 + F10 + F11 + F12 + F13
#   - F(6) = F14
#   - F(7) = F15 + F16
#Plot
log_ssa.reconstruct([7,8]).plot(alpha = 0.5)
log_ssa.reconstruct([9,10,11,12,13]).plot(alpha = 0.5)
log_ssa.reconstruct(14).plot(alpha = 0.5)
log_ssa.reconstruct([15,16]).plot(alpha = 0.3)
log_ssa.orig_TS.plot(alpha = 0.4)
plt.title("Loggers[,0] - Groups from 5 to 8")
plt.xlabel("Date")
plt.ylabel("Level [MASL - Meters Above Sea Level]")
legend = [r'$\tilde{{F}}^{{({0})}}$'.format(i) for i in range(4,8)] + ["Original TS"]
plt.legend(legend)
#Zoom to one year only
#With the components shifted to visualize them better
start = 365*2       #around 2009
end = start + 365   #around 2010
(log_ssa.reconstruct([7,8]) - 0.25).plot(alpha = 0.5)
(log_ssa.reconstruct([9,10,11,12,13]) - 0.5).plot(alpha = 0.5)
(log_ssa.reconstruct(14) - 0.75).plot(alpha = 0.5)
(log_ssa.reconstruct([15,16]) - 1).plot(alpha = 0.3)
log_ssa.orig_TS.plot(alpha = 0.4)
plt.title("Loggers[,0] - Groups from 5 to 8 - Zoom 1 year")
plt.xlabel("Date")
plt.ylabel("Level [MASL - Meters Above Sea Level]")
plt.xlim(start, end)
legend = [r'$\tilde{{F}}^{{({0})}}$'.format(i) for i in range(4,8)] + ["Original TS"]
plt.legend(legend)
#Groups F(4), F(5), F(6) and F(7) are new periodicities, with amplitudes decreasing
#after F(5).
#F(7) also presents a trend

#OBSERVATION (from D'Arcy, 2018, Kaggle)
# In time series data with a lot of high-frequency components,
# SSA will typically generate a number of stable harmonic components that are
# well-separated from each other (i.e. low w-correlation). However, without some
# domain knowledge of the process generating the time series itself,
# it is difficult to say whether these components correspond to interpretable
# processes that operate independently, or if the set of components should be
# summed and treated as a single component. SSA is blind to reality.

#Let's try to interpret the groups with the new informations:
#   - F(0): yearly periodicity
#   - F(1): overall trend
#   - F(2): weekly periodicity
#   - F(3): monthly periodicity
#   - F(4): daily periodicity
#   - F(5): the amplitude seems to follow the yearly periodicity
#       (looking at the non-zoomed graph)
#   - F(6): weekly periodicity
#   - F(7): weekly periodicity, but with a yearly "trend", try to check the
#       F15 and F16 elementary components separately?

#Combine all these groups' elementary components (17 in total) together and
#visualize them compared to the remaining ones (348 in total)
start = 365*2       #around 2009
end = start + 365   #around 2010
log_ssa.reconstruct(slice(0,17)).plot(alpha = 0.5)
(log_ssa.reconstruct(slice(17,365)) - 1.5).plot(alpha = 0.5)
log_ssa.orig_TS.plot(alpha = 0.3)
plt.legend(['First 17 Components', 'Remaining 348 Components', 'Original TS'])
plt.xlim(start, end)
#The remaining 346 components still present some periodicities, with gaps that
#seems related to the months, and some periodic 'coupled peaks'

#Let's try to group the first 40 elementary components
#   - F(8): F17 + F18
#   - F(9): F19 + F20
#   - F(10): F21 + F22
#   - F(11): F23 + F27
#   - F(12): F24 + 25
#   - F(13): F26
#   - F(14): F28 + F29
#   - F(15): F30 + F31 + F32 + F33 + F34
#   - F(16): F35 + F36
#   - F(17): F37 + F38
#   - F(18): F39 + F40 + F41

#Combine all these groups' elementary components (42 in total) together and
#visualize them compared to the remaining ones (323 in total)
start = 365*2       #around 2009
end = start + 365   #around 2010
log_ssa.reconstruct(slice(0,42)).plot(alpha = 0.5)
(log_ssa.reconstruct(slice(42,365)) - 1.5).plot(alpha = 0.5)
log_ssa.orig_TS.plot(alpha = 0.3)
plt.legend(['First 42 Components', 'Remaining 323 Components', 'Original TS'])
plt.xlim(start, end)
#Now the monthly behavior is less evident.
#There is still a periodicity in the peaks

#Experiment in dividing trend, periodicity and noise
numbers = range(3, 43)
sequence_of_numbers = [0,1]
for number in numbers:
      sequence_of_numbers.append(number)
start = 365*2       #around 2009
end = start + 365   #around 2010
log_ssa.orig_TS.plot(alpha = 0.3)
(log_ssa.reconstruct(2) - 0.5).plot(alpha = 0.5)
(log_ssa.reconstruct(sequence_of_numbers) - 1).plot(alpha = 0.5)
(log_ssa.reconstruct(slice(42,365)) - 1.5).plot(alpha = 0.5)
plt.legend(['Original TS', 'Trend', 'Periodicity', 'Noise'])
plt.xlabel("Date")
plt.ylabel("Level [MASL - Meters Above Sea Level]")
plt.title('First experiment in separing the time series')
#In one year
plt.xlim(start, end)



#Idea for automatic grouping
#   - Explore the W-Correlation matrix
#   - Components with correlation higher than a threshold (0.6?) are placed together
#   - Create the groups containing the components with correlation higher than threshold
#       between them
#   - If some components fall into more than one group, leave them in the group
#       for which they have the highest mean correlation with the other components
#       of the group



