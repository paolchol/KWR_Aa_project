# -*- coding: utf-8 -*-
"""
# Generation of 'SSA_information.p' #

Just a summed up script to generate the file 'SSA_information.p', without
all the procedures needed to obtain the components of the SSA, which can be found
in SSA_script.py, in the 'Supporting scripts' folder

@author: colompa
"""

from functions_dp import components_SSA
import pickle as pkl

df_log = pkl.load(open('df_log_dataframe.p', 'rb'))

SSA_results = {}
SSA_results[0] = components_SSA(0, df_log.columns[0], 2, [0,1], 3, 4, 365)
SSA_results[1] = components_SSA(1, df_log.columns[1], 0, [1,2], [3,4], 5, 365)
SSA_results[2] = components_SSA(2, df_log.columns[2], 0, [1,2], 3, 4, 365)
SSA_results[3] = components_SSA(3, df_log.columns[3], 0, [1,2], [3, 4], 5, 365)
SSA_results[4] = components_SSA(4, df_log.columns[4], 0, [1,2], 3, 4, 365)
SSA_results[5] = components_SSA(5, df_log.columns[5], 0, [1,2], [3, 4], 5, 365)
SSA_results[6] = components_SSA(6, df_log.columns[6], 0, [1,2], [3, 4], 5, 365)
SSA_results[7] = components_SSA(7, df_log.columns[7], 0, [1,2], [3, 4], 5, 365)
SSA_results[8] = components_SSA(8, df_log.columns[8], 0, [1,2], [3, 4], 5, 365)
SSA_results[9] = components_SSA(9, df_log.columns[9], 0, [1,2], [3, 4, 5, 6], 7, 365)
SSA_results[10] = components_SSA(10, df_log.columns[10], 0, [1,2], [3, 4], 5, 365)
SSA_results[11] = components_SSA(11, df_log.columns[11], 0, [1,2], [3, 4], 5, 365)
SSA_results[12] = components_SSA(12, df_log.columns[12], 0, [1,2], [3, 4, 5, 6], 7, 365)
SSA_results[13] = components_SSA(13, df_log.columns[13], 0, [1,2], [3, 4, 5, 6], 7, 365)
SSA_results[14] = components_SSA(14, df_log.columns[14], 0, [1,2], [3, 4], 5, 365)
SSA_results[15] = components_SSA(15, df_log.columns[15], 0, [1,2], [3, 4], 5, 365)
SSA_results[16] = components_SSA(16, df_log.columns[16], 0, [1,2], [3, 4], 5, 365)
SSA_results[17] = components_SSA(17, df_log.columns[17], 0, [1,2], 3, 4, 365)
SSA_results[18] = components_SSA(18, df_log.columns[18], 0, [1,2], [3, 4], 5, 365)
SSA_results[19] = components_SSA(19, df_log.columns[19], 0, [1,2], 3, 4, 365)

pkl.dump(SSA_results, open('SSA_information.p', 'wb'))


























