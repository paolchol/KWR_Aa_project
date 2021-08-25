# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:48:38 2021

@author: colompa
"""


log_SSA_values = {}


class components_SSA():
    def __init__(self, num, name, trend, Yper, Mper, Nstart, Nend):
        self.num = num
        self.name = name
        self.trend = trend
        self.Yper = Yper
        self.Mper = Mper
        self.Nstart = Nstart
        self.Nend = Nend


cc = components_SSA(1 , 'cc' , 0 , [1,2] , [3,4] , 5 , 365)
cc.trend

loggers.columns[0]


log_SSA_values[cc.num] = cc

log_SSA_values[1].trend
