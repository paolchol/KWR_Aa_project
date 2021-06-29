# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:11:24 2021

@author: colompa
"""

import pickle as plk

plk.dump(Fs_sel, open('ssa_value.p', 'wb'))

fs_temp = plk.load(open('ssa_value.p', 'rb'))