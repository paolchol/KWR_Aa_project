# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:11:24 2021

@author: colompa
"""

import pickle as pkl

pkl.dump(Fs_sel, open('ssa_value.p', 'wb'))

fs_temp = pkl.load(open('ssa_value.p', 'rb'))