#!/usr/bin/python	
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:46:36 2014

@author: andreabertana
"""

import itertools
no_zero_couples=np.zeros(45)
n=0
for i in it.combinations(range(10),2):
#    corr = ds_all[i[0]] - ds_all[i[1]]
    no_zero_couples[n]=len(np.nonzero(np.sum(ds_all[0].samples - ds_all[1].samples,0))[0]) - 1000
    n=n+1