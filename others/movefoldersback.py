# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:42:08 2014

@author: andreabertana
"""

import shutil
import os
#direction = np.array(['LR/', 'RL/'])
path_dest = '/home/andreabertana/Projects/HCP/data/'


for subjname in os.listdir(path_dest):
   
    source_second = path_dest + subjname + '/tfMRI_WM_LR/tfMRI_WM_RL/'
    destination = path_dest + subjname + '/'
    shutil.move(source_second, destination)
    print destination
    print source_second
    


        


