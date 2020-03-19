# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:57:50 2014

@author: andreabertana
"""

import numpy as np
import matplotlib.pyplot as plt

#define general path
path = '/home/andreabertana/Projects/HCP/results/MNI/MOTOR/'
#open files
all_10 = np.loadtxt(path + 'results/all_acc_10.txt')
all_20 = np.loadtxt(path + 'results/all_acc_20.txt')
all_30 = np.loadtxt(path + 'results/all_acc_30.txt')

#define path to save
path_save = path + 'plot/acc_oversubjects.png'

#join the accuracy in one object
all_trainsets = np.concatenate((all_10,all_20,all_30), axis=1)
#reshape it
all_trainsets= all_trainsets.reshape(3,9)

#plot each raw independently
line1, = plt.plot(all_trainsets[0,:], '-o')
line2, = plt.plot(all_trainsets[1,:], '-o')
line3, = plt.plot(all_trainsets[2,:], '-o')

#define x and y axis
plt.xlim(-1,9)
plt.xticks([0,1,2,3,4,5,6,7,8],['10','15','20','25','30','35','40','45','50'])
plt.ylim(0.5 , 1)

#define tiltle
plt.title('The effect of groups size on classification')

#define labels
plt.xlabel('Group size for bulding commonspace')
plt.ylabel('Acc%')

#define legend
plt.legend( (line1, line2, line3), ('trainset = 10', 'trainset = 20', 'trainset = 30') )  

#plt.saveit
plt.savefig(path_save)

#plt.show
plt.show()
