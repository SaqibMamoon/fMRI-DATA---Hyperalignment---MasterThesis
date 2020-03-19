# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:57:50 2014

@author: andreabertana
"""

import numpy as np
import matplotlib.pyplot as plt


#define path
path ='/home/andreabertana/Projects/HCP/results/MNI/MOTOR/'
path_det = path + 'detrend_anova_svm_all.txt'
path_smo = path + 'smoothing_anova_svm_all.txt'
path_mc = path + 'mc_anova_svm_all.txt'

path_anat_mc = path + 'mcanat_svm_all.txt'
path_anat_det = path + 'detrendanat_svm_all.txt'
path_anat_smo = path + 'smoothinganat_svm_all.txt'
#define path to save
path_save = path + 'plot/anova_LSVM_boxplot_motor_highres_font20_width11.png'

#open txt with accuracy
acc_smo = np.loadtxt(path_smo)
acc_det = np.loadtxt(path_det) 
acc_mc = np.loadtxt(path_mc)

ana_smo = np.loadtxt(path_anat_mc)
ana_det = np.loadtxt(path_anat_det) 
ana_mc = np.loadtxt(path_anat_smo)

#avarage between the two runs' folds
data_det = np.mean(acc_det, axis =1)
data_smo = np.mean(acc_smo, axis =1)
data_mc = np.mean(acc_mc, axis =1)

#create one data for plotting
data = [ana_mc, data_mc, ana_det ,data_det,ana_smo, data_smo] 
plt.figure(figsize=(14,9))

bp = plt.boxplot(data,positions=(1,2.2,4,5.2,7,8.2))#
plt.xticks([1.6,4.6,7.6],['motion corrected','detrended','smoothed'], fontsize = 20)
plt.title('Motor task: BSC accuracy after anatomical and hyperalignment',fontsize = 24)
#set colors
for i in np.arange(0,5,2):
    plt.setp(bp['boxes'][i], color ='green',lw = 3)
    plt.setp(bp['medians'][i], color ='black',lw = 3)
   
for i in np.arange(1,6,2):
    plt.setp(bp['boxes'][i],lw = 3) 
    plt.setp(bp['medians'][i], color ='black',lw = 3)
    
#set whiskers
for i in np.arange(0,12):    
    plt.setp(bp['whiskers'][i],lw = 3, color ='grey')    
#create legend
hK, = plt.plot([1,1], 'g-', lw = 3)
hB, = plt.plot([1,1], 'b-', lw = 3)
plt.legend((hK,hB),('Anatomical','Hyperalignment'),prop={'size':20} )
hB.set_visible(False)
hK.set_visible(False)
#set labels
plt.ylabel('Acc%',fontsize = 20)
#plt.xlabel('Distribution')
#Set mediana
numBoxes = len(data)
medians = range(numBoxes)
for i in range(numBoxes):
    med = bp['medians'][i]
    medianX = []
    medianY = []
    for j in range(2):
        medianX.append(med.get_xdata()[j])
        medianY.append(med.get_ydata()[j])
        plt.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
    plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
           color='r', marker='*', markeredgecolor='r', lw=3)

#Insert * legend
plt.figtext(0.80, 0.13, '*', color='red', backgroundcolor='silver',
           weight='roman', size='large')
plt.figtext(0.815, 0.13, ' Mean', color='black', weight='roman',
           size='large')
plt.ylim(0,1.2) 
plt.yticks(fontsize = 20)
plt.grid(axis = 'y', lw =2)  
plt.savefig(path_save, transparent = True)
plt.show()