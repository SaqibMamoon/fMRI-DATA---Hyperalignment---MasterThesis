# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:57:50 2014

@author: andreabertana
"""

import numpy as np
import matplotlib.pyplot as plt


#define path
path ='/home/andreabertana/Projects/HCP/results/MNI/WM/results/distri'
path_det = path + 'detrend_anova_svm_all.txt'
path_smo = path + 'smoothing_anova_svm_all.txt'
path_mc = path + 'mc_anova_svm_all.txt'

path_anat_mc = path + 'mcanat_svm_all.txt'
path_anat_det = path + 'detrendanat_svm_all.txt'
path_anat_smo = path + 'smoothinganat_svm_all.txt'
#define path to save
path_save = path + 'plot/anova_LSVM_boxplot.png'

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

#Insert some indices


#create one data for plotting
data = [ana_mc, data_mc, ana_det ,data_det,ana_smo, data_smo]

#plot
fig, ax1 = plt.subplots(figsize=(14,9))
fig.canvas.set_window_title('Anova_LSVM')
plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = plt.boxplot(data,positions=(1,2.2,4,5.2,7,8.2))#notch=0, sym='+', vert=1, whis=1.5)
plt.xticks([1,2,4,5,7,8],['ana_mc','hyper_mc','ana_det','hyper_det','ana_smo', 'hyper_smo'])
# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)

# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('WM task: BSC accuracy after anatomical and hyperalignment',fontsize = 24)
ax1.set_ylabel('Acc%',fontsize = 20)

#insert color to boxplot of hyperalinment 
#for i in np.arange(np.arange(0,6,2)):
    # Now fill the boxes with desired colors
boxColors = ['royalblue', 'y']
numBoxes = len(data)
medians = range(numBoxes)
for i in range(numBoxes):
  box = bp['boxes'][i]
  boxX = []
  boxY = []
  for j in range(5):
      boxX.append(box.get_xdata()[j])
      boxY.append(box.get_ydata()[j])
  boxCoords = zip(boxX,boxY)
  # Alternate between Dark Khaki and Royal Blue
  k = i % 2
  boxPolygon = plt.Polygon(boxCoords, facecolor=boxColors[k])
  # Now draw the median lines back over what we just filled in
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
           color='w', marker='*', markeredgecolor='k')

#plt.savefig(path_save)
plt.show()