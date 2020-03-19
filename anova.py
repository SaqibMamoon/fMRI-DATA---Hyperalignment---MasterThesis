# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:56:03 2014

@author: andreabertana
"""

import numpy as np
from mvpa2.suite import *
import pandas as pnd
import sys
import nibabel as nib
 
cli_argv = sys.argv

#define paths
path_data = cli_argv[1]
path_mask = cli_argv[2]
path_pandas = cli_argv[3]
path_label = cli_argv[4]
path_save= cli_argv[5]
print path_mask
#Load label with run and category information    
label = SampleAttributes(path_label)
#open label from pandas dataframe 
print 'Loading structure'
dataframe = pnd.read_csv(path_pandas,sep = ' ', header = 0)
#specific stimuli
block = dataframe.Block
block = np.array(block)
#Built pymvpa structure
fds = fmri_dataset(samples = path_data, targets =  label.targets, chunks = label.chunks, mask = path_mask)
#put info    
fds.sa['block'] = block
print fds.shape
#Create list og subjects
    
# IN this loop step is possible to detrend the dataset (if needed). To do that, 
# just uncomment the lines detrender .. , and detrended_fds.
# Remember to also change zscore and fds(remove rest category) 

#remove rest, dummy scan and cue volumes    
fds = fds[fds.sa.targets != 'LeftHandCueProcedure']
print 'Num_run = ' + str(fds.UC)
fds = fds[fds.sa.targets != 'RightHandCuePROC' ]
print fds.UT
fds = fds[fds.sa.targets != 'RightFoottCuePROC' ]
print fds.UT
fds = fds[fds.sa.targets != 'LeftFootCuePROC' ]
print fds.UT
fds = fds[fds.sa.targets != 'TongueCuePROC' ]
print fds.UT
fds = fds[fds.sa.targets != 'Fixation']
print fds.UT
fds = fds[fds.sa.targets != '0']
print fds.UT
print fds.shape
#normalize time-series.
zscore(fds)

#Define parameters for feature selection (one way anova in this case)
nf = 1000
fselector = FixedNElementTailSelector(nf, tail='upper',
                                     mode='select',sort=False)

anova = OneWayAnova()
#take only run 1 to calculate ANOVA (avoid overfitting)
train = fds[fds.sa.chunks != 2 ,:]

fscores = anova(train)
featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]
selected = featsels[0].forward(train)

#revtest  = np.arange(100,100 + selected.nfeatures)
#
#img = map2nifti(selected, revtest)
#
#data = img.get_data()

vox = np.array(selected.fa['voxel_indices'])

img_nifti = nib.load(path_data)
affine = img_nifti.get_affine()

mask = np.zeros(shape=(91,109,91))
#fill mask with info in voxel_indices
for i in np.arange(len(vox)):
    mask[vox[i,0], vox[i,1], vox[i,2]] = 1

img = nib.AnalyzeImage(mask,affine=affine)
nib.save(img, path_save)
    
