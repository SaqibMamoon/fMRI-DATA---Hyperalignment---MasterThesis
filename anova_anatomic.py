# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:56:03 2014

@author: andreabertana
"""

import numpy as np
from mvpa2.suite import *
import pandas as pnd
import sys


path = '/home/andreabertana/Projects/HCP/results/MNI/MOTOR/'
nm_subj = np.loadtxt('/home/andreabertana/Projects/HCP/results/list_subject_first10.txt')
verbose.level = 2
ds_all = list()
subj_all =list()
ds_all_encoded = list()

cli_argv = sys.argv

if cli_argv[1] == 'detrend':
    path_prefix = 'detrend/'
    path_suffix ='_det_mc.nii.gz' 
    path_s = 'detrend'
elif cli_argv[1] == 'smoothing':
    path_prefix = 'smoothing/'
    path_suffix = '_det_smo_motor.nii.gz'
    path_s = 'smoothing'
elif cli_argv[1] == 'mc':
    path_prefix = 'mc/motor/'
    path_suffix = '_mc_motor.nii.gz'
    path_s = 'mc'

for name in nm_subj:
    #convert int in str
    name = str(name.astype('int'))
    print name
    #define paths
    path_data = path + path_prefix + name + path_suffix
    path_mask = path + 'atlas_masks/nnzero/MNI_allnnzero.nii.gz'
    path_pandas = path + 'label/concatenated/' + name + '_labels.txt'
    path_label = path + 'label/' + name + '_labels_pymvpa.txt'
    #Load label with run and category information    
    label = SampleAttributes(path_label)
    #open label from pandas dataframe 
    dataframe = pnd.read_csv(path_pandas,sep = ' ', header = 0)
    #specific condition (0-back or 2-Back)
    block = dataframe.Block
    block = np.array(block)
    #Built pymvpa structure
    final = fmri_dataset(samples = path_data, targets =  label.targets, chunks = label.chunks, mask = path_mask)
    #put info    
    final.sa['block'] = block
    #Create list og subjects
    subj_all.append(final)
    
# IN this loop step is possible to detrend the dataset (if needed). To do that, 
# just uncomment the lines detrender .. , and detrended_fds.
# Remember to also change zscore and fds(remove rest category) 
for i in np.arange(len(subj_all)):
    fds = subj_all[i]
    ##### LINES FOR DETRENDIG
    #detrender = PolyDetrendMapper(polyord=1, chunks_attr='chunks')
    #detrended_fds = fds.get_mapped(detrender)
    #### ZSCORE
    #zscore(detrended_fds) # if detrendind
    zscore(fds)
    print fds.UT
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
    averager = mean_group_sample(['targets','block', 'chunks'])
    fds = fds.get_mapped(averager)
    print i 
    print fds.shape
    print fds.sa.targets
    ds_all_encoded.append(fds)

#Define parameters for feature selection (one way anova in this case)
# feature selection helpers
nf = 1000
fselector = FixedNElementTailSelector(nf, tail='upper',
                                     mode='select',sort=False)
sbfs = SensitivityBasedFeatureSelection(OneWayAnova(), fselector,
                                      enable_ca=['sensitivities'])
anova = OneWayAnova()
#inject the subjects ID into the datasets
for i,sd in enumerate(ds_all_encoded):
   	sd.sa['subject'] = np.repeat(i, len(sd))

ds_mni = vstack(ds_all_encoded[0:9])
fscores = anova(ds_mni)
featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]
selected = featsels[0].forward(ds_mni)

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
    
