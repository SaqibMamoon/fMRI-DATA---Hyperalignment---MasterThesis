#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:31:03 2014

@author: andreabertana
"""

# Between participants correlation score


# It requires a list of datasets (in PYMVPA style):
# ds_all
# it produces a new feature attribute called


# bpcs
# and filled according to bpcs computed according to
# Haxby 2011 definition

import numpy as np
from mvpa2.suite import *
import nibabel as nib

path = '/home/andreabertana/Projects/HCP/results/MNI/'
nm_subj = np.loadtxt('/home/andreabertana/Projects/HCP/results/list_subject_first10.txt')
#nm_subj = (['100307', '100408'])
verbose.level = 2
ds_all = list()
for name in nm_subj:
    name = str(name.astype('int'))
    print name
    #path for input files
    #path_mc = path + 'mc/' + name + '_mc.nii.gz'
    path_det = path + 'detrend/vt/' + name + '_det_mc_vt.nii.gz'
    #path_smo = path + 'smoothing/det/' + name + '_smooth_det.nii.gz'
    #path for vt mask
    #path_vt = path + 'vt_mask/' + name + '_nnzero_restricted_dx.nii.gz'
    #path_save = path + 'mc/correlation/dx/' + name + '_mc_best500_dx.nii.gz'
    path_vt = path + 'detrend/correlation/restricted/' + name + '_best1000_det.nii.gz'
    #path_save2 = path + 'detrend/correlation/restricted/dx/' + name + '_det_best300_dx.nii.gz'
    #path for labels
    final = fmri_dataset(samples = path_det, mask = path_vt)
    #Create list og subjects
    ds_all.append(final)

score_all_test = list()
raw = 0
care_test = np.zeros(shape = (90,3))
for p in np.arange(len(ds_test)):

    curr_p=ds_test[p].samples.T

    curr_list=np.zeros((curr_p.shape[0],curr_p.shape[0]))
    
    the_idxs=np.delete(np.arange(len(ds_test)),p)

    for r in np.arange(len(the_idxs)):
        
        curr_r=ds_test[the_idxs[r]].samples.T
        curr_list =np.corrcoef(curr_p,curr_r)[:curr_p.shape[0],curr_p.shape[0]:]
        score_all_test.append(curr_list)
        print p
        print r
        care_test[raw,0] = np.sum(1*np.isnan(score_all_test[raw]))
        care_test[raw,1] = p
        care_test[raw,2] = the_idxs[r]
        raw = raw + 1
        
score_all_train = list()
raw = 0
care_train = np.zeros(shape = (90,3))
for p in np.arange(len(ds_train)):

    curr_p=ds_train[p].samples.T

    curr_list=np.zeros((curr_p.shape[0],curr_p.shape[0]))
    
    the_idxs=np.delete(np.arange(len(ds_train)),p)

    for r in np.arange(len(the_idxs)):
        
        curr_r=ds_train[the_idxs[r]].samples.T
        curr_list =np.corrcoef(curr_p,curr_r)[:curr_p.shape[0],curr_p.shape[0]:]
        score_all_train.append(curr_list)
        print p
        print r
        care_train[raw,0] = np.sum(1*np.isnan(score_all_train[raw]))
        care_train[raw,1] = p
        care_train[raw,2] = the_idxs[r]
        raw = raw + 1

        


 
                
    


