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

path = '/home/andreabertana/Projects/HCP/results/MNI/WM/'
nm_subj = np.loadtxt('/home/andreabertana/Projects/HCP/results/list_subject_first10.txt')
#nm_subj = (['100307', '100408'])
verbose.level = 2
ds_all = list()
save_list = list()
vt_list = list()
save_list2 = list() 
for name in nm_subj:
    name = str(name.astype('int'))
    print name
    #path for input files
    #path_mc = path + 'mc/' + name + '_mc.nii.gz'
    #path_det = path + 'detrend/vt/' + name + '_det_mc_vt.nii.gz'
    path_smo = path + 'smoothing/det/' + name + '_smooth_det.nii.gz'
    #path for vt mask
    path_vt = path + 'atlas_masks/merge_harv_dx.nii.gz'
    #path_save = path + 'mc/correlation/dx/' + name + '_mc_best500_dx.nii.gz'
    path_save = path + 'smoothing/det/correlation/harvard/dx/' + name + '_det_best500_dx.nii.gz'
    path_save2 = path + 'smoothing/det/correlation/harvard/dx/' + name + '_det_best300_dx.nii.gz'
    #path for labels
    final = fmri_dataset(samples = path_smo, mask = path_vt)
    #Create list og subjects
    ds_all.append(final)
    save_list.append(path_save)
    save_list2.append(path_save2)
    vt_list.append(path_vt)

n_voxels = 500
for p in np.arange(len(ds_all)):


    curr_p=ds_all[p].samples.T


    curr_list=np.zeros((curr_p.shape[0],len(ds_all)))


    the_idxs=np.delete(np.arange(len(ds_all)),p)


    for r in np.arange(len(the_idxs)):
        print(the_idxs[r])
        
        curr_r=ds_all[the_idxs[r]].samples.T

        curr_list[:,r]=np.max(np.corrcoef(curr_p,curr_r)[:curr_p.shape[0],curr_p.shape[0]:],1)

    ds_all[p].fa['bpcs']=np.sum(curr_list,1)
    
fselector=FixedNElementTailSelector(n_voxels,tail='upper',mode='select',sort=False)
    
featsels=[StaticFeatureSelection(fselector(ds_all[sd].fa['bpcs'].value)) for sd in range(len(ds_all))]
    
ds_fs=[featsels[i].forward(sd) for i,sd in enumerate(ds_all)]

vox =300
fselect=FixedNElementTailSelector(vox,tail='upper',mode='select',sort=False)
    
featsel=[StaticFeatureSelection(fselect(ds_all[sd].fa['bpcs'].value)) for sd in range(len(ds_all))]
    
ds_fstre=[featsel[i].forward(sd) for i,sd in enumerate(ds_all)]
#save correaltion masks
    
for s in np.arange(len(ds_fs)):
    mask = nib.load(vt_list[s])
    data =mask.get_data()
    affine = mask.get_affine()
    idx = ds_fs[s].fa['voxel_indices']
    idx =np.array(idx)
    zero = np.zeros((91,109,91))
    for i in idx:
        zero[i[0], i[1], i[2]] = data[i[0], i[1], i[2]]
    img = nib.AnalyzeImage(zero,affine=affine)
    nib.save(img, save_list[s])
    print nib.save(img, save_list[s])        
    del(zero)
    del(img)
    
#BEST 300
for s in np.arange(len(ds_fstre)):
    mask = nib.load(vt_list[s])
    data =mask.get_data()
    affine = mask.get_affine()
    idx = ds_fstre[s].fa['voxel_indices']
    idx =np.array(idx)
    zero = np.zeros((91,109,91))
    for i in idx:
        zero[i[0], i[1], i[2]] = data[i[0], i[1], i[2]]
    img = nib.AnalyzeImage(zero,affine=affine)
    nib.save(img, save_list2[s])
    print nib.save(img, save_list2[s])        
    del(zero)
    del(img)   
    
sd = 0    
for name in nm_subj:
    name = str(name.astype('int'))
    scores = ds_all[sd].fa['bpcs'].value
    scores.sort()
    save = path + 'smoothing/det/correlation/harvard/dx/scores/' + name + 'scores'
    np.savetxt(save,scores)
    sd = sd +1
    
del(save_list)
del(vt_list)
del(ds_all) 
del(ds_fs)  
del(ds_fstre)
   
verbose.level = 2
ds_all = list()
save_list = list()
vt_list = list() 
save_list2 = list()
for name in nm_subj:
    name = str(name.astype('int'))
    print name
    #path for input files
    #path_mc = path + 'mc/' + name + '_mc.nii.gz'
    #path_det = path + 'detrend/vt/' + name + '_det_mc_vt.nii.gz'
    path_smo = path + 'smoothing/det/' + name + '_smooth_det.nii.gz'
    #path for vt mask
    path_vt = path + 'atlas_masks/merge_harv_sx.nii.gz'
    #path_save = path + 'mc/correlation/dx/' + name + '_mc_best500_dx.nii.gz'
    path_save = path + 'smoothing/det/correlation/harvard/sx/' + name + '_det_best500_sx.nii.gz'
    path_save2 = path + 'smoothing/det/correlation/harvard/sx/' + name + '_det_best300_sx.nii.gz'
    #path for labels
    final = fmri_dataset(samples = path_smo, mask = path_vt)
    #Create list og subjects
    ds_all.append(final)
    save_list.append(path_save)
    save_list2.append(path_save2)
    vt_list.append(path_vt)

n_voxels = 500
for p in np.arange(len(ds_all)):


    curr_p=ds_all[p].samples.T


    curr_list=np.zeros((curr_p.shape[0],len(ds_all)))


    the_idxs=np.delete(np.arange(len(ds_all)),p)


    for r in np.arange(len(the_idxs)):
        print(the_idxs[r])
        
        curr_r=ds_all[the_idxs[r]].samples.T

        curr_list[:,r]=np.max(np.corrcoef(curr_p,curr_r)[:curr_p.shape[0],curr_p.shape[0]:],1)

    ds_all[p].fa['bpcs']=np.sum(curr_list,1)
    
fselector=FixedNElementTailSelector(n_voxels,tail='upper',mode='select',sort=False)
    
featsels=[StaticFeatureSelection(fselector(ds_all[sd].fa['bpcs'].value)) for sd in range(len(ds_all))]
    
ds_fs=[featsels[i].forward(sd) for i,sd in enumerate(ds_all)]

vox =300
fselect=FixedNElementTailSelector(vox,tail='upper',mode='select',sort=False)
    
featsel=[StaticFeatureSelection(fselect(ds_all[sd].fa['bpcs'].value)) for sd in range(len(ds_all))]
    
ds_fstre=[featsel[i].forward(sd) for i,sd in enumerate(ds_all)]
#save correaltion masks
#bEST 500  
for s in np.arange(len(ds_fs)):
    mask = nib.load(vt_list[s])
    data =mask.get_data()
    affine = mask.get_affine()
    idx = ds_fs[s].fa['voxel_indices']
    idx =np.array(idx)
    zero = np.zeros((91,109,91))
    for i in idx:
        zero[i[0], i[1], i[2]] = data[i[0], i[1], i[2]]
    img = nib.AnalyzeImage(zero,affine=affine)
    nib.save(img, save_list[s])
    print nib.save(img, save_list[s])        
    del(zero)
    del(img)   
#BEST 300
for s in np.arange(len(ds_fstre)):
    mask = nib.load(vt_list[s])
    data =mask.get_data()
    affine = mask.get_affine()
    idx = ds_fstre[s].fa['voxel_indices']
    idx =np.array(idx)
    zero = np.zeros((91,109,91))
    for i in idx:
        zero[i[0], i[1], i[2]] = data[i[0], i[1], i[2]]
    img = nib.AnalyzeImage(zero,affine=affine)
    nib.save(img, save_list2[s])
    print nib.save(img, save_list2[s])        
    del(zero)
    del(img)   

sd = 0    
for name in nm_subj:
    name = str(name.astype('int'))
    scores = ds_all[sd].fa['bpcs'].value
    scores.sort()
    save = path + 'smoothing/det/correlation/harvard/sx/scores/' + name + 'scores'
    np.savetxt(save,scores)
    sd = sd +1  
    
del(save_list)
del(vt_list)
del(ds_all) 
del(ds_fs)  
del(ds_fstre)

