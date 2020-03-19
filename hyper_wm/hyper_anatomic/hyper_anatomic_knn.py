# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:54:24 2014

@author: andreabertana
"""

import numpy as np
from mvpa2.suite import *
import pandas as pnd
import sys


path = '/home/andreabertana/Projects/HCP/results/MNI/WM/'
nm_subj = np.loadtxt('/home/andreabertana/Projects/HCP/results/list_subject_first10.txt')
verbose.level = 2
ds_all = list()
subj_all =list()
ds_all_encoded = list() 

cli_argv = sys.argv

if cli_argv[1] == 'detrend':
    path_prefix = 'detrend/vt/harvard/'
    path_suffix ='_det_mc.nii.gz' 
    path_s = 'detrend'
elif cli_argv[1] == 'smoothing':
    path_prefix = 'smoothing/det/'
    path_suffix = '_smooth_det.nii.gz'
    path_s = 'smoothing'
elif cli_argv[1] == 'mc':
    path_prefix = 'mc/vt/'
    path_suffix = '_mc_vtharv.nii.gz'
    path_s = 'mc'


for name in nm_subj:
    name = str(name.astype('int'))
    print name
    path_func = path + path_prefix + name + path_suffix
    path_mni = path + 'atlas_masks/nnzero/MNIall_vtharv.nii.gz'
    #path for labels
    path_pandas = path + 'label/concatenated/' + name + '_labels.txt'
    path_label = path + 'label/' + name + '_labels_pymvpa.txt'
    label = SampleAttributes(path_label)
    #load label file
    #open label
    dataframe = pnd.read_csv(path_pandas,sep = ' ', header = 0)
    stimuli = dataframe.Stimuli
    stimuli = np.array(stimuli)
    #specific condition (0-back or 2-Back)
    block = dataframe.Block
    block = np.array(block)
    #Built pymvpa structure
    final = fmri_dataset(samples = path_func, targets =  label.targets, chunks = label.chunks, mask = path_mni)
    final.sa['stimuli'] = stimuli
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
    #now we can remove the rest category and the 0ros of the first volumes
    #fds = detrended_fds[detrended_fds.sa.targets != 'rest']# if detrendind
    print fds.UT
    fds = fds[fds.sa.targets != 'Rest']
    print 'Num_run = ' + str(fds.UC)
    fds = fds[fds.sa.targets != '0' ]
    print fds.UT
    fds = fds[fds.sa.stimuli != '0' ]
    print fds.UT
    fds = fds[fds.sa.targets != 'Cue0BackPROC']
    fds = fds[fds.sa.targets != 'Cue2BackPROC']
    print fds.UT
    print fds.shape
    ### ENCODING PART. Here all the volumes of each target stimulus independently for
    ### each run are stucked together and the mean is computed. 
    averager = mean_group_sample(['targets','block', 'chunks'])
    fds = fds.get_mapped(averager)
    print i 
    print fds.shape
    print fds.sa.targets
    ds_all_encoded.append(fds)

### DEFINING USEFULL VARIABLES
# number of subjects
nsubjs = len(ds_all_encoded)
# number of categories
ncats = len(ds_all_encoded[0].UT)
# number of run
nruns = len(ds_all_encoded[0].UC)

verbose(2, "%d subjects" % len(ds_all_encoded))
verbose(2, "Per-subject dataset: %i samples with %i features" % ds_all_encoded[0].shape)
verbose(2, "Stimulus categories: %s" % ', '.join(ds_all_encoded[0].UT))

### WE COULD ZSCORE DATA FROM THE REST CATEGORY
# zscore the data as differences from the rest category 
#for i in range(0,4):
#	zscore(ds_all[i])
### DEFINE THE CLASSIFIER
## use same linear support vector machine
#clf = LinearCSVMC()
## or multionomial Logistic Regression
#clf = clfs.smlr.SMLR()
clf = clfs.knn.kNN()

### IF ANOVA FEATURE SELECTION IS NEEDED
### ps. If used, remember to change the variable clf in cv with fsclf

# feature selection helpers
nf = 1000
fselector = FixedNElementTailSelector(nf, tail='upper',
                                     mode='select',sort=False)
sbfs = SensitivityBasedFeatureSelection(OneWayAnova(), fselector,
                                      enable_ca=['sensitivities'])
#create classifier with automatic feature selection
fsclf = FeatureSelectionClassifier(clf, sbfs)

#inject the subjects ID into the datasets
for i,sd in enumerate(ds_all_encoded):
   	sd.sa['subject'] = np.repeat(i, len(sd))

### Computing the similarity measured between categories of original data 
### samples by avarage the similarity structure (multivariate) of individual data.
#sm_orig = [np.corrcoef(sd.get_mapped(mean_group_sample(['targets'])).samples)for sd in ds_all_encoded]
# mean across subjects
#sm_orig_mean = np.mean(sm_orig, axis=0)
#ANATOMICAL BETWEEN SUBJECT 
verbose(2, "between-subject (anatomically aligned)...", cr=False, lf=False)
ds_mni = vstack(ds_all_encoded)
mni_start_time = time.time()
cv = CrossValidation(fsclf,
                     NFoldPartitioner(attr='subject'),
                     errorfx=mean_match_accuracy)
bsc_mni_results = cv(ds_mni)
verbose(2, "done in %.1f seconds" % (time.time() - mni_start_time,))

verbose(2, "between-subject (anatomically aligned): %.2f +/-%.3f" %
         (np.mean(bsc_mni_results),
           np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1)))
           
#res_mni =  np.mean(bsc_mni_results),np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1)         

path_save = path + path_s + 'anat_knn_all.txt'

np.savetxt(path_save,bsc_mni_results)