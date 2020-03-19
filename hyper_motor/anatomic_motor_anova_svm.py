# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:54:24 2014

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
clf = LinearCSVMC()
## or multionomial Logistic Regression
#clf = clfs.smlr.SMLR()
#knn = clfs.knn.kNN()

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
           
res_mni =  np.mean(bsc_mni_results),np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1)         

path_save = path + path_s + 'anat_svm_all.txt'

np.savetxt(path_save,bsc_mni_results)