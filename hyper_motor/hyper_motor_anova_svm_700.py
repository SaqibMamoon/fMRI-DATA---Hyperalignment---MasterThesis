#!/usr/bin/python	

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
    path_mask = path + 'atlas_masks/nnzero/' + name + '_motor_nnzero.nii.gz'
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
    #fds = fds[fds.chunks != 'nan']
    fds = subj_all[i]
    #zscore(fds)
    #now we can remove the rest category and the 0ros of the first volumes
    #fds = detrended_fds[detrended_fds.sa.targets != 'rest']# if detrendind
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
    zscore(fds)
    ds_all.append(fds)

### DEFINING USEFULL VARIABLES
# number of subjects
nsubjs = len(ds_all)
# number of categories
ncats = len(ds_all[0].UT)
# number of run
nruns = len(ds_all[0].UC)
verbose(2, "%d subjects" % len(ds_all))
verbose(2, "Per-subject dataset: %i samples with %i features" % ds_all[0].shape)
verbose(2, "Stimulus categories: %s" % ', '.join(ds_all[0].UT))
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
nf = 700
fselector = FixedNElementTailSelector(nf, tail='upper',
                                     mode='select',sort=False)
sbfs = SensitivityBasedFeatureSelection(OneWayAnova(), fselector,
                                      enable_ca=['sensitivities'])
#create classifier with automatic feature selection
fsclf = FeatureSelectionClassifier(clf, sbfs)

#inject the subjects ID into the datasets
for i,sd in enumerate(ds_all):
   	sd.sa['subject'] = np.repeat(i, len(sd))

# STARTING THE HYPERALIGNMENT
verbose(2, "between-subject (hyperaligned)...", cr=False, lf=False)
hyper_start_time = time.time()
bsc_hyper_results = []
#cross-validation over subjects
cv = CrossValidation(fsclf, NFoldPartitioner(attr='subject'),
                     errorfx=mean_match_accuracy,
			enable_ca=['stats'])
   
# HYPERALIGNMENT 
#Two choices for cross validation are possibles:
# - Leave-one-run-out for hyperalignment training
nruns = 1,2
# - Leave-two-run-out
#nruns =  (0,1),(2,3),(4,5),(6,7),(9,10),(11,0)
#NOW TRY LEAVE TWO RUN OUT
#for test_run in combinations(nruns,2):	
for test_run in nruns:	
#Split in training and testing set: 
    #For leave two run out, use this: 
    #ds_train = [sd[np.logical_and(sd.sa.chunks != test_run[0], sd.sa.chunks != test_run[1]) ,:] for sd in ds_all]
    #ds_test = [sd[np.logical_or(sd.sa.chunks == test_run[0], sd.sa.chunks == test_run[1]),:] for sd in ds_all]
    # For leave one run out, use this:
    ds_train = [sd[sd.sa.chunks != test_run,:] for sd in ds_all]
    ds_test = [sd[sd.sa.chunks == test_run,:] for sd in ds_all]
    #Encode for test set (for classification)

####IF ANOVA FEATURE SELECTION IS NEEDED (Remember two uncomment the following lines
#### together with ds_test_fs and remember to change also the variable ds_test in ds_hyper
#### with ds_test_fs)
    # manual feature selection for every individual dataset in the list
    anova = OneWayAnova()
    fscores = [anova(sd) for sd in ds_train]
    featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]
    ds_train_fs = [featsels[i].forward(sd) for i, sd in enumerate(ds_train)]

#Defining hyper-function and computing hyperalignment parameters.Change the variable ds_train in ds_train_fs if anova is used
    hyper = Hyperalignment(alignment=ProcrusteanMapper(svd='dgesvd',space='commonspace'))
    hypmaps = hyper(ds_train_fs) 
#Applying hyperalignment parameters on the test set (the run left out) 
    ds_test_fs= [featsels[i].forward(sd) for i, sd in enumerate(ds_test)]  
    ds_hyper = [ hypmaps[i].forward(sd) for i, sd in enumerate(ds_test_fs)]
    print i
    averager = mean_group_sample(['targets', 'block', 'chunks'])
    ds_hyper_encoded = [sd.get_mapped(averager) for sd in ds_hyper]
    # Computing similarity matrices with the same individual average but this time for hyperaligned data
    #sm_hyper_mean = np.mean([np.corrcoef(sd.get_mapped(mean_group_sample(['targets'])).samples) for sd in ds_hyper], axis=0)
    #Stuck all features(all voxels of all subjcects) that are in the common space and run crossvalidation. 
    ds_hyper = vstack(ds_hyper_encoded)
    #compute similarity maps for avaraged hyperaligned data (considering everithihg as if it was only oine subject)
    #sm_hyper = np.corrcoef(ds_hyper.get_mapped(mean_group_sample(['targets'])))
    # zscore each subject individually after transformation for optimal performance
    #zscore(ds_hyper, chunks_attr='subject')
    res_cv = cv(ds_hyper)
    bsc_hyper_results.append(res_cv)
	
 

bsc_hyper_results = hstack(bsc_hyper_results)           
          
verbose(2, "done in %.1f seconds" , (time.time() - hyper_start_time,))
verbose(2, "between-subject (hyperaligned): %.2f +/-%.3f" \
        , (np.mean(bsc_hyper_results),
           np.std(np.mean(bsc_hyper_results, axis=1)) / np.sqrt(nsubjs - 1)))
           
path_save = path + path_s + '_anova_svm_all_700.txt'

np.savetxt(path_save,bsc_hyper_results)

          
    