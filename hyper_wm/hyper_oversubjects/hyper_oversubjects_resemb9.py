#!/usr/bin/python
"""
Created on Thu Apr 10 14:01:19 2014

@author: andreabertana
"""


import numpy as np
from mvpa2.suite import *
import pandas as pnd


path = '/home/andreabertana/Projects/'
nm_subj = np.loadtxt('/home/andreabertana/Projects/list_subject65_resemb9.txt')
#list subjects with more cateogrie-s volumes 
#nr_subj_morevol = np.array([6,9,19,22,27,46,56,60,62,63,67,73,74,75,76])
#nm_subj = np.loadtxt('/home/andreabertana/Projects/list_subject_prob3.txt')
verbose.level = 2
#open lists to storage subjects data
ds_all = list()
subj_all =list() 
    
cli_argv = sys.argv

for name in nm_subj:
#convert int in str
    name = str(name.astype('int'))
    print name
    #define paths
    path_data = path + 'smoothing/harvard/' + name + '_det_mc.nii.gz' 
    path_mask = path + 'anova_mask/atlas_masks/' + name + '_anova1000.nii.gz'
    path_pandas = path + 'label/concatenated/' + name + '_labels.txt'
    path_label = path + 'label/' + name + '_labels_pymvpa.txt'
    #Load label with run and category information    
    label = SampleAttributes(path_label)
    #open label from pandas dataframe 
    dataframe = pnd.read_csv(path_pandas,sep = ' ', header = 0)
    #specific stimuli
    stimuli = dataframe.Stimuli
    stimuli = np.array(stimuli)
    #specific condition (0-back or 2-Back)
    block = dataframe.Block
    block = np.array(block)
    #Built pymvpa structure
    final = fmri_dataset(samples = path_data, targets =  label.targets, chunks = label.chunks, mask = path_mask)
    #put info    
    final.sa['stimuli'] = stimuli
    final.sa['block'] = block
    #Create list og subjects
    subj_all.append(final)

    
# IN this loop step is possible to detrend the dataset (if needed). To do that, 
# just uncomment the lines detrender .. , and detrended_fds.
# Remember to also change zscore and fds(remove rest category) 
for i in np.arange(len(subj_all)):
    fds = subj_all[i]
    #remove rest, dummy scan and cue volumes    
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
    #normalize time-series.
    zscore(fds)
    ds_all.append(fds)

#delete now subject_all to save memory
del(subj_all)

### DEFINING USEFULL VARIABLES
# number of subjects
nsubjs = len(ds_all)
# number of categories
ncats = len(ds_all[0].UT)
# number of run
nruns = len(ds_all[0].UC)
#LET's Print some info
verbose(2, "%d subjects" % len(ds_all))
verbose(2, "Per-subject dataset: %i samples with %i features" % ds_all[0].shape)
verbose(2, "Stimulus categories: %s" % ', '.join(ds_all[0].UT))

### WE COULD ZSCORE DATA FROM THE REST CATEGORY
# zscore the data as differences from the rest category (if did not remove before) 
#for i in range(0,4):
#	zscore(ds_all[i])

### DEFINE THE CLASSIFIER
## use same linear support vector machine
clf = LinearCSVMC()
#clf = SVM()
## or multionomial Logistic Regression
#clf = clfs.smlr.SMLR()
#nearest-nighbor classifier
#clf = clfs.knn.kNN()

### IF ANOVA FEATURE SELECTION IS NEEDED
### ps. If used, remember to change the variable clf in cv with fsclf

# feature selection helpers


#inject the subjects ID into the datasets
for i,sd in enumerate(ds_all):
   	sd.sa['subject'] = np.repeat(i, len(sd))
 
# STARTING THE HYPERALIGNMENT
verbose(2, "between-subject (hyperaligned)...", cr=False, lf=False)
hyper_start_time = time.time()
bsc_hyper_results = []


#split the dataset in train (first 50 subjects and test(second 30))
#ds_fifty = ds_all[0:50]
#ds_thirty = ds_all[50:]
for i in np.arange(10,55,5):
    i_save = str(i.astype('int'))

    ds_for_hyper = ds_all[0:i]
    ds_thirty = ds_all[50:]
     
    # take first run of ds_for_hyper for creating the commonspace
    ds_train_hyp = [sd[sd.sa.chunks != 2,:] for sd in ds_for_hyper]
    #take only data of the second run(first has been used for ANOVA) for the 30 subjects of test set
    ds_test_t_run2 = [sd[sd.sa.chunks == 2,:] for sd in ds_thirty]
    #take the first run to calculate the transformation
    ds_test_t_run1 = [sd[sd.sa.chunks != 2,:] for sd in ds_thirty]
    
    #Defining hyper-function
    hyper = Hyperalignment(alignment=ProcrusteanMapper(svd='dgesvd',space='commonspace'))
    #compute hyperalignment parameters
    hyper.train(ds_train_hyp) 
        
    for j in np.arange(10,35,10):
        #define the classifier (inside cause i wanna delete it when finished)
        clf = LinearCSVMC()
        #take j subjects for training the classifier where j=  10,20,30,40,50
        ds_classifier = ds_all[0:j]
        #take the second run of first i subjects to train the classifier
        ds_train_classif = [sd[sd.sa.chunks == 2,:] for sd in ds_classifier]
        #take the first run to compute the transformation
        ds_train_classif_run1 = [sd[sd.sa.chunks != 2,:] for sd in ds_classifier]
        
        #apply hyperalignment parameters on the test set (the run left out of first 10 subjects) 
        hypmaps_hyp = hyper(ds_train_classif_run1)
        ds_hyper_classif = [ hypmaps_hyp[k].forward(sd) for k, sd in enumerate(ds_train_classif)]
        
        #zscore timeseries in commonspace for better performance
        #zscore(ds_hyper_classif, chunks_attr='subject')
        #Encoding needs to train the classifier
        averager = mean_group_sample(['targets', 'block', 'chunks'])
        ds_hyper_classif_encoded = [sd.get_mapped(averager) for sd in ds_hyper_classif]
        
        #stack subject in common space to train classifier
        ds_hyper_classif_stack = vstack(ds_hyper_classif_encoded)
        #train classifier
        clf.train(ds_hyper_classif_stack)
        
        #project the second run of test group subjects (ds_test_t_run1) in commonspace
        hypmaps_t = hyper(ds_test_t_run1)
        ds_hyper_test = [ hypmaps_t[h].forward(sd) for h, sd in enumerate(ds_test_t_run2)]
        
        #zscore timeseries in commonspace for better performance
        #zscore(ds_hyper_test, chunks_attr='subject')
        #Encoding needs to train the classifier
        averager = mean_group_sample(['targets', 'block', 'chunks'])
        ds_hyper_test_encoded = [sd.get_mapped(averager) for sd in ds_hyper_test]    
        
        #take it independently (or stack together)
        ds_hyper_test = vstack(ds_hyper_test_encoded)
        #test the classifier
        prediction = clf.predict(ds_hyper_test.samples)
        #get accuracy of predictions
        acc = np.mean(prediction == ds_hyper_test.sa.targets)
        #bsc_hyper_results.append(acc)
        #save predictions and real categories
        j= str(j.astype('int'))
        path_realcat = './Projects/results/realcat'+ i_save + '_' + j + '_9.txt'
        np.savetxt(path_realcat , ds_hyper_test.sa.targets, fmt='%s') 
        path_prediction = './Projects/results/prediction'+ i_save + '_' + j +'_9.txt'
        np.savetxt(path_prediction,prediction, fmt='%s')
        path_save =  './Projects/results/acc_' + i_save + '_' + j + '_9.txt' 
        acc = np.array(acc).reshape(1,)
        np.savetxt(path_save,acc)
        #delete variables to be sure of non-overlapping
        del(hypmaps_hyp)
        del(ds_classifier)
        del(ds_train_classif)
        del(ds_hyper_classif)
        del(ds_hyper_classif_encoded)
        del(ds_hyper_classif_stack)
        del(ds_hyper_test)
        del(ds_hyper_test_encoded)
        del(prediction)
        del(acc)
        del(clf)
    del(hyper)
    
    #bsc_hyper_results = hstack(bsc_hyper_results)
    
#        verbose(2, "within-subject: %.2f +/-%.3f", (np.mean(wsc_results),
#                   np.std(wsc_results) / np.sqrt(nsubjs - 1)))         
#                  
#        verbose(2, "done in %.1f seconds" , (time.time() - hyper_start_time,))
#        verbose(2, "between-subject (hyperaligned): %.2f +/-%.3f" \
#                , (np.mean(bsc_hyper_results),
#                   np.std(np.mean(bsc_hyper_results, axis=1)) / np.sqrt(nsubjs - 1)))
#        
#        res_ws =  np.mean(wsc_results), np.std(wsc_results) / np.sqrt(nsubjs - 1)       
#        res_hyp = np.mean(bsc_hyper_results),np.std(np.mean(bsc_hyper_results, axis=1)) / np.sqrt(nsubjs - 1)
#        
    #res = (res_ws, res_hyp)
    #path_save = path+ '_anova_svm' + i +'.txt'

  