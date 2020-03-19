# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:29:18 2014

@author: andreabertana
"""
import numpy as np

zeros = list()
nans = list()
for i in np.arange(len(ds_all)):
    notan = np.zeros(ds_all[i].shape[0])
    zero = np.zeros(ds_all[i].shape[0])
    for raw in np.arange(ds_all[i].shape[0]):
        notan[raw] = np.sum(1*np.isnan(ds_all[i].samples.T[:,raw]))
        zero[raw] = np.sum(1*(ds_all[i].samples.T[:,raw] == 0))
    zeros.append(zero)
    nans.append(notan)

mat_prod = list() 
info_subj = np.zeros(shape = (90,2))  
raw = 0 
for i in np.arange(len(ds_train)):
    samp_one = ds_train[i].samples.T
    the_idxs=np.delete(np.arange(len(ds_train)),i)
    for r in the_idxs:
        samp_two = ds_train[r].samples
        prod = np.dot(samp_one, samp_two)
        mat_prod.append(prod)
        info_subj[raw,0] = i
        info_subj[raw,1] = r
        raw = raw + 1
        
zeros = list()
nans = list()
for i in np.arange(len(ds_all_encoded)):
    notan = np.zeros(ds_all_encoded[i].shape[0])
    zero = np.zeros(ds_all_encoded[i].shape[0])
    for raw in np.arange(ds_all_encoded[i].shape[0]):
        notan[raw] = np.sum(1*np.isnan(ds_all_encoded[i].samples.T[:,raw]))
        zero[raw] = np.sum(1*(ds_all_encoded[i].samples.T[:,raw] == 0))
    zeros.append(zero)
    nans.append(notan)
    

hyper_or_not = np.zeros(shape = (90,3))
test_run = 1  
raw = 0    
ds_train = [sd[sd.sa.chunks != test_run,:] for sd in ds_all]
ds_test = [sd[sd.sa.chunks == test_run,:] for sd in ds_all]   
for i in np.arange(len(ds_train[i])): 
    first = ds_train[i]
    the_idxs=np.delete(np.arange(len(ds_train)),i)
    subj_list = list()
    for x in np.arange(len(the_idxs)):
        second = ds_train[x]
        subj_list.append(first)
        subj_list.append(second) 
        try:
            hyper = Hyperalignment() 
            hypmaps = hyper(subj_list)
            hyper_or_not[raw,0] = i
            hyper_or_not[raw,1] = x
            hyper_or_not[raw,2] = 1
            raw = raw +1
        except np.linalg.linalg.LinAlgError:
            print 'Error'
 
    
    #if 'SVD did not converge' in err.message:
        #your error handling block
#    else:
#       raise
         
    
    


        
    