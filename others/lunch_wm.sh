#!/bin/bash
 #this script should be run from andreabertana dir


#correlation smoothing motor task
#python /home/andreabertana/Projects/HCP/pipeline/correlation/correlation_pymvpa_smo.py
#merge bialteral masks
/home/andreabertana/Projects/HCP/pipeline/mask/fixing_hemismask.sh
#classificazione smoothing corr 600 svm
python /home/andreabertana/Projects/HCP/pipeline/hyper_wm/hyper_smo600_svm.py
#classificazione smoothing corr 1000 svm
python /home/andreabertana/Projects/HCP/pipeline/hyper_wm/hyper_smo1000_svm.py
#classificazione smoothing corr 1000 knn
python /home/andreabertana/Projects/HCP/pipeline/hyper_wm/hyper_smo1000_knn.py
#classificazione smoothing corr 600 knn
python /home/andreabertana/Projects/HCP/pipeline/hyper_wm/hyper_smo600_knn.py
