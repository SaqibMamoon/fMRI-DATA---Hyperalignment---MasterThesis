#!/bin/bash

#This script should be run from the home.
#It launch all classification scripts and for each script and each condition(smoothing, detrending, mc) and (knn, svm) save a txt file with relative classification accuracy

chmod +x /home/andreabertana/Projects/HCP/pipeline/hyper_wm/hyper_anova/hyper_anova_svm.py

cd /home/andreabertana/Projects/HCP/pipeline/hyper_wm/

#Classification for ANOVA feature selection
#for Knn
#smoothing
#python ./hyper_anova/hyper_anova_svm.py smoothing
#detrending
python ./hyper_anova/hyper_anova_svm.py detrend
#motion correction
python ./hyper_anova/hyper_anova_svm.py  mc
