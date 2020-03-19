#!/bin/bash

#This script should be run from the home.
#It launch all classification scripts and for each script and each condition(smoothing, detrending, mc) and (knn, svm) save a txt file with relative classification accuracy

chmod +x /home/andreabertana/Projects/HCP/pipeline/hyper_language/hyper_anova_svm_lang.py

cd /home/andreabertana/Projects/HCP/pipeline/hyper_language/



#for Svm
#smoothing
python ./hyper_anova_svm_lang.py smoothing
#detrending
python ./hyper_anova_svm_lang.py detrend
#motion correction
python ./hyper_anova_svm_lang.py mc

python ./anatomic_anova_svm_lang.py smoothing
#detrending
python ./anatomic_anova_svm_lang.py detrend
#motion correction
python ./anatomic_anova_svm_lang.py mc






















