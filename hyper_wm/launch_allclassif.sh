#!/bin/bash

#This script should be run from the home.
#It launch all classification scripts and for each script and each condition(smoothing, detrending, mc) and (knn, svm) save a txt file with relative classification accuracy

chmod +x /home/andreabertana/Projects/HCP/pipeline/hyper_wm/hyper_anatomic/hyper_anatomic_knn.py
chmod +x /home/andreabertana/Projects/HCP/pipeline/hyper_wm/hyper_anatomic/hyper_anatomic_svm.py
chmod +x /home/andreabertana/Projects/HCP/pipeline/hyper_wm/hyper_corr1000/hyper_corr1000_knn.py
chmod +x /home/andreabertana/Projects/HCP/pipeline/hyper_wm/hyper_corr1000/hyper_corr1000_svm.py
chmod +x /home/andreabertana/Projects/HCP/pipeline/hyper_wm/hyper_corr600/hyper_corr600_knn.py
chmod +x /home/andreabertana/Projects/HCP/pipeline/hyper_wm/hyper_anova/hyper_anova_svm.py
chmod +x /home/andreabertana/Projects/HCP/pipeline/hyper_wm/hyper_anova/hyper_anova_knn.py

cd /home/andreabertana/Projects/HCP/pipeline/hyper_wm/

#Classification for anatomical alignment with ANOVA feature selection
#for Knn
#smoothing
#python ./hyper_anatomic/hyper_anatomic_knn.py smoothing
#detrending
#python ./hyper_anatomic/hyper_anatomic_knn.py detrend
#motion correction
#python ./hyper_anatomic/hyper_anatomic_knn.py mc

#for Svm
#smoothing
python ./hyper_anatomic/hyper_anatomic_svm.py smoothing
#detrending
python ./hyper_anatomic/hyper_anatomic_svm.py detrend
#motion correction
python ./hyper_anatomic/hyper_anatomic_svm.py mc

#Classification with correlation feature selection for 1000 voxels 
#for Knn
#smoothing 
#python ./hyper_corr1000/hyper_corr1000_knn.py smoothing
#detrending
#python ./hyper_corr1000/hyper_corr1000_knn.py detrend
#motion correction
#python ./hyper_corr1000/hyper_corr1000_knn.py mc

#for svm
#smoothing 
python ./hyper_corr1000/hyper_corr1000_svm.py smoothing
#detrending
python ./hyper_corr1000/hyper_corr1000_svm.py detrend
#motion correction
python ./hyper_corr1000/hyper_corr1000_svm.py mc

#Classification with correlation feature selection for 600 voxels
#for knn
#smoothing 
#python ./hyper_corr600/hyper_corr600_knn.py smoothing
#detrending
#python ./hyper_corr600/hyper_corr600_knn.py detrend
#motion correction
#python ./hyper_corr600/hyper_corr600_knn.py mc

#for svm
#smoothing 
python ./hyper_corr600/hyper_corr600_svm.py smoothing
#detrending
python ./hyper_corr600/hyper_corr600_svm.py detrend
#motion correction
python ./hyper_corr600/hyper_corr600_svm.py mc

#Classification for ANOVA feature selection
#for Knn
#smoothing
#python ./hyper_anova/hyper_anova_svm.py smoothing
#detrending
#python ./hyper_anova/hyper_anova_svm.py detrend
#motion correction
#python ./hyper_anova/hyper_anova_svm.py  mc

#for svm
#smoothing 
#python ./hyper_anova/hyper_anova_knn.py smoothing
#detrending
#python ./hyper_anova/hyper_anova_knn.py detrend
#motion correction
#python ./hyper_anova/hyper_anova_knn.py mc





















