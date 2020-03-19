#!/bin/bash
 #this script should be run from andreabertana dir

cd /home/andreabertana/Projects/HCP/pipeline/hyper_motor/

#classificatin hyper ANOVA 300
#mc
python ./hyper_motor_anova_svm_300.py mc
#detrend
python ./hyper_motor_anova_svm_300.py detrend
#smoothing
python ./hyper_motor_anova_svm_300.py smoothing

#classificatin hyper ANOVA 500
#mc
python ./hyper_motor_anova_svm_500.py mc
#detrend
python ./hyper_motor_anova_svm_500.py detrend
#smoothing
python ./hyper_motor_anova_svm_500.py smoothing

#classificatin hyper ANOVA 700
#mc
python ./hyper_motor_anova_svm_700.py mc
#detrend
python ./hyper_motor_anova_svm_700.py detrend
#smoothing
python ./hyper_motor_anova_svm_700.py smoothing


#clas anatomical ANOVA 300
#mc
python ./anatomic_motor_anova_svm_300.py mc
#detrend
python ./anatomic_motor_anova_svm_300.py detrend
#smoothing
python ./anatomic_motor_anova_svm_300.py smoothing

#clas anatomical ANOVA 500
#mc
python ./anatomic_motor_anova_svm_500.py mc
#detrend
python ./anatomic_motor_anova_svm_500.py detrend
#smoothing
python ./anatomic_motor_anova_svm_500.py smoothing

#clas anatomical ANOVA 700
#mc
python ./anatomic_motor_anova_svm_700.py mc
#detrend
python ./anatomic_motor_anova_svm_700.py detrend
#smoothing
python ./anatomic_motor_anova_svm_700.py smoothing



