#!/bin/bash
pwd
for subjname in $(cat /home/andreabertana/Projects/HCP/results/list_subject_first10.txt)
#subjname=100307
do
cd /home/andreabertana/Projects/HCP/results/MNI/MOTOR
pwd
#this script convert  nifti files to .brik

#3dcopy ./mc/$subjname'_mc.nii.gz' ./afni/mc/$subjname'_mc'
echo 'conversion done'
#calculate intracranial mask to limit smoothing
#3dAutomask -prefix ./afni/whole_mask/$subjname'_mask' ./afni/mc/$subjname'_mc'+tlrc
echo 'mask done'
#smoothing 4mm 
#3dmerge -1blur_fwhm 4.0 -doall -prefix ./afni/smoothing/$subjname'_smo' ./afni/mc/$subjname'_mc'+tlrc   
echo 'smo done'
#scaling
#3dTstat -prefix ./afni/mean/$subjname'_mean' ./afni/smoothing/$subjname'_smo'+tlrc 
echo 'mean done'
#3dcalc -prefix ./afni/scaling/$subjname'_norm' -a ./afni/smoothing/$subjname'_smo'+tlrc -b ./afni/mean/$subjname'_mean'+tlrc -c ./afni/whole_mask/$subjname'_mask'+tlrc -expr 'c*100*((a - b)/ b)'
echo 'norm done'
#glm
#waver -WAV -TR 0.72s -input ./afni/label/stim/$subjname'_t_afni.txt' -numout 568 > ./afni/label/hrf/$subjname't_hrf.txt'
#waver -WAV -TR 0.72s -input ./afni/label/stim/$subjname'_rh_afni.txt' -numout 568 > ./afni/label/hrf/$subjname'rh_hrf.txt'
#waver -WAV -TR 0.72s -input ./afni/label/stim/$subjname'_lh_afni.txt' -numout 568 > ./afni/label/hrf/$subjname'lh_hrf.txt'
#waver -WAV -TR 0.72s -input ./afni/label/stim/$subjname'_rf_afni.txt' -numout 568 > ./afni/label/hrf/$subjname'rf_hrf.txt'
#waver -WAV -TR 0.72s -input ./afni/label/stim/$subjname'_lf_afni.txt' -numout 568 > ./afni/label/hrf/$subjname'lf_hrf.txt'
echo 'waver done'

3dDeconvolve -input ./afni/scaling/$subjname'_norm'+tlrc -mask ./afni/whole_mask/$subjname'_mask'+tlrc -num_stimts 5 -stim_file 1 ./afni/label/hrf/$subjname't_hrf.txt' -stim_label 1 tongue -stim_file 2 ./afni/label/hrf/$subjname'rh_hrf.txt' -stim_label 2 right_hand -stim_file 3 ./afni/label/hrf/$subjname'lh_hrf.txt' -stim_label 3 left_hand -stim_file 4 ./afni/label/hrf/$subjname'rf_hrf.txt' -stim_label 4 right_foot -stim_file 5 ./afni/label/hrf/$subjname'lf_hrf.txt' -stim_label 5 left_foot -num_glt 1 -glt_label 1 all_the_motor -gltsym 'SYM: +tongue +right_hand +left_hand +right_foot +left_foot' -fout -tout -bucket ./afni/glm/$subjname'_glm'

echo 'done'
cd .. 
done

