#!/bin/bash
 #this script should be run from andreabertana dir


for subjname in $(cat ./Projects/HCP/results/list_subject_first10.txt)
do
#subjname="100307"

#mc and merge
#should be lunched merge&mc.sh that iterate over subjects in the dir /home/andreabertana/Projects/HCP/MNI/
#echo ./Projects/HCP/pipeline/merge_mc.sh $subjname 

#detrending on raw data
#python script detrending_pymvpa.py that iterate over subject in same dir /home/andreabertana/Projects/HCP/MNI/
#./Projects/HCP/pipeline/detrending_pymvpa.py /home/andreabertana/Projects/HCP/results/MNI/merge/$subjname'_merge.nii.gz' /home/andreabertana/Projects/HCP/results/MNI/label/concatenated/$subjname'_labels.txt' /home/andreabertana/Projects/HCP/results/MNI/detrend/raw/$subjname'_det_raw.nii.gz'

#detrending on mc
#python script detrending_pymvpa.py 
./Projects/HCP/pipeline/detrending_pymvpa.py /home/andreabertana/Projects/HCP/results/MNI/mc/$subjname'_mc.nii.gz' /home/andreabertana/Projects/HCP/results/MNI/label/concatenated/$subjname'_labels.txt' /home/andreabertana/Projects/HCP/results/MNI/detrend/vt/harvard/$subjname'_det_mc.nii.gz'

#apply_mask on mc
#should be applied on mc and mc + det (input and output to specify)
#echo ./Projects/HCP/pipeline/applymask.sh /home/andreabertana/Projects/HCP/results/MNI/mc/$subjname'_mc.nii.gz' /home/andreabertana/Projects/HCP/results/MNI/mc/vt/$subjname'_mc_vt.nii.gz' 

#apply_mask on mc + det
#should be applied on mc and mc + det (input and output to specify)
#echo ./Projects/HCP/pipeline/mask/apply_mask.sh /home/andreabertana/Projects/HCP/results/MNI/detrend/$subjname'_det_mc.nii.gz' /home/andreabertana/Projects/HCP/results/MNI/detrend/$subjname'_det_mc_vt.nii.gz' 


#smoothing on mc 
#shell. input and output are writte below and should be manually selected depending the file that has to be smoothed
#echo ./Projects/HCP/pipeline/smoothing.sh /home/andreabertana/Projects/HCP/results/MNI/mc/vt/$subjname'_mc_vt.nii.gz' /home/andreabertana/Projects/HCP/results/MNI/smoothing/mc/$subjname'_mc_smo_vt.nii.gz' 

#smoothing on det
#shell. input and output are writte below and should be manually selected depending the file that has to be smoothed
#	echo ./Projects/HCP/pipeline/smoothing.sh /home/andreabertana/Projects/HCP/results/MNI/detrend/mc/vt/$subjname'_det_mc_vt.nii.gz' /home/andreabertana/Projects/HCP/results/MNI/smoothing/det/$subjname'_det_mc_smo_vt.nii.gz'

done	
