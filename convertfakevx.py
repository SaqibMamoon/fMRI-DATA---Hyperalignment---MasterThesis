#!/usr/bin/python

import numpy as np
import nibabel as nib
#This script convert a 2 d matrix with saptial info (3 column ) + 1 info per spatial location in a 3d matrix where x,y,z are the location for the info (basically nifti files)! So, after correlation is easy to create this mask and apply it directly to the main data and take the ts of the voxels we selected!

path = ('/home/andreabertana/Projects/HCP/results/MNI/')

nm_subj = np.array(['subj3','subj4','subj6','subj7','subj9','subj10'])	

#mask in native space of these subjects
for name in nm_subj:
	mask = np.zeros((91,109,91))
	print filename
	filename = (path + name + 'problems')
	bsc_matrices = np.array(np.loadtxt(filename))
	for row in np.arange(bsc_matrices.shape[0]):
		mask[bsc_matrices[row,0],bsc_matrices[row,1],bsc_matrices[row,2]] = 1

	img = nib.AnalyzeImage(mask, affine =None)
	nib.save(img, path + name + 'nifti.nii.gz')
	del(mask)
	del(img)
	

