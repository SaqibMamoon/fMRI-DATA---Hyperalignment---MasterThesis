#!/usr/bin/python
		

import numpy as np
import os.path
import nibabel as nib

path_save ='/home/andreabertana/Projects/haxby_8cat/results/detrend/pymvpa_det/mat_vxts/'
nm_subj = np.array(['subj1','subj2','subj3','subj4','subj5'])	
hemis = np.array(['dx','sx'])
for i in nm_subj:
	print i
	path_f = os.path.join=
	path_m = os.path.join	
	for j in hemis: 
		print j
		defpath_m = path_m + j + '.nii.gz'
		mask = nib.load(defpath_m)
		img = nib.load(path_f)
		idx = np.array(np.nonzero(1*(mask.get_data()>0)))
		img_data = img.get_data()		
		matrix =  np.zeros(shape=(np.size(idx,1),1455))
		for v in np.arange(np.size(idx,1)): 
			print v
			xyz = idx[:,v]
			print xyz
			matrix[v,0:3] = xyz
			matrix[v,3:]=img_data[xyz[0],xyz[1],xyz[2],:]
		np.savetxt(path_save +'matrix_vxts_' + i + j + '.txt', matrix)

print 'All subject has been done' 
 

 











