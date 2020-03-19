#!/usr/bin/python

import nibabel as nib
import scipy.signal
import os

path_main = '/home/andreabertana/Projects/HCP/results/MNI/mc/'
subj = os.listdir('./Projects/HCP/MNI/')    
for s in subj:
    path = path_main + s + '_mc.nii.gz'
    print path
    print "Loading data"    
    fmri = nib.load(path)
    data = fmri  #.data.astype('float32')
    #fmri= unload()
	
    #detrend. Axis 0 should be time.bp should run detrend taking 12 different parts,each with 121vol (12 steps: 		1452/121 = 12) that respect one block in Haxby design.
    print "Detrending"
    detrended= scipy.signal.detrend(data.get_data(), axis =3,type= 'linear', bp=[405])
    filename = ('/home/andreabertana/Projects/HCP/results/MNI/detrend/' + s + '_detrend.nii.gz')
    print 'Store detrended data for later re-use'
    affine = data.get_affine()
    img = nib.AnalyzeImage(detrended,affine=affine)
    nib.save(img,filename)
print "Done."
print "All Done"



