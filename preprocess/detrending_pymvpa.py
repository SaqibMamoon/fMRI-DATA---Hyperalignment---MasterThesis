#!/usr/bin/python

from mvpa2.suite import *
import pandas as pnd
import sys

path ='/home/andreabertana/Projects/HCP/results/MNI/MOTOR/'
#subj = os.listdir('/home/andreabertana/Projects/HCP/MNI/') 
#for name in subj:
cli_argv = sys.argv 
#pathdata = path + 'mc/672756_mc.nii.gz'
pathdata = cli_argv[1]
print pathdata
#pathlabel = path + 'label/concatenated/672756_labels.txt'
pathlabel = cli_argv[2]
print pathlabel
pathmask = cli_argv[3]
#open label
#dataframe = pnd.read_csv(pathlabel,sep = ' ', header = 0)
#chunks = dataframe.Run
#targets = dataframe.Category
label = SampleAttributes(pathlabel)
#open data for pymvpa, with targets(catgories) and chunks(runs) info
data = fmri_dataset(samples = pathdata, chunks = label.chunks,  targets = label.targets, mask = pathmask)
#definfe a detrender, here is chunk-wise linear	
detrender = PolyDetrendMapper(polyord = 1 , chunks_attr='chunks')
#apply detrending on dataset
det = data.get_mapped(detrender)
#save files , defining the reversemapper
save = map2nifti(det)
#save.to_filename(path + '/detrend/672756_det_mc.nii.gz')
save.to_filename(cli_argv[4])
print 'the file has been saved'
del(data)
print 'data has been deleted'
del(det)
print 'save has been deleted'
del(save)

print 'All subject are done'

