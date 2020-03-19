# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:18:11 2014

@author: andreabertana
"""

import numpy as np
import matplotlib.pyplot as plt


n_groups = 3

whithin = (44,41,48) 
std_whit = (0.46, 0.64, 0.5 )

anatomical = (49, 61 ,55 )
std_ana = (0.62,0.28, 0.37)

hyper = (65, 73 , 64)
std_hyper = (0.39, 0.49, 0.61)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2

opacity = 0.6
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, whithin, bar_width,
                 alpha=opacity,
                 color='b',
                 yerr=std_whit,
                 error_kw=error_config,
                 label='Whithin')

rects2 = plt.bar(index + bar_width, anatomical, bar_width,
                 alpha=opacity,
                 color='r',
                 yerr=std_ana,
                 error_kw=error_config,
                 label='Anatomical')

rects3 = plt.bar(index + bar_width*2, hyper, bar_width,
                 alpha=opacity,
                 color='g',
                 yerr=std_hyper,
                 error_kw=error_config,
                 label='Hyperalignment')

plt.xlabel('Preprocess')
plt.ylabel('Accuracy %')
plt.title('SVM classification with ANOVA')
plt.xticks(index + 0.3, ('MC', 'Det', 'Smo'))
plt.legend()

plt.tight_layout()
plt.show()