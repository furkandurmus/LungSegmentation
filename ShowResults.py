# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 02:51:27 2021

@author: furkan
"""
import numpy as np
from matplotlib import pyplot as plt
from LungSegmentation import Segment
import os

raw = r"d:\AIinHealthCare\cikti\segmented_lungs.npy"

pro = r"d:\AIinHealthCare\cikti\filtered_lungs.npy"

data_folder = "D:/AIinHealthCare/AR_SER5_V10_2015/Ham"


segment_object = Segment(data_folder)
segmented_lungs = np.load(raw)
filtered_lungs = np.load(pro)


indx = 1
plt.figure()
f, axarr = plt.subplots(3,1) 
axarr[0].imshow(segmented_lungs[indx], cmap='gray')
axarr[1].imshow(filtered_lungs[indx], cmap='gray')
segment_object.ornek_kesit(indx)
# plt.show()

