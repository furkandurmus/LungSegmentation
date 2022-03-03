# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 01:51:30 2021

@author: furkan
"""

# from cgitb import handler

# from isort import stream
from LungSegmentation import Segment
import os
import numpy as np
import argparse
from glob import glob

import logging

parser = argparse.ArgumentParser()


parser.add_argument("--data_folder", default="d:\\AIinHealthCare\\images\\*\\Ham", help="path of the dicom image folder")
parser.add_argument("--output_folder", default="d:\\AIinHealthCare\\cikti\\", help="folder to save output files")
parser.add_argument("--image_type", default=np.int16, help="numpy image type")

opt = parser.parse_args()

if not os.path.exists(opt.output_folder):
    os.makedirs(opt.output_folder)
    print("Output Folder Created!!")




images = glob("d:\AIinHealthCare\images\*")
labels = glob("d:\AIinHealthCare\labels\*")


#Check if image and label folders have same patient folders
# assert([os.path.basename(i) for i in images] == [os.path.basename(i) for i in labels])


for i in images:
    SegmentObject = Segment(i, opt.image_type)
    filtered_lungs, segmented_lungs = SegmentObject.separate_lung()
    # segment_object.ornek_kesit(3)
    np.save(opt.output_folder + f"{os.path.basename(i)}_Image.npy", segmented_lungs)
    np.save(opt.output_folder + f"{os.path.basename(i)}_Label.npy", filtered_lungs)



# if __name__ == '__main__':

#     segment_object = Segment(opt.data_folder, opt.image_type)
#     filtered_lungs, segmented_lungs = segment_object.separate_lung()
#     # segment_object.ornek_kesit(3)
#     np.save(opt.output_folder + "segmented_lungs.npy", segmented_lungs)
#     np.save(opt.output_folder + "filtered_lungs.npy", filtered_lungs)




















