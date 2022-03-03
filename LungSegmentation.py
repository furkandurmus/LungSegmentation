# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 03:09:25 2020

@author: furkan
"""

import numpy as np
import pydicom as dicom
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology, segmentation
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('log_file.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class Segment():
    def __init__(self, path, image_type=np.int16):
        self.path = path
        self.image_type = image_type
        logger.info(f"Path of the file: {self.path}")
        
        
    def load_data(self):
        self.slices = [dicom.read_file(self.path + '/' + s) for s in os.listdir(self.path)]
        
        """ HU value = pixel_value*RescaleSlope + Intercept"""
        self.images = np.stack([s.pixel_array for s in self.slices])
        self.images = self.images.astype(self.image_type)
        self.images = self.images*self.slices[0].RescaleSlope + self.slices[0].RescaleIntercept      
        return self.images
                
        

    def get_meta_data(self):
        self.metadata = {
            "intercept": self.slices[1].RescaleIntercept,
            "slope": self.slices[1].RescaleSlope
            }
    
    def ornek_kesit(self,i):
        images = self.load_data()
        scan_sample = images[i]
        plt.imshow(scan_sample, cmap = 'gray')
        plt.show()
        
    

        
    
    def generate_marker(self):
        self.image = self.load_data()
        marker_internal = self.image < -400
        marker_external = np.empty_like(marker_internal)
        marker_watershed = np.empty_like(marker_internal).astype(np.int32)

        for i in range(len(self.images)):
            marker_internal[i,:,:] = segmentation.clear_border(marker_internal[i,:,:])
            marker_internal_labels = measure.label(marker_internal[i])
            areas = [r.area for r in measure.regionprops(marker_internal_labels)]
            areas.sort()
            if len(areas) > 2:
                for region in measure.regionprops(marker_internal_labels):
                    if region.area < areas[-2]:
                        for coordinates in region.coords:                
                            marker_internal_labels[coordinates[0], coordinates[1]] = 0
                            marker_internal[i] = marker_internal_labels > 0

            external_a = ndimage.binary_dilation(marker_internal[i], iterations=10)
            external_b = ndimage.binary_dilation(marker_internal[i], iterations=55)
            marker_external[i] = external_b ^ external_a

            marker_watershed[i] = np.zeros((self.image.shape[1], self.image.shape[2]), dtype=np.int)

            marker_watershed[i] += marker_internal[i] * 255
            marker_watershed[i] += marker_external[i] * 128
            
        return marker_internal, marker_external, marker_watershed
    
    

    
    def separate_lung(self):
        #Creation of the markers as shown above:
        # breakpoint()
        marker_internal, marker_external, marker_watershed = self.generate_marker()
        segmented = np.empty_like(marker_watershed).astype(self.image_type)
        lungfilter = np.empty_like(marker_watershed).astype(self.image_type)
        logger.info("Segmentation is started...")
        logger.info(f"Number of dicom images in current folder:  {len(self.images)}")
        try:
            for i in tqdm(range(len(self.images))):
                sobel_filtered_dx = ndimage.sobel(self.image[i], 1)
                sobel_filtered_dy = ndimage.sobel(self.image[i], 0)
                sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
                sobel_gradient *= 255.0 / np.max(sobel_gradient)
                
                #Watershed algorithm
                # watershed = morphology.watershed(sobel_gradient, marker_watershed[i])
                watershed = segmentation.watershed(sobel_gradient, marker_watershed[i])

                
                #Reducing the image created by the Watershed algorithm to its outline
                outline = ndimage.morphological_gradient(watershed, size=(3,3))
                outline = outline.astype(bool)
                
                #Performing Black-Tophat Morphology for reinclusion
                #Creation of the disk-kernel and increasing its size a bit
                blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 1, 1, 0],
                                    [0, 0, 1, 1, 1, 0, 0]]
                # blackhat_struct = [[1, 1, 1],
                #                     [1, 1, 1],
                #                     [1, 1, 1]]                
                blackhat_struct = ndimage.iterate_structure(blackhat_struct, 4)
                #Perform the Black-Hat
                outline += ndimage.black_tophat(outline, structure=blackhat_struct)
                
                #Use the internal marker and the Outline that was just created to generate the lungfilter
                lungfilter[i] = np.bitwise_or(marker_internal[i], outline)
                #Close holes in the lungfilter
                #fill_holes is not used here, since in some slices the heart would be reincluded by accident
                lungfilter[i] = ndimage.morphology.binary_closing(lungfilter[i], structure=np.ones((5,5)), iterations=3)
                
                #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
                segmented[i] = np.where(lungfilter[i] == 1, self.image[i], -2000*np.ones((self.image[i].shape[0], self.image[i].shape[1])))
            
            # return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed
        except Exception as err:
            print(f"{err} is occured!!")
        return lungfilter, segmented



        
        
        


   
# S  = Segment(INPUT) 
# ornek = S.slices[225].pixel_array

#to get a sample from original data give integer as slice number
# S.ornek_kesit(10) 
#segmented_images değişkenine o pathdeki image çıktılarını aldık
# segmented_images = S.separate_lung()
"""ornek 225. kesitimiz. thresholdlayarak maske oluiturduk a ile
basit bi averaging filter ile de denoise ettik.
 """
# a = ornek < -250
# kernel = np.ones((5,5),np.float32)/25
# dst = cv2.filter2D(np.int16(a),-1,kernel)


# marker_internal = segmentation.clear_border(ornek)
# marker_internal_labels = measure.label(marker_internal)
# areas = [r.area for r in measure.regionprops(marker_internal_labels)]


