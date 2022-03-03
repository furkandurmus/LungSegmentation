# Lung Segmentation

Segmentation of lungs from CT images. Run the code by passing arguments to the main.py. The segmented lung and its masks are stored in the --data_output folder as np int16 format.


Data folder should be organized as below;

```bash
DATA FOLDER                
 ├──  images
 |      ├──Patient_0
 |           ├── 1.DICOM     
 |           ├── ...
 |           └── n.DICOM
 |      ├──Patient_1
 |           ├── 1.DICOM     
 |           ├── ...
 |           └── n.DICOM
 |      ├──Patient_2
 |           ├── 1.DICOM     
 |           ├── ...
 |           └── n.DICOM

```

A sample output slice can be obtained by running ShowResults.py;

![2022-03-04_02-19-10](https://user-images.githubusercontent.com/12261453/156670425-8971b03e-cbf6-4e33-97aa-0da563783517.png)



I got the Watershed segmentation algorithm from this Kaggle post: https://www.kaggle.com/ankasor/improved-lung-segmentation-using-watershed
