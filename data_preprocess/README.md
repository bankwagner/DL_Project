# Description
  - This folder contains the file(prep.ipynb) for the preprocessing and data augmentation.    
  - Run the cells
  - The results (.npy files) will saved in the base_model directory
# Required file structure
<pre> 
data_preprocess/
    └── ACDC/
        └── database/
            ├── training/
            │   ├── patient001/
            │   │   ├── patient001_frame01.nii
            │   │   ├── patient001_frame01_gt.nii
            │   │   └── ...
            │   └── ...
            └── testing/
                ├── patient101/
                │   ├── patien1001_frame01.nii
                │   └── patient101_frame01_gt.nii
                └── ...
</pre>
