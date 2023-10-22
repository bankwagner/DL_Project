# Description
  - This folder contains the file(prep.py) for the preprocessing and data augmentation.  
  - You can download the ACDC dataset from here: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb  
  - Extract the file in this directory and run the prep.ipynb
# Required file structure
***
.
└── data_preprocess/
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
***
