# Install required libraries
pip install -U segmentation_models dicom2nifti nilearn --quiet

# Download the pretrained model
gdown https://drive.google.com/file/d/1H0NnKsYLr8l5o1Xe5vfUiZj4Y6STWUKB/edit -O /content/linknet_pre-trained.h5 --fuzzy

# Download the scripts from google drive
gdown https://drive.google.com/file/d/1JGF5ySYY9qbPTwVJnX5d126d5FIjMmeQ/edit -O /content/linknet_train.py --fuzzy
python linknet_train.py