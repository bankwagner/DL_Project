# Install required libraries
pip install -U segmentation_models dicom2nifti nilearn --quiet

# Download the pretrained model
gdown https://drive.google.com/file/d/1_C-tAH4wee_fkPdO8INrvZ2IGpDW8xh0/edit -O /content/fpn_pre-trained.h5 --fuzzy

# Download the scripts from google drive
gdown https://drive.google.com/file/d/1G59hS3z1hLUlTGpabZrbtsRFF71eKdm1/edit -O /content/fpn_train.py --fuzzy
python fpn_train.py