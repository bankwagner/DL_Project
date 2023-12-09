# Install required libraries
pip install -U segmentation_models dicom2nifti nilearn --quiet

# Download the pretrained model
gdown https://drive.google.com/file/d/1J8hIRo-GT6XrkWemIum8DjRisjff9lca/edit -O /content/unet_pre-trained.h5 --fuzzy

# Download the scripts from google drive
gdown https://drive.google.com/file/d/1IdoIA_YLQ_Wmmjlt23XpvUNBKkQs4OJg/edit -O /content/unet_train.py --fuzzy
python unet_train.py