# run with !bash unet_setup.sh

# Remove unnecessary folder in Colab (use only if needed)
# rm -rf sample_data

# Install required libraries
pip install -U segmentation_models dicom2nifti nilearn --quiet

# Download the pretrained model
gdown https://drive.google.com/uc?id=1-D7xDCVeJs5QNgQaFKnCfqGarl-iMLx4 -O /content/unet_pre-trained.h5

# Download the scripts from the GitHub repository
wget https://raw.githubusercontent.com/bankwagner/DL_Project/???/prep.py
wget https://raw.githubusercontent.com/bankwagner/DL_Project/???/unet_train.py
wget https://raw.githubusercontent.com/bankwagner/DL_Project/???/unet_test.py
# wget https://raw.githubusercontent.com/bankwagner/DL_Project/???/unet_predict.py