# run with !bash unet_setup.sh

# Remove unnecessary folder in Colab (use only if needed)
# rm -rf sample_data

# Install required libraries
pip install -U segmentation_models dicom2nifti nilearn --quiet

# Download the pretrained model
gdown https://drive.google.com/uc?id=1-D7xDCVeJs5QNgQaFKnCfqGarl-iMLx4 -O /content/unet_pre-trained.h5

# Download the scripts from google drive
gdown https://drive.google.com/uc?id=1ELdyDnqglXtg6MMXUZhQxdIhYslgzSKF -O /content/unet_train.py
gdown https://drive.google.com/uc?id=1F2EqM3dlCgPhzn-2BITObzCdNjBuj0zY -O /content/unet_test.py
# gdown https://drive.google.com/uc?id=1-D7xDCVeJs5QNgQaFKnCfqGarl-iMLx4 -O /content/unet_predict.py

# Download the correct __init__.py file and replace the wrong one
gdown https://drive.google.com/uc?id=10D02qBmF0MYd1g8lpD4obDWd_ucr83cj -O /content/__init__.py
cp /content/__init__.py /usr/local/lib/python3.10/dist-packages/efficientnet/__init__.py
# sudo cp /content/__init__.py /usr/local/lib/python3.10/dist-packages/efficientnet/__init__.py
