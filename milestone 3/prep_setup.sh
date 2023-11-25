# run with !bash prep_setup.sh

# Remove unnecessary folder in Colab (use only if needed)
# rm -rf sample_data

# Install required libraries
pip install -U segmentation_models dicom2nifti nilearn --quiet

# Download the original dataset, and the prep.py script from google drive
gdown https://drive.google.com/uc?id=1qAUJtiPZfT3jm4V4qg-VRVucpZva-Qj_ -O /content/ACDC.zip && unzip -q /content/ACDC.zip -d /content
gdown https://drive.google.com/uc?id=1N9nNwe8D-RZqlVKvuk5La8RnL_YT8rHx -O /content/prep.py

# Execute the prep.py script
python prep.py
