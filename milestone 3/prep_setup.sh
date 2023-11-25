# run with !bash prep_setup.sh

# Remove unnecessary folder in Colab (use only if needed)
# rm -rf sample_data

# Install required libraries
pip install -U segmentation_models dicom2nifti nilearn --quiet

# Download the original dataset
gdown https://drive.google.com/uc?id=1qAUJtiPZfT3jm4V4qg-VRVucpZva-Qj_ -O /content/ACDC.zip && unzip -q /content/ACDC.zip -d /content

# Download the prep.py script from the GitHub repository
wget https://raw.githubusercontent.com/bankwagner/DL_Project/3fb1463d2dbd5596d775063b9f6504c031b182a8/milestone%202/prep.py

# Execute the prep.py script
python prep.py
