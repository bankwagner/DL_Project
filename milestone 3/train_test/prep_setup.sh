# Remove unnecessary folder in Colab (use only if needed)
# rm -rf sample_data

# Install required libraries
pip install -U segmentation_models dicom2nifti nilearn --quiet

# Download the original dataset, and the prep.py script from google drive
gdown https://drive.google.com/file/d/1qAUJtiPZfT3jm4V4qg-VRVucpZva-Qj_/edit -O /content/ACDC.zip --fuzzy && unzip -q /content/ACDC.zip -d /content
gdown https://drive.google.com/file/d/17qlna9cM-kI4B7XohwSHwPoD5jx0-Qlx/edit -O /content/prep.py --fuzzy

# Execute the prep.py script
python prep.py