# run with !bash ensemble_setup.sh

# Remove unnecessary folder in Colab (use only if needed)
# rm -rf sample_data

# Download the ensemble.py script from google drive
gdown https://drive.google.com/uc?id=1V9aEl6ExCnwiUz0Ig8V5PsimPBs0Uj6j -O /content/ensemble.py

# Download the models from google drive - comment out the ones you don't need
gdown https://drive.google.com/uc?id=1-D7xDCVeJs5QNgQaFKnCfqGarl-iMLx4 -O /content/unet_pre-trained.h5
gdown https://drive.google.com/uc?id=1QwfRAiRYlQr3gIKzMQgcrkyJy-6uXDI6 -O /content/fpn_pre-trained.h5
gdown https://drive.google.com/uc?id=1mDbP_60OPpXYiQ6KE2oysSEwSwovdu0a -O /content/linknet_pre-trained.h5