"""
    How to run? put the image as a 2. argv:
    
    !python prep_test.py test_image
    
    This file will preprocess the test image
    Args:
         The input image (jpg,jpeg,png,PIL.Image)
        

    Returns:
        resized and normalized image as "test_image.npz" compressed numpy array with shape of (256,256,3)
"""

import sys
import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.transform as skt
import cv2
from scipy.ndimage import rotate
from PIL import Image


def max_intenzity(image):
    """
    Normalize the intensity values of an image to the range [0, 255].

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The normalized, max intenzity image.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    min_max_norm_img = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return min_max_norm_img


def normalize(image):
    """
    Normalize the intensity values of an image to the range [0, 1].

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The normalized image.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    norm_img = ((image - min_val) / (max_val - min_val))
    return norm_img


def max_int_resize_and_normalize(numpy_img):
    """
    Resize, normalize, and convert the input image 

    Args:
        numpy_img : The input image.
        

    Returns:
        resized and normalized image
    """
    
    resized_norm_data = skt.resize(max_intenzity(numpy_img), (256,256,3), order=1, preserve_range=False,anti_aliasing=True)
    resized_norm_data = normalize(resized_norm_data)

    return resized_norm_data



if __name__ == "__main__":
    print(" - - - preprocessing started - - - ")

    # Set environment variable for the SM_FRAMEWORK
    os.environ["SM_FRAMEWORK"] = "tf.keras"

    if len(sys.argv) != 2:
        print("Usage: python prep_test.py image_path")
    else:
        # Get the image file path from command-line argument
        image = sys.argv[1]

    # if the image are not in PIL.Image format, convert them 
    if image.endswith(".jpg") or image.endswith(".png") or image.endswith(".jpeg"):
        image = tf.keras.preprocessing.image.load_img(image)
        image = max_int_resize_and_normalize(image)
    
    #if the image are in PIL format, convert them to numpy format
    elif isinstance(image, Image.Image):
        image = max_int_resize_and_normalize(image)
    
    if image.shape != (256, 256, 3):
        print("Error: wrong image size.")
        sys.exit(1)

    # clip pixel values as a safety measure
    image = np.clip(image, 0, 1)
    
    # Print information about the preprocessed data
    if image is not None:
        print("Image converted to NumPy array successfully.")
        print("Data type of the array:", image.dtype)
        print("test_image:\t shape: ", image.shape, "\tmin: ", np.min(image), "max: ", np.max(image))
     
    # Save the preprocessed data to disk   
    np.savez_compressed("test_image.npz", image)
    del image
    gc.collect()
    
    print(" - - - preprocessing finished - - - ")
    
    sys.exit(0)

    
