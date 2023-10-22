"""
Importing the necessary libraries

"""

import dicom2nifti
import nibabel as nib
import nilearn as nil
import nilearn.image as nili
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import rotate

import skimage.transform as skt
from sklearn.model_selection import train_test_split

import torchio as tio
import torch
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf


os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
"""
Read the DICOM files and convert them to NIFTI format
Global variables:

patient_vals_train      -> list of training values  
patient_vals_train_gt   -> list of training ground truth values
patient_vals_test       -> list of testing values
patient_vals_test_gt    -> list of testing ground truth values


"""

patient_vols_train =[]
patient_vols_train_gt =[]
patient_vols_test =[]
patient_vols_test_gt =[]


original_affine = np.diag([-1, -1, 1, 1])


def load_nifti_files():
    for i in range(1, 101):
        for frame_num in range(1, 17): 
            if os.path.isfile(f'ACDC/database/training/patient{101-i:03}/patient{101-i:03}_frame{frame_num:02}.nii.gz'):
                patient_vols_train.append(nib.load(f'ACDC/database/training/patient{101-i:03}/patient{101-i:03}_frame{frame_num:02}.nii.gz'))
            if os.path.isfile(f'ACDC/database/training/patient{101-i:03}/patient{101-i:03}_frame{frame_num:02}_gt.nii.gz'):
                patient_vols_train_gt.append(nib.load(f'ACDC/database/training/patient{101-i:03}/patient{101-i:03}_frame{frame_num:02}_gt.nii.gz'))
        
        
    for i in range(101, 151):
        for frame_num in range(1, 17): 
            if os.path.isfile(f'ACDC/database/testing/patient{i}/patient{i}_frame{frame_num:02}.nii.gz'):
                patient_vols_test.append(nib.load(f'ACDC/database/testing/patient{i}/patient{i}_frame{frame_num:02}.nii.gz'))
            if os.path.isfile(f'ACDC/database/testing/patient{i}/patient{i}_frame{frame_num:02}_gt.nii.gz'):
                patient_vols_test_gt.append(nib.load(f'ACDC/database/testing/patient{i}/patient{i}_frame{frame_num:02}_gt.nii.gz'))

    return


"""
Data Preprocessing
    def max_int_resize_and_normalize:
    1. Convert to numpy array
    2. Set max intenzity 
    3. Resize and normalize
    4. Convert back to nifti format

    
"""
def numpy2nifti(numpy_img):   
    return nib.Nifti1Image(numpy_img, original_affine)

def nifti2numpy(nifti_img):
    return nifti_img.get_fdata()

def max_intenzity(image):
    min_val = np.min(image)
    max_val = np.max(image)
    min_max_norm_img = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return min_max_norm_img

def normalize(image):
    image = image.astype(np.float32)
    image -= np.mean(image)
    image /= np.std(image)
    return numpy2nifti(image)

# image and label
def max_int_resize_and_normalize_(numpy_img, label):
    
    resized_norm_data, resized_norm_label = skt.resize(max_intenzity(nifti2numpy(numpy_img)), (256,256,10), order=1, preserve_range=False,anti_aliasing=True), skt.resize(max_intenzity(nifti2numpy(label)), (256,256,10), order=1, preserve_range=False,anti_aliasing=True)
    resized_norm_data, resized_norm_label = normalize(resized_norm_data), normalize(resized_norm_label)
    
    return resized_norm_data, resized_norm_label 

# only image
def max_int_resize_and_normalize(numpy_img):
    
    resized_norm_data = skt.resize(max_intenzity(nifti2numpy(numpy_img)), (256,256,10), order=1, preserve_range=False,anti_aliasing=True)
    resized_norm_data = normalize(resized_norm_data)
    
    return resized_norm_data


"""
Define the functions for the nifti data augmentation
Every function get nifti image as input and return nifti image as output
Save new nifti images to the database
    
    1. Rotation
    2. Random gaussian noise
    3. Mirror around y axis
    4. Mirror around x axis
    
    
"""
def rotation_z_with_affine(image,rotation_degree=90):
    rotation_radians = np.radians(rotation_degree)
    rotation_affine = np.array([[np.cos(rotation_radians), -np.sin(rotation_radians), 0, 0],
                                [np.sin(rotation_radians), np.cos(rotation_radians), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    
    affine_so_far = image.affine.dot(rotation_affine)
    return nib.Nifti1Image(image, affine=affine_so_far)

def rotation_z(nifti_img,rotation_degree):
    img = nifti_img.get_fdata()
    rotated_data = rotate(img, rotation_degree, (0, 1), reshape=False)
    return nib.Nifti1Image(rotated_data, original_affine)

def smooth(nifti_img, fwhm):
    return nili.smooth_img(nifti_img, fwhm=fwhm)

# affine change
def mirror_y_with_affine(numpy_img):
    return nib.Nifti1Image(numpy_img, np.eye(4))

# npy data change
def mirror_y(nifti_img):
    mirrored_data = np.flip(nifti2numpy(nifti_img), axis=0)
    return nib.Nifti1Image(mirrored_data, original_affine)

# affine change
def mirror_x_with_affine(numpy_img):
    matrix = np.diag([-1, 1, 1, 1])
    return nib.Nifti1Image(numpy_img, matrix)

# npy data change
def mirror_x(nifti_img):
    mirrored_data = np.flip(nifti2numpy(nifti_img), axis=1)
    return nib.Nifti1Image(mirrored_data, original_affine)
    

"""
dataset, dataloader

"""

def get_db_size(dataset):
    print('Dataset size:', len(dataset), 'subjects') 
    return len(dataset) 

def create_dataloader(X, y, batch_size):
    images = torch.Tensor(X)
    ground_truths = torch.Tensor(y)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # shuffle the data
    
    return dataloader
def test_dataloader(X,batch_size):
    test_images = torch.Tensor(X)
    test_dataset = TensorDataset(test_images)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader
    
def eval(dataset, model):
    model.eval()
    with torch.no_grad():  # Kikapcsoljuk a gradiens számítást a tesztelés során
        for inputs in dataset:
            outputs = model(inputs)
    return outputs
    
"""
1. Read all the nifti images(train and test) and store them in lists
2. Prepare the data for the network(reshape, normalize)
3. Split the data to train, test and validation
4. Data augmentation 
5. Save the normal and augmented data to npy files

"""

if __name__ == "__main__":
    #Load the nifti files into lists(nifti format)
    load_nifti_files()

        #Max intenzity,Resize and Normalize all images and ground truth images
    for i in range(len(patient_vols_train)):
        patient_vols_train[i], patient_vols_train_gt[i] = max_int_resize_and_normalize_(patient_vols_train[i], patient_vols_train_gt[i])
    for i in range(len(patient_vols_test)):
        patient_vols_test[i], patient_vols_test_gt[i] = max_int_resize_and_normalize_(patient_vols_test[i], patient_vols_test_gt[i])  

    # Data augmentation
        # rotate and mirror the images, then add random noise to random images
        # every train image has 2 rotated,2 mirrored and 1 noisy version
        # make the same with the ground truth images
        # 25*5 = 125 
    augmented_img_train = []
    augmented_img_test = []
    augmented_gt_train = []
    augmented_gt_test = []
    
    np.random.seed(42)  
    random_numbers = np.random.randint(0,200, size=25)

    for j in range(0,25):
        i=random_numbers[j]
        augmented_img_train.append(rotation_z(patient_vols_train[i], 90))
        augmented_img_train.append(rotation_z(patient_vols_train[i], 270))
        augmented_img_train.append(mirror_y(patient_vols_train[i]))
        augmented_img_train.append(mirror_x(patient_vols_train[i]))
        augmented_img_train.append(smooth(patient_vols_train[i], 2))
        augmented_gt_train.append(rotation_z(patient_vols_train_gt[i], 90))
        augmented_gt_train.append(rotation_z(patient_vols_train_gt[i], 270))
        augmented_gt_train.append(mirror_y(patient_vols_train_gt[i]))
        augmented_gt_train.append(mirror_x(patient_vols_train_gt[i]))
        augmented_gt_train.append(smooth(patient_vols_train[i], 2))
        
        # for the test images we make the same
        # 10*5 
    random_numbers = np.random.randint(0,100, size=10)
    for j in range(0,10):  
        i=random_numbers[j] 
        augmented_img_test.append(rotation_z(patient_vols_test[i], 90))
        augmented_img_test.append(rotation_z(patient_vols_test[i], 270))
        augmented_img_test.append(mirror_y(patient_vols_test[i]))
        augmented_img_test.append(mirror_x(patient_vols_test[i]))
        augmented_img_test.append(smooth(patient_vols_test[i],2))
        augmented_gt_test.append(rotation_z(patient_vols_test_gt[i], 90))
        augmented_gt_test.append(rotation_z(patient_vols_test_gt[i], 270))
        augmented_gt_test.append(mirror_y(patient_vols_test_gt[i]))
        augmented_gt_test.append(mirror_x(patient_vols_test_gt[i]))
        augmented_gt_test.append(smooth(patient_vols_test_gt[i],2)) 
        
    X_train = []
    for img in patient_vols_train:
        for i in range(1,img.shape[-1],2):
            new_data = np.stack((img.get_fdata()[:,:,i],)*3, axis=-1).astype('float32')
            X_train.append(new_data)
            
    X_train_aug = []
    for img in augmented_img_train:
        for i in range(1,img.shape[-1],2):
            new_data = np.stack((img.get_fdata()[:,:,i],)*3, axis=-1).astype('float32')
            X_train_aug.append(new_data)
            
        
    Y_train = []
    for img in patient_vols_train_gt:
        for i in range(1,img.shape[-1],2):
            new_data = img.get_fdata()[:,:,i].astype('float32')
            Y_train.append(new_data)
    Y_train_aug = []
    for img in augmented_gt_train:
        for i in range(1,img.shape[-1],2):
            new_data = img.get_fdata()[:,:,i].astype('float32')
            Y_train_aug.append(new_data)
                
    X_test = []
    for img in patient_vols_test:
        for i in range(1,img.shape[-1],2):
            new_data = np.stack((img.get_fdata()[:,:,i],)*3, axis=-1).astype('float32')
            X_test.append(new_data)
    X_test_aug = []
    for img in augmented_img_test:
        for i in range(1,img.shape[-1],2):
            new_data = np.stack((img.get_fdata()[:,:,i],)*3, axis=-1).astype('float32')
            X_test_aug.append(new_data)

    y_test = []
    for img in patient_vols_test_gt:
        for i in range(1,img.shape[-1],2):
            new_data = img.get_fdata()[:,:,i].astype('float32')
            y_test.append(new_data)
    y_test_aug = []   
    for img in augmented_gt_test:
        for i in range(1,img.shape[-1],2):
            new_data = img.get_fdata()[:,:,i].astype('float32')
            y_test_aug.append(new_data)     

    X_train = np.array(X_train)
    X_train_aug = np.array(X_train_aug)
    Y_train = np.array(Y_train)
    Y_train_aug = np.array(Y_train_aug)
    X_test = np.array(X_test)
    X_test_aug = np.array(X_test_aug)
    y_test = np.array(y_test)
    y_test_aug = np.array(y_test_aug)


    Y_train = np.expand_dims(Y_train, axis=3)
    Y_train_aug = np.expand_dims(Y_train_aug, axis=3)
    y_test = np.expand_dims(y_test, axis=3)
    y_test_aug = np.expand_dims(y_test_aug, axis=3)

    """
    X_train shape: (1000, 256, 256, 3)
    X_train_aug shape: (625, 256, 256, 3)
    y_train shape: (1000, 256, 256, 1)
    y_train_aug shape: (625, 256, 256, 1)
    X_test shape: (500, 256, 256, 3)
    X_test_aug shape: (250, 256, 256, 3)
    y_test shape: (500, 256, 256, 1)
    y_test_aug shape: (250, 256, 256, 1)

    """

    np.save('X_train.npy', X_train)
    np.save('Y_train.npy', Y_train)
    np.save('X_train_aug.npy', X_train_aug)
    np.save('Y_train_aug.npy', Y_train_aug)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    np.save('X_test_aug.npy', X_test_aug)
    np.save('y_test_aug.npy', y_test_aug)