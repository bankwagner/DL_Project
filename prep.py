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

from scipy.ndimage import rotate

import skimage.transform as skt
from sklearn.model_selection import train_test_split

import torchio as tio
import torch
from torch.utils.data import DataLoader, TensorDataset

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
patient_vals_train_4d   -> list of training 4d values
patient_vals_test_4d    -> list of testing 4d values

"""

patient_vols_train =[]
patient_vols_train_gt =[]
patient_vols_test =[]
patient_vols_test_gt =[]
patient_vols_train_4d =[] 
patient_vols_test_4d =[] 

original_affine = np.diag([-1, -1, 1, 1])

augmented_img_train = []
augmented_img_test = []
augmented_gt_train = []
augmented_gt_test = []


def load_nifti_files():
    for i in range(1, 101):
        for frame_num in range(1, 17): 
            if os.path.isfile(f'ACDC/database/training/patient{101-i:03}/patient{101-i:03}_frame{frame_num:02}.nii.gz'):
                patient_vols_train.append(nib.load(f'ACDC/database/training/patient{101-i:03}/patient{101-i:03}_frame{frame_num:02}.nii.gz'))
            if os.path.isfile(f'ACDC/database/training/patient{101-i:03}/patient{101-i:03}_frame{frame_num:02}_gt.nii.gz'):
                patient_vols_train_gt.append(nib.load(f'ACDC/database/training/patient{101-i:03}/patient{101-i:03}_frame{frame_num:02}_gt.nii.gz'))

        if(os.path.isfile(f'ACDC/database/training/patient{101-i:03}/patient{101-i:03}_4d.nii.gz')):     
            patient_vols_train_4d.append(nib.load(f'ACDC/database/training/patient{101-i:03}/patient{101-i:03}_4d.nii.gz'))
        
        
    for i in range(101, 151):
        for frame_num in range(1, 17): 
            if os.path.isfile(f'ACDC/database/testing/patient{i}/patient{i}_frame{frame_num:02}.nii.gz'):
                patient_vols_test.append(nib.load(f'ACDC/database/testing/patient{i}/patient{i}_frame{frame_num:02}.nii.gz'))
            if os.path.isfile(f'ACDC/database/testing/patient{i}/patient{i}_frame{frame_num:02}_gt.nii.gz'):
                patient_vols_test_gt.append(nib.load(f'ACDC/database/testing/patient{i}/patient{i}_frame{frame_num:02}_gt.nii.gz'))

        if(os.path.isfile(f'ACDC/database/testing/patient{i}/patient{i}_4d.nii.gz')):     
            patient_vols_test_4d.append(nib.load(f'ACDC/database/testing/patient{i}/patient{i}_4d.nii.gz'))
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
    
    resized_norm_data, resized_norm_label = skt.resize(max_intenzity(nifti2numpy(numpy_img)), (216,256,10), order=1, preserve_range=False,anti_aliasing=True), skt.resize(max_intenzity(nifti2numpy(label)), (216,256,10), order=1, preserve_range=False,anti_aliasing=True)
    resized_norm_data, resized_norm_label = normalize(resized_norm_data), normalize(resized_norm_label)
    
    return resized_norm_data, resized_norm_label 

# only image
def max_int_resize_and_normalize(numpy_img):
    
    resized_norm_data = skt.resize(max_intenzity(nifti2numpy(numpy_img)), (216,256,10), order=1, preserve_range=False,anti_aliasing=True)
    resized_norm_data = normalize(resized_norm_data)
    
    return resized_norm_data




# Az átméretezett kép létrehozása
#resized_image = nib.Nifti1Image(resized_data, affine=original_image.affine)



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
store the augmented nifti images to the list
"""
def save_nifti_to_database(nifti_img, type):
    if(type == "train"):
        augmented_img_train.append(nifti_img)
    if(type == "train_gt"):
        augmented_gt_train.append(nifti_img)
    if(type == "test"):
        augmented_img_test.append(nifti_img)
    elif(type == "test_gt"):
        augmented_gt_test.append(nifti_img)
    else:
        print("Wrong type")
    
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
2. Concatenation of lists
3. Prepare the data for the network(reshape, normalize)
4. Split the data to train, test and validation
4. Data augmentation 
5. Save the augmented data to a list
6. Train the network with the training datas(contains data augmentated images) in nifti format


    https://colab.research.google.com/drive/112NTL8uJXzcMw4PQbUvMQN-WHlVwQS3i
"""

if __name__ == "__main__":
    
    #Load the nifti files into lists(nifti format)
    load_nifti_files()

    #Max intenzity,Resize and Normalize all images and ground truth images
    for i in range(len(patient_vols_train)):
        patient_vols_train[i], patient_vols_train_gt[i] = max_int_resize_and_normalize_(patient_vols_train[i], patient_vols_train_gt[i])
    for i in range(len(patient_vols_test)):
        patient_vols_test[i], patient_vols_test_gt[i] = max_int_resize_and_normalize_(patient_vols_test[i], patient_vols_test_gt[i])
    for i in range(len(patient_vols_train_4d)):
        
        patient_vols_train_4d[i] = max_int_resize_and_normalize(patient_vols_train_4d[i])
    for i in range(len(patient_vols_test_4d)):
        patient_vols_test_4d[i] = max_int_resize_and_normalize(patient_vols_test_4d[i])
    
    # Data augmentation
    # rotate and mirror the images, then add random noise to random images
    # every train image has 3 rotated,2 mirrored and 1 noisy version
    # make the same with the ground truth images
   
    # 200*7 = 1400 images for train
    for i in range(len(patient_vols_train)):
        augmented_img_train.append(rotation_z(patient_vols_train[i], 90))
        augmented_img_train.append(rotation_z(patient_vols_train[i], 180))
        augmented_img_train.append(rotation_z(patient_vols_train[i], 270))
        augmented_img_train.append(mirror_y(patient_vols_train[i]))
        augmented_img_train.append(mirror_x(patient_vols_train[i]))
        augmented_img_train.append(smooth(patient_vols_train[i], 2))
        augmented_gt_train.append(rotation_z(patient_vols_train_gt[i], 90))
        augmented_gt_train.append(rotation_z(patient_vols_train_gt[i], 180))
        augmented_gt_train.append(rotation_z(patient_vols_train_gt[i], 270))
        augmented_gt_train.append(mirror_y(patient_vols_train_gt[i]))
        augmented_gt_train.append(mirror_x(patient_vols_train_gt[i]))
        augmented_gt_train.append(smooth(patient_vols_train[i], 2))
    
    # for the test images we make the same
    # 100*7 = 700 images for test 
    for i in range(len(patient_vols_test)):   
        augmented_img_test.append(rotation_z(patient_vols_test[i], 90))
        augmented_img_test.append(rotation_z(patient_vols_test[i], 180))
        augmented_img_test.append(rotation_z(patient_vols_test[i], 270))
        augmented_img_test.append(mirror_y(patient_vols_test[i]))
        augmented_img_test.append(mirror_x(patient_vols_test[i]))
        augmented_img_test.append(smooth(patient_vols_test[i],2))
        augmented_gt_test.append(rotation_z(patient_vols_test_gt[i], 90))
        augmented_gt_test.append(rotation_z(patient_vols_test_gt[i], 180))
        augmented_gt_test.append(rotation_z(patient_vols_test_gt[i], 270))
        augmented_gt_test.append(mirror_y(patient_vols_test_gt[i]))
        augmented_gt_test.append(mirror_x(patient_vols_test_gt[i]))
        augmented_gt_test.append(smooth(patient_vols_test_gt[i],2)) 
      
    # add random noise to random augmented images
    # make random 70 train images and 35 test images noisy
    # between 200 and 1200, because the first 200 images already have noisy sample(the last 200)
    
    np.random.seed(42)  # Beállítjuk a seed-et
    random_numbers = np.random.randint(200, 1200, size=70)
    
    for i in range(70):
        augmented_img_train.append(smooth(augmented_img_train[i],2))
        augmented_gt_train.append(smooth(augmented_gt_train[i],2))
     
    random_numbers = np.random.randint(100, 600, size=35)    
    
    for i in range(35):
        augmented_img_test.append(smooth(augmented_img_test[i],2))
        augmented_gt_test.append(smooth(augmented_gt_test[i],2))       
        
# Concatenate the lists
    X_train = patient_vols_train + augmented_img_train
    y_train = patient_vols_train_gt + augmented_gt_train
    
    X_test = patient_vols_test + augmented_img_test
    y_test = patient_vols_test_gt + augmented_gt_test

    # convert list of nifti images to numpy array
    X_train = np.array([nifti2numpy(img) for img in X_train])
    y_train = np.array([nifti2numpy(img) for img in y_train])
    X_test = np.array([nifti2numpy(img) for img in X_test])
    y_test = np.array([nifti2numpy(img) for img in y_test])


 

    
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)  
    #train_loader = create_dataloader(X_train, y_train, batch_size=32)
    #test_loader = test_dataloader(X_test,batch_size)
    print("ok")
    
"""   
# define model
    model = sm.Unet(BACKBONE, encoder_weights='imagenet')
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    # fit model
    # # if you use data generator use model.fit_generator(...) instead of model.fit(...)
    # # more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=16,
        epochs=100,
        validation_data=(X_test, y_test),
    )
    """