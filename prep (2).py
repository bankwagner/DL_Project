import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
import dicom2nifti
import nibabel as nib
import nilearn as nil
import nilearn.image as nili
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.transform as skt
from scipy.ndimage import rotate

from sklearn import preprocessing


# convert numpy image to nifti format
def numpy2nifti(numpy_img):
    return nib.Nifti1Image(numpy_img, original_affine)

# convert nifti image to numpy format
def nifti2numpy(nifti_img):
    return nifti_img.get_fdata()

# set the intensity of the image
def max_intenzity(image):
    min_val = np.min(image)
    max_val = np.max(image)
    min_max_norm_img = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return min_max_norm_img

# normalize and return in nifti format
def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    norm_img = ((image - min_val) / (max_val - min_val))
    return numpy2nifti(norm_img)
    
# normalize and return numpy format
def normalize_test(image):
    min_val = np.min(image)
    max_val = np.max(image)
    norm_img = ((image - min_val) / (max_val - min_val))
    return norm_img

# resize the train and label images (256,256,10),
#normalize only the train
#return in nifti format
def max_int_resize_and_normalize_no_label(numpy_img, label):

    resized_norm_data = skt.resize(max_intenzity(nifti2numpy(numpy_img)), (256,256,10), order=1, preserve_range=False,anti_aliasing=True)
    resized_norm_label = skt.resize(nifti2numpy(label),(256,256,10), order=1, preserve_range=False,anti_aliasing=True)
    resized_norm_data = normalize(resized_norm_data)
    resized_norm_label = numpy2nifti(resized_norm_label)

    return resized_norm_data, resized_norm_label

def max_int_resize_and_normalize_zeros(numpy_img, label):
    shape=numpy_img.shape[-1]
    w = numpy_img.shape[0]
    h = numpy_img.shape[1]
    original_array = nifti2numpy(numpy_img)
    original_label = nifti2numpy(label)

    # Create a new array of zeros with the desired size (256, 256, 10)
    resized_array = np.zeros((256, 256, shape))
    resized_array_label = np.zeros((256, 256, shape))

    # Fill the corresponding region of the new array with the original data
    resized_array[:w, :h, :] = original_array
    resized_array_label[:w, :h, :] = original_label
    
    resized_norm_data = normalize(resized_array)
    resized_label = numpy2nifti(resized_array_label)

    return resized_norm_data, resized_label

# same function, but only an image without label
def max_int_resize_and_normalize_test(numpy_img):

    resized_norm_data = skt.resize(max_intenzity(numpy_img), (256,256,10), order=1, preserve_range=False,anti_aliasing=True)
    resized_norm_data = normalize_test(resized_norm_data)

    return resized_norm_data

# load the nifti files and put in blocks
def load_nifti_files():
    for i in range(1, 101):
        for frame_num in range(1, 17):
            if os.path.isfile(
                f"ACDC/database/training/patient{101-i:03}/patient{101-i:03}_frame{frame_num:02}.nii.gz"
            ):
                patient_vols_train.append(
                    nib.load(
                        f"ACDC/database/training/patient{101-i:03}/patient{101-i:03}_frame{frame_num:02}.nii.gz"
                    )
                )
            if os.path.isfile(
                f"ACDC/database/training/patient{101-i:03}/patient{101-i:03}_frame{frame_num:02}_gt.nii.gz"
            ):
                patient_vols_train_gt.append(
                    nib.load(
                        f"ACDC/database/training/patient{101-i:03}/patient{101-i:03}_frame{frame_num:02}_gt.nii.gz"
                    )
                )

    for i in range(101, 151):
        for frame_num in range(1, 17):
            if os.path.isfile(
                f"ACDC/database/testing/patient{i}/patient{i}_frame{frame_num:02}.nii.gz"
            ):
                patient_vols_test.append(
                    nib.load(
                        f"ACDC/database/testing/patient{i}/patient{i}_frame{frame_num:02}.nii.gz"
                    )
                )
            if os.path.isfile(
                f"ACDC/database/testing/patient{i}/patient{i}_frame{frame_num:02}_gt.nii.gz"
            ):
                patient_vols_test_gt.append(
                    nib.load(
                        f"ACDC/database/testing/patient{i}/patient{i}_frame{frame_num:02}_gt.nii.gz"
                    )
                )

    return

# rotate the image around z axis
def rotation_z(nifti_img, rotation_degree):
    img = nifti_img.get_fdata()
    rotated_data = rotate(img, rotation_degree, (0, 1), reshape=False)
    return nib.Nifti1Image(rotated_data, original_affine)

# add random noise to the image
def smooth(nifti_img, fwhm):
    return nili.smooth_img(nifti_img, fwhm=fwhm)

# mirroring around the y axis
def mirror_y(nifti_img):
    mirrored_data = np.flip(nifti2numpy(nifti_img), axis=0)
    return nib.Nifti1Image(mirrored_data, original_affine)
    
# mirroring around the x axis
def mirror_x(nifti_img):
    mirrored_data = np.flip(nifti2numpy(nifti_img), axis=1)
    return nib.Nifti1Image(mirrored_data, original_affine)

# transform the input image(jpg), return in numpy format
def test_transform(image):
    img = max_int_resize_and_normalize_test(image)
    for i in range(1,img.shape[-1],2):
        new_data = np.stack((img[:,:,i],)*3, axis=-1).astype('float32')
        
    # return type: numpy 
    # shape : (256,256,3)
    return new_data

if __name__ == "__main__":
    print(" - - - preprocessing started - - - ")

    os.environ["SM_FRAMEWORK"] = "tf.keras"

    patient_vols_train = []
    patient_vols_train_gt = []
    patient_vols_test = []
    patient_vols_test_gt = []

    original_affine = np.diag([-1, -1, 1, 1])

    load_nifti_files()

    for i in range(len(patient_vols_train)):
        patient_vols_train[i], patient_vols_train_gt[i] = max_int_resize_and_normalize_no_label(
            patient_vols_train[i], patient_vols_train_gt[i]
        )
    for i in range(len(patient_vols_test)):
        patient_vols_test[i], patient_vols_test_gt[i] = max_int_resize_and_normalize_no_label(
            patient_vols_test[i], patient_vols_test_gt[i]
        )

    augmented_img_train = []
    augmented_img_test = []
    augmented_gt_train = []
    augmented_gt_test = []

    np.random.seed(42)

    # data augmentation
    random_numbers = np.random.randint(0, 200, size=25)
    for j in range(0, 25):
        i = random_numbers[j]
        augmented_img_train.append(rotation_z(patient_vols_train[i], 90))
        augmented_img_train.append(rotation_z(patient_vols_train[i], 270))
        augmented_img_train.append(mirror_y(patient_vols_train[i]))
        augmented_img_train.append(mirror_x(patient_vols_train[i]))
        augmented_img_train.append(smooth(patient_vols_train[i], 2))
        augmented_gt_train.append(rotation_z(patient_vols_train_gt[i], 90))
        augmented_gt_train.append(rotation_z(patient_vols_train_gt[i], 270))
        augmented_gt_train.append(mirror_y(patient_vols_train_gt[i]))
        augmented_gt_train.append(mirror_x(patient_vols_train_gt[i]))
        augmented_gt_train.append(smooth(patient_vols_train_gt[i], 2))

    random_numbers = np.random.randint(0, 100, size=10)
    for j in range(0, 10):
        i = random_numbers[j]
        augmented_img_test.append(rotation_z(patient_vols_test[i], 90))
        augmented_img_test.append(rotation_z(patient_vols_test[i], 270))
        augmented_img_test.append(mirror_y(patient_vols_test[i]))
        augmented_img_test.append(mirror_x(patient_vols_test[i]))
        augmented_img_test.append(smooth(patient_vols_test[i], 2))
        augmented_gt_test.append(rotation_z(patient_vols_test_gt[i], 90))
        augmented_gt_test.append(rotation_z(patient_vols_test_gt[i], 270))
        augmented_gt_test.append(mirror_y(patient_vols_test_gt[i]))
        augmented_gt_test.append(mirror_x(patient_vols_test_gt[i]))
        augmented_gt_test.append(smooth(patient_vols_test_gt[i], 2))

    # convert shape from (256,256,10)-> (256,256,3)
    x_train = []
    for img in patient_vols_train:
        for i in range(1, img.shape[-1], 1):
            new_data = np.stack((img.get_fdata()[:, :, i],) * 3, axis=-1).astype("float32")
            x_train.append(new_data)

    x_train_aug = []
    for img in augmented_img_train:
        for i in range(1, img.shape[-1], 1):
            new_data = np.stack((img.get_fdata()[:, :, i],) * 3, axis=-1).astype("float32")
            x_train_aug.append(new_data)

    y_train = []
    for img in patient_vols_train_gt:
        for i in range(1, img.shape[-1], 1):
            new_data = np.stack((img.get_fdata()[:, :, i],) * 1, axis=-1).astype("float32")
            y_train.append(new_data)

    y_train_aug = []
    for img in augmented_gt_train:
        for i in range(1, img.shape[-1], 1):
            new_data = np.stack((img.get_fdata()[:, :, i],) * 1, axis=-1).astype("float32")
            y_train_aug.append(new_data)

    x_test = []
    for img in patient_vols_test:
        for i in range(1, img.shape[-1], 1):
            new_data = np.stack((img.get_fdata()[:, :, i],) * 3, axis=-1).astype("float32")
            x_test.append(new_data)

    x_test_aug = []
    for img in augmented_img_test:
        for i in range(1, img.shape[-1], 1):
            new_data = np.stack((img.get_fdata()[:, :, i],) * 3, axis=-1).astype("float32")
            x_test_aug.append(new_data)

    y_test = []
    for img in patient_vols_test_gt:
        for i in range(1, img.shape[-1], 1):
            new_data = np.stack((img.get_fdata()[:, :, i],) * 1, axis=-1).astype("float32")
            y_test.append(new_data)

    y_test_aug = []
    for img in augmented_gt_test:
        for i in range(1, img.shape[-1], 1):
            new_data = np.stack((img.get_fdata()[:, :, i],) * 1, axis=-1).astype("float32")
            y_test_aug.append(new_data)

    del patient_vols_train
    del patient_vols_train_gt
    del patient_vols_test
    del patient_vols_test_gt
    gc.collect()
    del augmented_img_train
    del augmented_img_test
    del augmented_gt_train
    del augmented_gt_test
    gc.collect()

    x_train = np.array(x_train)
    x_train_aug = np.array(x_train_aug)
    y_train = np.array(y_train)
    y_train_aug = np.array(y_train_aug)
    x_test = np.array(x_test)
    x_test_aug = np.array(x_test_aug)
    y_test = np.array(y_test)
    y_test_aug = np.array(y_test_aug)


    print("x_train shape:\t\t", x_train.shape)
    print("x_train_aug shape:\t", x_train_aug.shape)
    print("y_train shape:\t\t", y_train.shape)
    print("y_train_aug shape:\t", y_train_aug.shape)
    print("x_test shape:\t\t", x_test.shape)
    print("x_test_aug shape:\t", x_test_aug.shape)
    print("y_test shape:\t\t", y_test.shape)
    print("y_test_aug shape:\t", y_test_aug.shape)

    # save the numpy files for the training
    np.savez_compressed("x_train.npz", x_train)
    np.savez_compressed("x_train_aug.npz", x_train_aug)
    np.savez_compressed("y_train.npz", y_train)
    np.savez_compressed("y_train_aug.npz", y_train_aug)
    np.savez_compressed("x_test.npz", x_test)
    np.savez_compressed("x_test_aug.npz", x_test_aug)
    np.savez_compressed("y_test.npz", y_test)
    np.savez_compressed("y_test_aug.npz", y_test_aug)

    del x_train
    del x_train_aug
    del y_train
    del y_train_aug
    del x_test
    del x_test_aug
    del y_test
    del y_test_aug
    gc.collect()

    print(" - - - preprocessing finished - - - ")
