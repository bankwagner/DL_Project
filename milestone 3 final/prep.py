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


def numpy2nifti(numpy_img):
    """
    Convert a numpy array to a Nifti1Image object.

    Args:
        numpy_img (np.ndarray): The input numpy array.

    Returns:
        nib.Nifti1Image: The converted Nifti1Image object.
    """
    return nib.Nifti1Image(numpy_img, original_affine)


def nifti2numpy(nifti_img):
    """
    Convert a Nifti1Image object to a numpy array.

    Args:
        nifti_img (nib.Nifti1Image): The input Nifti1Image object.

    Returns:
        np.ndarray: The converted numpy array.
    """
    return nifti_img.get_fdata()


def max_intenzity(image):
    """
    Normalize the intensity values of an image to the range [0, 255].

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The normalized image.
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
    return numpy2nifti(norm_img)


def max_int_resize_and_normalize_no_label(numpy_img, label):
    """
    Resize, normalize, and convert the input image and label to Nifti1Image objects.

    Args:
        numpy_img (nib.Nifti1Image): The input image.
        label (nib.Nifti1Image): The input label.

    Returns:
        tuple: A tuple containing the resized and normalized image (Nifti1Image) and label (Nifti1Image).
    """
    resized_norm_data = skt.resize(max_intenzity(nifti2numpy(numpy_img)), (256,256,10), order=1, preserve_range=False,anti_aliasing=True)
    resized_norm_label = skt.resize(nifti2numpy(label),(256,256,10), order=1, preserve_range=False,anti_aliasing=True)
    resized_norm_data = normalize(resized_norm_data)
    resized_norm_label = numpy2nifti(resized_norm_label)

    return resized_norm_data, resized_norm_label


def load_nifti_files():
    """
    Load the Nifti files for training and testing.

    Returns:
        None
    """
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


def rotation_z_with_affine(image, rotation_degree=90):
    """
    Rotate an image around the z-axis with a given rotation degree using an affine transformation.

    Args:
        image (nib.Nifti1Image): The input image.
        rotation_degree (float): The rotation degree in degrees (default: 90).

    Returns:
        nib.Nifti1Image: The rotated image.
    """
    rotation_radians = np.radians(rotation_degree)
    rotation_affine = np.array(
        [
            [np.cos(rotation_radians), -np.sin(rotation_radians), 0, 0],
            [np.sin(rotation_radians), np.cos(rotation_radians), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    affine_so_far = image.affine.dot(rotation_affine)
    return nib.Nifti1Image(image, affine=affine_so_far)


def rotation_z(nifti_img, rotation_degree):
    """
    Rotate an image around the z-axis with a given rotation degree.

    Args:
        nifti_img (nib.Nifti1Image): The input image.
        rotation_degree (float): The rotation degree in degrees.

    Returns:
        nib.Nifti1Image: The rotated image.
    """
    img = nifti_img.get_fdata()
    rotated_data = rotate(img, rotation_degree, (0, 1), reshape=False)
    return nib.Nifti1Image(rotated_data, original_affine)


def smooth(nifti_img, fwhm):
    """
    Apply Gaussian smoothing to an image.

    Args:
        nifti_img (nib.Nifti1Image): The input image.
        fwhm (float): The full width at half maximum (FWHM) value for the Gaussian kernel.

    Returns:
        nib.Nifti1Image: The smoothed image.
    """
    return nili.smooth_img(nifti_img, fwhm=fwhm)


def mirror_y_with_affine(numpy_img):
    """
    Mirror an image along the y-axis using an affine transformation.

    Args:
        numpy_img (np.ndarray): The input image.

    Returns:
        nib.Nifti1Image: The mirrored image.
    """
    return nib.Nifti1Image(numpy_img, np.eye(4))


def mirror_y(nifti_img):
    """
    Mirror an image along the y-axis.

    Args:
        nifti_img (nib.Nifti1Image): The input image.

    Returns:
        nib.Nifti1Image: The mirrored image.
    """
    mirrored_data = np.flip(nifti2numpy(nifti_img), axis=0)
    return nib.Nifti1Image(mirrored_data, original_affine)


def mirror_x_with_affine(numpy_img):
    """
    Mirror an image along the x-axis using an affine transformation.

    Args:
        numpy_img (np.ndarray): The input image.

    Returns:
        nib.Nifti1Image: The mirrored image.
    """
    matrix = np.diag([-1, 1, 1, 1])
    return nib.Nifti1Image(numpy_img, matrix)


def mirror_x(nifti_img):
    """
    Mirror an image along the x-axis.

    Args:
        nifti_img (nib.Nifti1Image): The input image.

    Returns:
        nib.Nifti1Image: The mirrored image.
    """
    mirrored_data = np.flip(nifti2numpy(nifti_img), axis=1)
    return nib.Nifti1Image(mirrored_data, original_affine)


if __name__ == "__main__":
    print(" - - - preprocessing started - - - ")

    # Set environment variable for the SM_FRAMEWORK
    os.environ["SM_FRAMEWORK"] = "tf.keras"

    # Initialize lists to store training and testing data
    patient_vols_train = []
    patient_vols_train_gt = []
    patient_vols_test = []
    patient_vols_test_gt = []

    original_affine = np.diag([-1, -1, 1, 1])

    # Load Nifti files
    load_nifti_files()

    for i in range(len(patient_vols_train)):
        patient_vols_train[i], patient_vols_train_gt[i] = max_int_resize_and_normalize_no_label(
            patient_vols_train[i], patient_vols_train_gt[i]
        )
    for i in range(len(patient_vols_test)):
        patient_vols_test[i], patient_vols_test_gt[i] = max_int_resize_and_normalize_no_label(
            patient_vols_test[i], patient_vols_test_gt[i]
        )

    # Initialize lists to store augmented images for training and testing
    augmented_img_train = []
    augmented_img_test = []
    augmented_gt_train = []
    augmented_gt_test = []

    np.random.seed(42)

    # Augment training data
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

    # Augment testing data
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
    
    # Processing the data
    x_train = []
    for img in patient_vols_train:
        for i in range(1, img.shape[-1], 2):
            new_data = np.stack((img.get_fdata()[:, :, i],) * 3, axis=-1).astype("float32")
            x_train.append(new_data)

    x_train_aug = []
    for img in augmented_img_train:
        for i in range(1, img.shape[-1], 2):
            new_data = np.stack((img.get_fdata()[:, :, i],) * 3, axis=-1).astype("float32")
            x_train_aug.append(new_data)

    y_train = []
    for img in patient_vols_train_gt:
        for i in range(1, img.shape[-1], 2):
            new_data = img.get_fdata()[:, :, i].astype("float32")
            y_train.append(new_data)

    y_train_aug = []
    for img in augmented_gt_train:
        for i in range(1, img.shape[-1], 2):
            new_data = img.get_fdata()[:, :, i].astype("float32")
            y_train_aug.append(new_data)

    x_test = []
    for img in patient_vols_test:
        for i in range(1, img.shape[-1], 2):
            new_data = np.stack((img.get_fdata()[:, :, i],) * 3, axis=-1).astype("float32")
            x_test.append(new_data)

    x_test_aug = []
    for img in augmented_img_test:
        for i in range(1, img.shape[-1], 2):
            new_data = np.stack((img.get_fdata()[:, :, i],) * 3, axis=-1).astype("float32")
            x_test_aug.append(new_data)

    y_test = []
    for img in patient_vols_test_gt:
        for i in range(1, img.shape[-1], 2):
            new_data = img.get_fdata()[:, :, i].astype("float32")
            y_test.append(new_data)

    y_test_aug = []
    for img in augmented_gt_test:
        for i in range(1, img.shape[-1], 2):
            new_data = img.get_fdata()[:, :, i].astype("float32")
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

    # Create numpy arrays and clip pixel values as a safety measure
    x_train = np.array(x_train)
    x_train_aug = np.array(x_train_aug)
    y_train = np.array(y_train)
    y_train_aug = np.array(y_train_aug)
    x_test = np.array(x_test)
    x_test_aug = np.array(x_test_aug)
    y_test = np.array(y_test)
    y_test_aug = np.array(y_test_aug)

    y_train = np.expand_dims(y_train, axis=3)
    y_train = np.round(y_train)
    y_train_aug = np.expand_dims(y_train_aug, axis=3)
    y_train_aug = np.round(y_train_aug)
    y_test = np.expand_dims(y_test, axis=3)
    y_test = np.round(y_test)
    y_test_aug = np.expand_dims(y_test_aug, axis=3)
    y_test_aug = np.round(y_test_aug)

    x_train = np.clip(x_train, 0, 1)
    x_train_aug = np.clip(x_train_aug, 0, 1)
    y_train = np.clip(y_train, 0, 3)
    y_train_aug = np.clip(y_train_aug, 0, 3)
    x_test = np.clip(x_test, 0, 1)
    x_test_aug = np.clip(x_test_aug, 0, 1)
    y_test = np.clip(y_test, 0, 3)
    y_test_aug = np.clip(y_test_aug, 0, 3)
    
    # Print information about the preprocessed data
    print("x_train:\t shape: ", x_train.shape, "\tmin: ", np.min(x_train), "max: ", np.max(x_train))
    print("x_train_aug:\t shape: ", x_train_aug.shape, "\tmin: ", np.min(x_train_aug), "max: ", np.max(x_train_aug))
    print("y_train:\t shape: ", y_train.shape, "\tmin: ", np.min(y_train), "max: ", np.max(y_train), "unique: ", np.unique(y_train))
    print("y_train_aug:\t shape: ", y_train_aug.shape, "\tmin: ", np.min(y_train_aug), "max: ", np.max(y_train_aug), "unique: ", np.unique(y_train_aug))
    print("x_test:\t\t shape: ", x_test.shape, "\tmin: ", np.min(x_test), "max: ", np.max(x_test))
    print("x_test_aug:\t shape: ", x_test_aug.shape, "\tmin: ", np.min(x_test_aug), "max: ", np.max(x_test_aug))
    print("y_test:\t\t shape: ", y_test.shape, "\tmin: ", np.min(y_test), "max: ", np.max(y_test), "unique: ", np.unique(y_test))
    print("y_test_aug:\t shape: ", y_test_aug.shape, "\tmin: ", np.min(y_test_aug), "max: ", np.max(y_test_aug), "unique: ", np.unique(y_test_aug))

    # Save preprocessed data to compressed files
    np.savez_compressed("x_train.npz", x_train)
    np.savez_compressed("x_train_aug.npz", x_train_aug)
    np.savez_compressed("y_train.npz", y_train)
    np.savez_compressed("y_train_aug.npz", y_train_aug)
    np.savez_compressed("x_test.npz", x_test)
    np.savez_compressed("x_test_aug.npz", x_test_aug)
    np.savez_compressed("y_test.npz", y_test)
    np.savez_compressed("y_test_aug.npz", y_test_aug)

    # Delete variables to free up memory
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
