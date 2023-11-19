import gdown
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.metrics import MeanIoU, Precision, Recall
import segmentation_models as sm
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import csv

if __name__ == "__main__":
    model_name = "unet"
    n_classes = 4
    file_id = '1qAUJtiPZfT3jm4V4qg-VRVucpZva-Qj_'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', '/content/ACDC.zip', quiet=False)

    x_train = np.load('x_train.npz')['arr_0']
    y_train = np.load('y_train.npz')['arr_0']
    x_test = np.load('x_test.npz')['arr_0']
    y_test = np.load('y_test.npz')['arr_0']

    x_train_aug = np.load('x_train_aug.npz')['arr_0']
    y_train_aug = np.load('y_train_aug.npz')['arr_0']
    x_test_aug = np.load('x_test_aug.npz')['arr_0']
    y_test_aug = np.load('y_test_aug.npz')['arr_0']

    x_train = np.clip(np.concatenate((x_train, x_train_aug), axis=0), 0.0, 1.0)
    y_train = np.concatenate((y_train, y_train_aug), axis=0)
    x_test = np.clip(np.concatenate((x_test, x_test_aug), axis=0), 0.0, 1.0)
    y_test = np.concatenate((y_test, y_test_aug), axis=0)

    y_train = np.round(3*y_train)
    y_test = np.round(3*y_test)

    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.16, random_state=42)

    y_train_cat = to_categorical(y_train, num_classes=n_classes).reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
    y_val_cat = to_categorical(y_val, num_classes=n_classes).reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))
    y_test_cat = to_categorical(y_test, num_classes=n_classes).reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))


    # generate the test results and compile them into a .csv
    model = load_model(f'/content/{model_name}_model.h5', compile=False) # comment out this line, if you didnt save the model in the step before

    n_classes = 4
    IOU_keras = MeanIoU(num_classes=n_classes)
    precision = Precision()
    recall = Recall()

    mean_iou = 0.0
    mean_f1 = 0.0

    for i in range(len(x_test)):
        test_img = x_test[i]
        test_img_gt = y_test[i]

        test_img_input = np.expand_dims(test_img, 0)
        test_img_pred = np.argmax(model.predict(test_img_input, verbose=0), axis=3)[0, :, :]

        # Calculate IUO (Intersection over Union) and update state
        IOU_keras.update_state(test_img_gt[:, :, 0], test_img_pred)
        mean_iou += IOU_keras.result().numpy()

        # Update state for Precision and Recall
        precision.update_state(test_img_gt[:, :, 0], test_img_pred)
        recall.update_state(test_img_gt[:, :, 0], test_img_pred)

        precision_result = precision.result().numpy()
        recall_result = recall.result().numpy()

        # Calculate F1 score
        f1_score = 2 * (precision_result * recall_result) / (precision_result + recall_result + 1e-10)
        mean_f1 += f1_score

    mean_iou /= len(x_test)
    mean_f1 /= len(y_test)

    print("Mean IoU =", mean_iou)
    print("Mean F1 Score =", mean_f1)

    with open(f'{model_name}_test_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Mean IoU', mean_iou])
        writer.writerow(['Mean F1 Score', mean_f1])

        preprocess_input = sm.get_preprocessing(BACKBONE)
    model = load_model(f'/content/{model_name}_model.h5', compile=False) # comment out this line, if you didnt save the model in the step before

    img_num = random.randint(0, len(x_test) - 1)

    test_img = x_test[img_num]
    test_img_gt = y_test[img_num]

    test_img_input = np.expand_dims(preprocess_input(test_img), 0)
    test_img_pred = np.argmax(model.predict(test_img_input), axis=3)[0,:,:]

    plt.figure(figsize=(12, 8))

    # First row: Original image, Ground truth label, and Full prediction
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img, cmap='gray')

    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(test_img_gt)

    plt.subplot(233)
    plt.title('Prediction on Test Image')
    plt.imshow(test_img_pred)

    # Second row: Displaying the segments of the three classes separately
    class_segments = [np.where(test_img_pred == i + 1, i + 1, 0) for i in range(3)]
    titles = ['Class 1 Segment', 'Class 2 Segment', 'Class 3 Segment']

    for i in range(3):
        plt.subplot(2, 3, i + 4)
        plt.title(titles[i])
        plt.imshow(class_segments[i], cmap='gray')

    plt.savefig(f'/content/{model_name}_prediction_plot.png')
