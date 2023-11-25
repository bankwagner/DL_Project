import gc
import csv
import random
import argparse
import numpy as np
import segmentation_models as sm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.metrics import MeanIoU, Precision, Recall
from keras.models import load_model
from keras.utils import to_categorical

def create_test_metrics(x_test, y_test, model, model_name, model_type):
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

    with open(f'{model_name}_{model_type}_test_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Mean IoU', mean_iou])
        writer.writerow(['Mean F1 Score', mean_f1])

    gc.collect()
    del IOU_keras
    del precision
    del recall

def create_prediction_plot(x_test, y_test, model, model_name, model_type):
    preprocess_input = sm.get_preprocessing('resnet34')

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

    plt.savefig(f'/content/{model_name}_{model_type}_prediction_plot.png')
    plt.show()

    gc.collect()
    del test_img_input
    del test_img_pred
    del class_segments

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run evaluation script.')
    parser.add_argument('--model', choices=['pre-trained', 'self-trained'], default='pre-trained',
                        help='Specify the model to use: "pre-trained" or "self-trained". Default is "pre-trained".')
    return parser.parse_args()

if __name__ == "__main__":
    gc.collect()
    
    n_classes = 4
    model_name = "unet"
    args = parse_arguments()
    model = load_model(f'/content/{model_name}_{args.model}.h5', compile=False)

    # x_train = np.load('x_train.npz')['arr_0']
    # y_train = np.load('y_train.npz')['arr_0']
    x_test = np.load('x_test.npz')['arr_0']
    y_test = np.load('y_test.npz')['arr_0']
    # x_train_aug = np.load('x_train_aug.npz')['arr_0']
    # y_train_aug = np.load('y_train_aug.npz')['arr_0']
    x_test_aug = np.load('x_test_aug.npz')['arr_0']
    y_test_aug = np.load('y_test_aug.npz')['arr_0']
    
    # x_train = np.clip(np.concatenate((x_train, x_train_aug), axis=0), 0.0, 1.0)
    # x_train = np.concatenate((x_train, x_train_aug), axis=0)
    # y_train = np.concatenate((y_train, y_train_aug), axis=0)
    # x_test = np.clip(np.concatenate((x_test, x_test_aug), axis=0), 0.0, 1.0)
    x_test = np.concatenate((x_test, x_test_aug), axis=0)
    y_test = np.concatenate((y_test, y_test_aug), axis=0)

    # x_train[x_train < 0] = 0
    # x_train[x_train > 1] = 1
    # x_test[x_test < 0] = 0
    # x_test[x_test > 1] = 1
    # y_train = np.round(3*y_train)
    # y_test = np.round(3*y_test)

    # print("x_train: \t", x_train.shape, np.min(x_train), np.max(x_train), "\t", x_train[0][128][120:121])
    # print("y_train: \t", y_train.shape, np.min(y_train), np.max(y_train), "\t", y_train[0][128][120:121], "\tlabels: ", np.unique(y_train))
    print("x_test: \t", x_test.shape, np.min(x_test), np.max(x_test), "\t", x_test[0][128][120:121])
    print("y_test: \t", y_test.shape, np.min(y_test), np.max(y_test), "\t", y_test[0][128][120:121], "\tlabels: ", np.unique(y_test))


    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.16, random_state=42)

    # y_train_cat = to_categorical(y_train, num_classes=n_classes).reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
    # y_val_cat = to_categorical(y_val, num_classes=n_classes).reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))
    # y_test_cat = to_categorical(y_test, num_classes=n_classes).reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

    # print(x_train.shape, x_val.shape, x_test.shape)
    # print(y_train.shape, y_val.shape, y_test.shape)
    # print(y_train_cat.shape, y_val_cat.shape, y_test_cat.shape)
    # print("y_train_cat: \t", y_train_cat.shape, np.min(y_train_cat), np.max(y_train_cat), y_train_cat[0][128][120:121], "\tlabels: ", np.unique(y_train_cat))
    # print("y_val_cat: \t", y_val_cat.shape, np.min(y_val_cat), np.max(y_val_cat), y_val_cat[0][128][120:121], "\tlabels: ", np.unique(y_val_cat))
    # print("y_test_cat: \t", y_test_cat.shape, np.min(y_test_cat), np.max(y_test_cat), y_test_cat[0][128][120:121], "\tlabels: ", np.unique(y_test_cat))
    
    # preprocess_input = sm.get_preprocessing('resnet34')
    # x_train = preprocess_input(x_train)
    # x_val = preprocess_input(x_val)

    create_prediction_plot(x_test, y_test, model, model_name, args.model)
    
    create_test_metrics(x_test, y_test, model, model_name, args.model)
