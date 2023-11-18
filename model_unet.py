import segmentation_models as sm
# esetleg valahogy ki lehetne menteni, és ide feltölteni futtatás előtt: /usr/local/lib/python3.10/dist-packages/efficientnet/__init__.py
import tensorflow as tf
import cv2
import os
import csv
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras
from matplotlib import pyplot as plt
import random
from keras.models import load_model
from keras.metrics import MeanIoU, Precision, Recall

def run_tests(dataset_folder, model):

  image_files = sorted(os.listdir(f'{dataset_folder}/test'))
  mask_files = sorted(os.listdir(f'{dataset_folder}/testannot'))

  test_images = []
  test_masks = []

  for img_path, mask_path in zip(image_files, mask_files):
      img = cv2.imread(f'{dataset_folder}/test/{img_path}', 1)
      img = img.astype(float) / 255.0
      test_images.append(img)

      mask = cv2.imread(f'{dataset_folder}/testannot/{mask_path}', 0)
      mask = mask.astype(float)
      test_masks.append(mask)

  x_test = np.array(test_images)
  y_test = np.expand_dims(np.array(test_masks), axis=3)

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

      IOU_keras.update_state(test_img_gt[:, :, 0], test_img_pred)
      mean_iou += IOU_keras.result().numpy()

      precision.update_state(test_img_gt[:, :, 0], test_img_pred)
      recall.update_state(test_img_gt[:, :, 0], test_img_pred)

      precision_result = precision.result().numpy()
      recall_result = recall.result().numpy()

      f1_score = 2 * (precision_result * recall_result) / (precision_result + recall_result + 1e-10)
      mean_f1 += f1_score

  mean_iou /= len(x_test)
  mean_f1 /= len(x_test)

  print("Mean IoU =", mean_iou)
  print("Mean F1 Score =", mean_f1)

  with open('unet_test_results.csv', mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(['Metric', 'Value'])
      writer.writerow(['Mean IoU', mean_iou])
      writer.writerow(['Mean F1 Score', mean_f1])


def sanity_check_postrun(x_val, y_val, preprocess_input, model):

    img_num = random.randint(0, len(x_val) - 1)

    test_img = x_val[img_num]
    test_img_gt = y_val[img_num]

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

    plt.savefig('/content/base_model_prediction_plot.png')
    # plt.show()


def create_loss_pngs(history):
  loss, val_loss, acc, val_acc = history.history['loss'], history.history['val_loss'], history.history['iou_score'], history.history['val_iou_score']
  fscore, val_fscore = history.history['f1-score'], history.history['val_f1-score']
  epochs = range(1, len(loss) + 1)

  # Write the training history to a CSV file
  with open('training_history_unet.csv', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(['Epoch', 'Loss', 'Val Loss', 'Accuracy', 'Val Accuracy', 'F-Score', 'Val F-Score'])
      writer.writerows(zip(epochs, loss, val_loss, acc, val_acc, fscore, val_fscore))

  # Plot the training history
  fig, axes = plt.subplots(1, 3, figsize=(18, 4))

  axes[0].plot(epochs, loss, 'y', label='Training loss')
  axes[0].plot(epochs, val_loss, 'r', label='Validation loss')
  axes[1].plot(epochs, acc, 'y', label='Training IOU')
  axes[1].plot(epochs, val_acc, 'r', label='Validation IOU')
  axes[2].plot(epochs, fscore, 'y', label='Training F-Score')
  axes[2].plot(epochs, val_fscore, 'r', label='Validation F-Score')

  [ax.set_xlabel('Epochs') for ax in axes] and [ax.legend() for ax in axes]

  axes[0].set_title('Training and validation loss')
  axes[0].set_ylabel('Loss')
  axes[1].set_title('Training and validation IOU')
  axes[1].set_ylabel('IOU')
  axes[2].set_title('Training and validation F-Score')
  axes[2].set_ylabel('F-Score')

  plt.savefig('/content/unet_metrics_plot.png')
  plt.show()


def setup_files(SIZE_X, SIZE_Y, dataset_folder):
  #TODO: resize?
  x_train = np.load(f'{dataset_folder}/x_train.npz').tolist()
  x_train.append(np.load(f'{dataset_folder}/x_train_aug.npz').tolist())
  print("x_train: \t", x_train.shape)
  print("x_train max: \t", x_train.max())

  y_train = np.load(f'{dataset_folder}/y_train.npz').tolist()
  y_train.append(np.load(f'{dataset_folder}/y_train_aug.npz').tolist())
  y_train = np.expand_dims(y_train, axis=3)
  print("y_train: \t", y_train.shape)
  print("y_train unique: \t", np.unique(y_train))

  x_test = np.load(f'{dataset_folder}/x_test.npz').tolist()
  x_test.append(np.load(f'{dataset_folder}/x_test_aug.npz').tolist())
  print("x_test: \t", x_test.shape)

  y_test = np.load(f'{dataset_folder}/y_test.npz').tolist()
  y_test.append(np.load(f'{dataset_folder}/y_test_aug.npz').tolist())
  y_test = np.expand_dims(y_test, axis=3)
  print("y_test: \t", y_test.shape)

  return x_train, x_test, y_train, y_test


def sanity_check_prerun(X, Y):
  i = random.randint(0, 600)
  print("sanity check random number generated: ", i)
  plt.figure(figsize=(12,4))
  plt.subplot(121)
  plt.imshow(X[i])
  plt.subplot(122)
  plt.imshow(Y[i])


if __name__ == '__main__':
  # __init__.py -> /usr/local/lib/python3.10/dist-packages/efficientnet/__init__.py
  SIZE_X = 256
  SIZE_Y = 256
  dataset_folder = f'/content/' #f'/content/ACDC/database' # a helyes dockeres mappa kell ide, ahova a prep.py generál
  BATCH_SIZE = 8
  EPOCHS = 30
  LR = 0.0001
  n_classes = 4 # ez igazából fix
  preprocess_input = sm.get_preprocessing('resnet34')

  X, x_test, Y, y_test = setup_files(SIZE_X, SIZE_Y, dataset_folder)
  x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

  # sanity_check_prerun(x_train, y_train)
  # sanity_check_prerun(x_val, y_val)
  # sanity_check_prerun(x_test, y_test)

  x_train = preprocess_input(x_train)
  x_val = preprocess_input(x_val)
  x_test = preprocess_input(x_test)

  y_train_cat = to_categorical(y_train, num_classes=n_classes).reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
  y_val_cat = to_categorical(y_val, num_classes=n_classes).reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))
  y_test_cat = to_categorical(y_test, num_classes=n_classes).reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

  print("The inputs are:")
  print("\tx_train: \t", x_train.shape)
  print("\tx_val: \t\t", x_val.shape)
  print("\tx_val: \t\t", x_test.shape)
  print("Before converting to categorical data:")
  print("\ty_train: \t", y_train.shape)
  print("\ty_val: \t\t", y_val.shape)
  print("\ty_val: \t\t", y_test.shape)
  print("After converting to categorical data:")
  print("\ty_train_cat: \t", y_train_cat.shape)
  print("\ty_val_cat: \t", y_val_cat.shape)
  print("\ty_val_cat: \t", y_test_cat.shape)
  print("Class values in the dataset: ", np.unique(y_train))
  # print("Class values in the dataset: ", np.unique(y_train_cat))
  # print("Class values in the dataset: ", np.unique(y_val_cat))

  optim = keras.optimizers.Adam(LR)
  metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
  dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25]))
  focal_loss = sm.losses.CategoricalFocalLoss()
  total_loss = dice_loss + (1 * focal_loss)
  loss = total_loss

  model = sm.Unet('resnet34', encoder_weights='imagenet', activation='softmax', input_shape=(SIZE_X, SIZE_Y, 3), classes=n_classes)
  model.compile(optimizer=optim, loss=loss, metrics=metrics)
  print(model.summary())
    
  history = model.fit(
     x_train,
     y_train_cat,
     batch_size=BATCH_SIZE,
     epochs=EPOCHS,
     verbose=1,
     validation_data=(x_val, y_val_cat))

  create_loss_pngs(history)

  model.save('/base_model_unet.h5')

  model = load_model('/base_model_unet.h5', compile=False)

  sanity_check_postrun(x_val, y_val, preprocess_input, model)

  run_tests(dataset_folder, model)
