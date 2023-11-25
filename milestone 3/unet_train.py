import gc
import csv
import keras
import argparse
import numpy as np
import segmentation_models as sm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def plot_history(history, model_name):
    gc.collect()
    loss, val_loss, acc, val_acc = history.history['loss'], history.history['val_loss'], history.history['iou_score'], history.history['val_iou_score']
    fscore, val_fscore = history.history['f1-score'], history.history['val_f1-score']
    epochs = range(1, len(loss) + 1)

    # Save training history to a CSV file
    with open(f'{model_name}_training_history.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss', 'Val Loss', 'Accuracy', 'Val Accuracy', 'F-Score', 'Val F-Score'])
        writer.writerows(zip(epochs, loss, val_loss, acc, val_acc, fscore, val_fscore))

    print(f"Saved {model_name}_training_history.csv to folder.")

    # Plot and save training metrics
    _, axes = plt.subplots(1, 3, figsize=(18, 4))
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

    plt.savefig(f'/content/{model_name}_training_metrics_plot.png')
    plt.show()

    print(f"Saved {model_name}_training_metrics_plot.png to folder.")

def data_generator(x_data, y_data, batch_size):
    # Function to generate batches of training data
    num_samples = x_data.shape[0]
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            x_batch = x_data[batch_indices]
            y_batch = y_data[batch_indices]
            yield x_batch, y_batch

def train_model(batch_size, learning_rate, epochs):
    # Function to train the segmentation model
    gc.collect()

    print("Starting training with batch size: ", batch_size, ", learning rate: ", learning_rate, ", and epochs: ", epochs)
    model_name = "unet"
    n_classes = 4

    x_train = np.load('x_train.npz')['arr_0']
    y_train = np.load('y_train.npz')['arr_0']
    x_test = np.load('x_test.npz')['arr_0']
    y_test = np.load('y_test.npz')['arr_0']
    x_train_aug = np.load('x_train_aug.npz')['arr_0']
    y_train_aug = np.load('y_train_aug.npz')['arr_0']
    x_test_aug = np.load('x_test_aug.npz')['arr_0']
    y_test_aug = np.load('y_test_aug.npz')['arr_0']
    
    # x_train = np.clip(np.concatenate((x_train, x_train_aug), axis=0), 0.0, 1.0)
    x_train = np.concatenate((x_train, x_train_aug), axis=0)
    y_train = np.concatenate((y_train, y_train_aug), axis=0)
    # x_test = np.clip(np.concatenate((x_test, x_test_aug), axis=0), 0.0, 1.0)
    x_test = np.concatenate((x_test, x_test_aug), axis=0)
    y_test = np.concatenate((y_test, y_test_aug), axis=0)

    # x_train[x_train < 0] = 0
    # x_train[x_train > 1] = 1
    # x_test[x_test < 0] = 0
    # x_test[x_test > 1] = 1
    # y_train = np.round(3*y_train)
    # y_test = np.round(3*y_test)

    print("x_train: \t", x_train.shape, np.min(x_train), np.max(x_train), "\t", x_train[0][128][120:121])
    print("y_train: \t", y_train.shape, np.min(y_train), np.max(y_train), "\t", y_train[0][128][120:121], "\tlabels: ", np.unique(y_train))
    print("x_test: \t", x_test.shape, np.min(x_test), np.max(x_test), "\t", x_test[0][128][120:121])
    print("y_test: \t", y_test.shape, np.min(y_test), np.max(y_test), "\t", y_test[0][128][120:121], "\tlabels: ", np.unique(y_test))

    preprocess_input = sm.get_preprocessing('resnet34')

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.16, random_state=42)

    y_train_cat = to_categorical(y_train, num_classes=n_classes).reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
    y_val_cat = to_categorical(y_val, num_classes=n_classes).reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))
    y_test_cat = to_categorical(y_test, num_classes=n_classes).reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

    print(x_train.shape, x_val.shape, x_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)
    print(y_train_cat.shape, y_val_cat.shape, y_test_cat.shape)
    print("y_train_cat: \t", y_train_cat.shape, np.min(y_train_cat), np.max(y_train_cat), y_train_cat[0][128][120:121], "\tlabels: ", np.unique(y_train_cat))
    print("y_val_cat: \t", y_val_cat.shape, np.min(y_val_cat), np.max(y_val_cat), y_val_cat[0][128][120:121], "\tlabels: ", np.unique(y_val_cat))
    print("y_test_cat: \t", y_test_cat.shape, np.min(y_test_cat), np.max(y_test_cat), y_test_cat[0][128][120:121], "\tlabels: ", np.unique(y_test_cat))
    
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)

    # Create data generators for training and validation
    train_generator = data_generator(x_train, y_train_cat, batch_size=batch_size)
    val_generator = data_generator(x_val, y_val_cat, batch_size=batch_size)

    # Define loss functions
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # Build the segmentation model
    model = sm.Unet('resnet34', encoder_weights='imagenet', activation='softmax', classes=n_classes)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=total_loss, metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)])
    gc.collect()

    # Train the model
    history = model.fit(train_generator,
                        steps_per_epoch=len(x_train) // batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_generator,
                        validation_steps=len(x_val) // batch_size)

    model.save(f'/content/{model_name}_self-trained.h5')
    print(f"Saved {model_name}_self-trained.h5 to folder.")
    return history, model_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the unet model with custom parameters.")
    
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")

    args = parser.parse_args()

    history, model_name = train_model(args.batch_size, args.learning_rate, args.epochs)
    plot_history(history, model_name)
    gc.collect()
