from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.metrics import MeanIoU
import segmentation_models as sm
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import random
import os
random.seed(42)

def random_gridsearch(predictions, y_test, n_classes):
    print(f"\nPerforming {4 ** len(predictions)} rounds of random gridsearch to find optimal weights...")
    df_results = []
    for i in range(4 ** len(predictions)):
        weights = [round(random.uniform(0, 1), 2) for _ in range(len(predictions))]
        IOU_weighted = MeanIoU(num_classes=n_classes)
        weighted_predictions = np.tensordot(predictions, weights, axes=((0), (0)))
        weighted_ensemble_prediction = np.argmax(weighted_predictions, axis=3)
        IOU_weighted.update_state(y_test[:, :, :, 0], weighted_ensemble_prediction)
        print(f"Round {i + 1} predicting for weights: {weights} \t -->  IOU = {IOU_weighted.result().numpy()}")
        df_results.append({'wt' + str(j + 1): weight for j, weight in enumerate(weights)} | {'IOU': IOU_weighted.result().numpy()})

    df = pd.DataFrame(df_results)
    max_iou_row = df.iloc[df['IOU'].idxmax()]
    print("Max IOU of", max_iou_row['IOU'], "obtained with weights:", max_iou_row.drop('IOU').to_dict())
    return max_iou_row


def eval_models(model_names, weights, predictions, y_test, n_classes):
    print("\nEvaluating models (with equal weights)...")
    num_models = len(predictions)
    weighted_predictions = np.tensordot(predictions, weights, axes=((0), (0)))
    weighted_ensemble_prediction = np.argmax(weighted_predictions, axis=3)

    IOU_models = [MeanIoU(num_classes=n_classes) for _ in range(num_models)]
    IOU_weighted = MeanIoU(num_classes=n_classes)

    for i in range(num_models):
        y_pred_argmax = np.argmax(predictions[i], axis=3)
        IOU_models[i].update_state(y_test[:, :, :, 0], y_pred_argmax)

    IOU_weighted.update_state(y_test[:, :, :, 0], weighted_ensemble_prediction)

    iou_score_models = [iou.result().numpy() for iou in IOU_models]
    iou_score_weighted = IOU_weighted.result().numpy()

    print('Calculating IOU Scores for weights:', weights)
    for i in range(num_models):
        print(f'IOU Score for {model_names[i]} = {iou_score_models[i]}')
    print('IOU Score for equally weighted average ensemble =', iou_score_weighted)
    return iou_score_models[np.argmax(iou_score_models)], iou_score_weighted


def list_available_models(model_path=os.getcwd()):
    models = [file for file in os.listdir(model_path) if file.endswith(".h5")]
    return models


def generate_predictions(loaded_models, x_test, y_test, max_iou_row, j):
        print("\nGenerating predictions for a random test image...")
        test_img_number = random.randint(0, len(x_test))
        test_img = x_test[test_img_number] #(256, 256, 3)
        test_img_input = np.expand_dims(test_img, 0) #(1, 256, 256, 3)
        ground_truth = y_test[test_img_number] #(256, 256, 1)

        test_predictions = []
        for model in loaded_models:
            test_predictions.append(model.predict(test_img_input))
        test_predictions = np.array(test_predictions) #(n_models, 1, 256, 256, 4)

        opt_weights = [max_iou_row[i] for i in range(len(loaded_models))]
        print("Optimal weights used: ", opt_weights)
        weighted_test_predictions = np.tensordot(test_predictions, opt_weights, axes=((0),(0))) #(1, 256, 256, 4)
        weighted_ensemble_test_prediction = np.argmax(weighted_test_predictions, axis=3)[0,:,:] #(256, 256)

        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(test_img[:, :, 0], cmap='gray')
        plt.grid(False)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        plt.subplot(232)
        plt.title('Testing Label')
        plt.imshow(ground_truth[:, :, 0], cmap='jet')
        plt.grid(False)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        plt.subplot(233)
        plt.title('Prediction on test image')
        plt.imshow(weighted_ensemble_test_prediction, cmap='jet')
        plt.grid(False)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        plt.savefig(f'/content/ensemble_prediction_{j + 1}_with_weigths_{opt_weights}.png')
        plt.show()



def create_cm(opt_weighted_ensemble_prediction, optimal_weights, y_test):
    print("\nCreating confusion matrix...")

    labels = [1., 2., 3., 4.]
    cm = confusion_matrix(np.squeeze(y_test).flatten(), np.squeeze(opt_weighted_ensemble_prediction).flatten(), labels=labels)

    fig, ax = plt.subplots(figsize=(6,6))
    sns.set(font_scale=0.8)
    sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

    #Plot fractional incorrect misclassifications
    incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
    incorr_fraction[np.isnan(incorr_fraction)] = 0.0

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.bar(np.arange(len(labels)), incorr_fraction)
    plt.xlabel('True Label')
    plt.ylabel('Fraction of incorrect predictions')
    plt.xticks(np.arange(len(labels)), labels)
    plt.savefig(f'/content/confusion_matrix_with_weights_{optimal_weights}.png')
    plt.show()

    accuracy = accuracy_score(np.squeeze(y_test).flatten(), np.squeeze(opt_weighted_ensemble_prediction).flatten())
    precision = precision_score(np.squeeze(y_test).flatten(), np.squeeze(opt_weighted_ensemble_prediction).flatten(), average='weighted')
    recall = recall_score(np.squeeze(y_test).flatten(), np.squeeze(opt_weighted_ensemble_prediction).flatten(), average='weighted')
    f1 = f1_score(np.squeeze(y_test).flatten(), np.squeeze(opt_weighted_ensemble_prediction).flatten(), average='weighted')

    print("Confusion matrix: \n", cm)
    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ensemble selected models.')
    parser.add_argument('--model_path', type=str, default=os.getcwd(),
                        help='Path to the directory containing models.')
    parser.add_argument('--models', nargs='+', metavar='model',
                        help='List of models to ensemble (e.g., unet_pre-trained linknet_self-trained).')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of samples to use for evaluation.')
    parser.add_argument('--nr_predictions', type=int, default=1,
                        help='Number of test predictions to generate.')

    args = parser.parse_args()

    max_selectable_models = 4
    if len(args.models) > max_selectable_models:
        print(f"Warning: Only {max_selectable_models} models can be selected at once. The first {max_selectable_models} models will be used.")
        args.models = args.models[:max_selectable_models]

    if not args.models:
        print("Error: Please provide at least one model.")
    else:
        models = list_available_models(args.model_path)
        selected_models = []

        for model in args.models:
            if f"{model}.h5" in models:
                selected_models.append(model)
            else:
                print(f"Error: The selected model '{model}' is not available in the specified path.")

        if selected_models:
            print(f"Selected models: {selected_models}")

    loaded_models = []
    for model in selected_models:
        model = load_model(f"{args.model_path}/{model}.h5", compile=False)
        loaded_models.append(model)
        print(f"Model {model} loaded.")

    x_test = np.load('x_test.npz')['arr_0']
    y_test = np.load('y_test.npz')['arr_0']
    x_test_aug = np.load('x_test_aug.npz')['arr_0']
    y_test_aug = np.load('y_test_aug.npz')['arr_0']
    x_test = np.concatenate((x_test, x_test_aug), axis=0)
    y_test = np.concatenate((y_test, y_test_aug), axis=0)

    preprocessing = sm.get_preprocessing("resnet34") #this could later be required to be a parameter
    x_test = preprocessing(x_test)

    try:
        random_indices = random.sample(range(len(x_test)), args.sample_size)
    except Exception as e:
        print(f"Error: The sample size is larger than the number of available samples ({len(x_test)}).")
        print("Please reduce the sample size.")
        print(e)
        exit()
    
    x_test = x_test[random_indices] #performance precaution -> (100, 256, 256, 3)
    y_test = y_test[random_indices] #performance precaution -> (100, 256, 256, 1)
    print("x_test: \t", x_test.shape, np.min(x_test), np.max(x_test), "\t", x_test[0][128][120:121])
    print("y_test: \t", y_test.shape, np.min(y_test), np.max(y_test), "\t", y_test[0][128][120:121], "\tlabels: ", np.unique(y_test))

    n_classes = 4

    predictions = []
    for model in loaded_models:
        predictions.append(model.predict(x_test))
    predictions = np.array(predictions)

    weights = [0.1] * len(selected_models)
    best_single_model_iou, iou_score_equally_weigthed = eval_models(selected_models, weights, predictions, y_test, n_classes)

    max_iou_row = random_gridsearch(predictions, y_test, n_classes)
    optimal_weights = [max_iou_row['wt' + str(i + 1)] for i in range(len(selected_models))]
    ensemble_iou = max_iou_row['IOU']

    print("Best single model's IOU:", best_single_model_iou)
    improvement_percentage = ((ensemble_iou - best_single_model_iou) / best_single_model_iou) * 100
    improvement_absolute = ensemble_iou - best_single_model_iou

    print(f'Ensemble performance is {improvement_percentage:.3f}% better than the best model alone (improvement in IOU: {improvement_absolute:.3f} with weights {optimal_weights})')

    for j in range(args.nr_predictions):
        generate_predictions(loaded_models, x_test, y_test, max_iou_row, j)

    all_test_preds = np.array([model.predict(x_test) for model in loaded_models]) #(n_models, 100, 256, 256, 4)
    print("\nall_test_preds: \t\t\t", all_test_preds.shape, "\t --> should be: (n_models, sample_size, 256, 256, 4)")
    opt_weighted_preds = np.tensordot(all_test_preds, optimal_weights, axes=((0),(0))) #(100, 256, 256, 4)
    print("opt_weighted_preds: \t\t\t", opt_weighted_preds.shape, "\t --> should be: (sample_size, 256, 256, 4)")
    opt_weighted_ensemble_prediction = np.argmax(opt_weighted_preds, axis=3) #(100, 256, 256)
    print("opt_weighted_ensemble_prediction: \t", opt_weighted_ensemble_prediction.shape, "\t --> should be: (sample_size, 256, 256)")
    opt_weighted_ensemble_prediction = np.expand_dims(opt_weighted_ensemble_prediction, 3) #(100, 256, 256, 1)
    print("opt_weighted_ensemble_prediction: \t", opt_weighted_ensemble_prediction.shape, "\t --> should be: (sample_size, 256, 256, 1)")
    opt_weighted_ensemble_prediction = opt_weighted_ensemble_prediction / 1.

    print("y_test: \t\t\t\t", y_test.shape, "\t --> should be: (sample_size, 256, 256, 1)") #(100, 256, 256, 1)

    if not np.all(np.unique(opt_weighted_ensemble_prediction) == np.unique(y_test)):
        print("Error: The unique labels in the predictions and the ground truth do not match:", np.unique(opt_weighted_ensemble_prediction), "vs.", np.unique(y_test))

    if np.squeeze(opt_weighted_ensemble_prediction).shape != np.squeeze(y_test).shape:
        print("Error: The shape of the predictions and the ground truth do not match:", np.squeeze(opt_weighted_ensemble_prediction).shape, "vs.", np.squeeze(y_test).shape)

    create_cm(opt_weighted_ensemble_prediction, optimal_weights, y_test)
    