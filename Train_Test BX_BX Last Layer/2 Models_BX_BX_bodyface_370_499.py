import os
import sys
import json
import torch
import pickle
import sklearn
import scipy
import scipy.io
import tifffile
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from sklearn import svm
from sklearn import metrics
import torch.optim as optim
from pydoc import importfile
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as dataset
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor


# this code test the robustness of NN results
# for a NN :
# we train a classifier with BH datas , and test the classifier with BI datas
# we train a classifier with BH datas , and test the classifier with BL datas
# we train a classifier with BI datas , and test the classifier with BH datas
# we train a classifier with BI datas , and test the classifier with BL datas
# we train a classifier with BL datas , and test the classifier with BH datas
# we train a classifier with BL datas , and test the classifier with BI datas


#   ***************************************************************************
#   |  this code uses 6 body images , bodies are 3 human and 3 animal bodies  |
#   |  this code uses 6 face images , faces are 3 human and 3 animal faces    |
#   ***************************************************************************


model_names = ['AlexNet', 'CorNet-R', 'CorNet-S', 'CorNet-Z', 'GoogleNet', 'MobileNet-V2', 'ResNet18', 'ResNet50',
               'ResNet101',
               'SqueezeNet1_0',
               'SqueezeNet1_1', 'VGG-16', 'VGG-19']

last_layer_directory = "C:\\Users\\Asus\Desktop\\Project\\Last Layer Before Classifier For All Models\\"
last_layers_before_classifier_list = []
k = 0
for file in os.listdir(last_layer_directory):
    if file == 'desktop.ini':
        pass
    else:
        last_layers_before_classifier_list.append(pd.read_pickle(last_layer_directory + model_names[k] + '.p'))
        print(last_layer_directory + model_names[k] + '.p')
        k += 1

train_output = ['face', 'face', 'face', 'face', 'face', 'face', 'body', 'body', 'body', 'body',
                'body', 'body']
test_output = ['face', 'face', 'face', 'face', 'face', 'face', 'body', 'body', 'body', 'body',
               'body', 'body']
train_test_labels = ["BH / BI ", "BH / BL ", "BI / BH ", "BI / BL ", "BL / BH ",
                     "BL / BI "]

# sys.exit()

for i in range(len(model_names)):
    F, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
    accuracies = []
    train_input_BH = np.transpose(last_layers_before_classifier_list[i][:, 77:89])
    train_input_BI = np.transpose(last_layers_before_classifier_list[i][:, 104:116])
    train_input_BL = np.transpose(last_layers_before_classifier_list[i][:, 131:143])

    test_input_BH = np.transpose(last_layers_before_classifier_list[i][:, 77:89])
    test_input_BI = np.transpose(last_layers_before_classifier_list[i][:, 104:116])
    test_input_BL = np.transpose(last_layers_before_classifier_list[i][:, 131:143])

    # <editor-fold desc="BH train">
    my_classifier = svm.SVC(kernel='linear')
    my_classifier.fit(train_input_BH, train_output)
    prediction = my_classifier.predict(test_input_BI)
    accuracies.append(metrics.accuracy_score(test_output, prediction))
    print("BH train / BI test Accuracy:", metrics.accuracy_score(test_output, prediction))
    print(sklearn.metrics.confusion_matrix(test_output, prediction))
    ax1.set_title("BH / BI")
    sklearn.metrics.plot_confusion_matrix(estimator=my_classifier, X=test_input_BI, y_true=test_output,
                                          ax=ax1, cmap='Blues')

    my_classifier = svm.SVC(kernel='linear')
    my_classifier.fit(train_input_BH, train_output)
    prediction = my_classifier.predict(test_input_BL)
    accuracies.append(metrics.accuracy_score(test_output, prediction))
    print("BH train / BL test Accuracy:", metrics.accuracy_score(test_output, prediction))
    print(sklearn.metrics.confusion_matrix(test_output, prediction))
    ax2.set_title("BH / BL")
    sklearn.metrics.plot_confusion_matrix(estimator=my_classifier, X=test_input_BL, y_true=test_output,
                                          ax=ax2, cmap='Blues')

    # </editor-fold>

    # <editor-fold desc="BI train">
    my_classifier = svm.SVC(kernel='linear')
    my_classifier.fit(train_input_BI, train_output)
    prediction = my_classifier.predict(test_input_BH)
    accuracies.append(metrics.accuracy_score(test_output, prediction))
    print("BI train / BH test Accuracy:", metrics.accuracy_score(test_output, prediction))
    print(sklearn.metrics.confusion_matrix(test_output, prediction))
    ax3.set_title("BI / BH")
    sklearn.metrics.plot_confusion_matrix(estimator=my_classifier, X=test_input_BH, y_true=test_output,
                                          ax=ax3, cmap='Blues')

    my_classifier = svm.SVC(kernel='linear')
    my_classifier.fit(train_input_BI, train_output)
    prediction = my_classifier.predict(test_input_BL)
    accuracies.append(metrics.accuracy_score(test_output, prediction))
    print("BI train / BL test Accuracy:", metrics.accuracy_score(test_output, prediction))
    print(sklearn.metrics.confusion_matrix(test_output, prediction))
    ax4.set_title("BI / BL")
    sklearn.metrics.plot_confusion_matrix(estimator=my_classifier, X=test_input_BL, y_true=test_output,
                                          ax=ax4, cmap='Blues')

    # </editor-fold>

    # <editor-fold desc="BL train">
    my_classifier = svm.SVC(kernel='linear')
    my_classifier.fit(train_input_BL, train_output)
    prediction = my_classifier.predict(test_input_BH)
    accuracies.append(metrics.accuracy_score(test_output, prediction))
    print("BL train / BH test Accuracy:", metrics.accuracy_score(test_output, prediction))
    print(sklearn.metrics.confusion_matrix(test_output, prediction))
    ax5.set_title("BL / BH")
    sklearn.metrics.plot_confusion_matrix(estimator=my_classifier, X=test_input_BH, y_true=test_output,
                                          ax=ax5, cmap='Blues')

    my_classifier = svm.SVC(kernel='linear')
    my_classifier.fit(train_input_BL, train_output)
    prediction = my_classifier.predict(test_input_BI)
    accuracies.append(metrics.accuracy_score(test_output, prediction))
    print("BL train / BI test Accuracy:", metrics.accuracy_score(test_output, prediction))
    print(sklearn.metrics.confusion_matrix(test_output, prediction))
    ax6.set_title("BL / BI")
    sklearn.metrics.plot_confusion_matrix(estimator=my_classifier, X=test_input_BI, y_true=test_output,
                                          ax=ax6, cmap='Blues')

    # </editor-fold>

    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\bodyface\\'
        'Models_Brain BX_BX accuracies and confusion matrix\\' +
        model_names[i] + ' Confusion Matrix.png',
        dpi=200)
    plt.close()
    print("********************************************")
    # <editor-fold desc="Save Accuracy Values As Plot">
    plt.figure(figsize=(30, 30))
    plt.yticks(fontsize=30)
    plt.ylim(0, 1.0)
    plt.bar(train_test_labels, accuracies, color='green', linewidth=3,
            label=model_names[i])
    plt.plot()
    plt.legend(fontsize=30)
    plt.xticks(fontsize=35, rotation=90)
    plt.ylabel("Accuracy", fontsize=30)
    plt.title("Train / Test", fontsize=55)
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\bodyface\\'
        'Models_Brain BX_BX accuracies and confusion matrix\\' +
        model_names[i] + ' Accuracy.png',
        dpi=100)
    plt.close()
    # </editor-fold>
    scipy.io.savemat(
        'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\bodyface\\'
        'model BX_BX accuracies mats\\' + model_names[
            i] + '.mat',
        {'accuracies': accuracies})

print("hi")
