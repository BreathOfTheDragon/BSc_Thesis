import os
import sys
import json
import torch
import scipy
import scipy.io
import pickle
import sklearn
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

# this code test the robustness of brain data
# we do this for the brain data averaged from 370 to 499
# so , the brain data in here means the data from 370 to 499 averaged
# for brain :
# we train a classifier with brain BH datas , and test the classifier with brain BI datas
# we train a classifier with brain BH datas , and test the classifier with brain BL datas
# we train a classifier with brain BI datas , and test the classifier with brain BH datas
# we train a classifier with brain BI datas , and test the classifier with brain BL datas
# we train a classifier with brain BL datas , and test the classifier with brain BH datas
# we train a classifier with brain BL datas , and test the classifier with brain BI datas
# we plot the accuracy of brain for BX train / BX test for only the 370/499 time interval
# this gives us the plot with 6 BX/BX bars

#   ***************************************************************************
#   |  this code uses 6 body images , bodies are 3 human and 3 animal bodies  |
#   |  this code uses 6 face images , faces are 3 human and 3 animal faces    |
#   ***************************************************************************

model_names = 'Brain'

neuron_output = scipy.io.loadmat(
    "C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat")[
    'neurons_stacked_averaged_for_370_to_499']

print(neuron_output)
train_output = ['face', 'face', 'face', 'face', 'face', 'face', 'body', 'body', 'body', 'body',
                'body', 'body', 'natural', 'natural', 'natural', 'natural', 'natural', 'natural',
                'manmade', 'manmade', 'manmade', 'manmade', 'manmade', 'manmade']

test_output = ['face', 'face', 'face', 'face', 'face', 'face', 'body', 'body', 'body', 'body',
                'body', 'body', 'natural', 'natural', 'natural', 'natural', 'natural', 'natural',
                'manmade', 'manmade', 'manmade', 'manmade', 'manmade', 'manmade']

train_test_labels = ["BH / BI ", "BH / BL ", "BI / BH ", "BI / BL ", "BL / BH ",
                     "BL / BI "]

F, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
print(ax1)
accuracies = []
train_input_BH = np.transpose(neuron_output[:, 77:101])
train_input_BI = np.transpose(neuron_output[:, 104:128])
train_input_BL = np.transpose(neuron_output[:, 131:155])

test_input_BH = np.transpose(neuron_output[:, 77:101])
test_input_BI = np.transpose(neuron_output[:, 104:128])
test_input_BL = np.transpose(neuron_output[:, 131:155])

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
    'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\BFNM\\'
    'Models_Brain BX_BX accuracies and confusion matrix\\' +
    model_names + ' Confusion Matrix.png',
    dpi=200)
plt.close()
print("********************************************")

plt.figure(figsize=(30, 30))
plt.yticks(fontsize=30)
plt.ylim(0, 1.0)
plt.bar(train_test_labels, accuracies, color='green', linewidth=3,
        label=model_names)
plt.plot()
plt.legend(fontsize=30)
plt.xticks(fontsize=35, rotation=90)
plt.ylabel("Accuracy", fontsize=30)
plt.title("Train / Test", fontsize=55)
plt.draw()
plt.savefig(
    'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\BFNM\\'
    'Models_Brain BX_BX accuracies and confusion matrix\\' +
    model_names + ' Accuracy.png',
    dpi=100)
plt.close()
scipy.io.savemat(
    'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\BFNM\\'
    'Brain BX_BX_Accuracy 0_900 mats\\' + model_names +
    '_for_370_to_499.mat', {'accuracies': accuracies})
