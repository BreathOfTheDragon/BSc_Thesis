import os
import sys
import json
import torch
import scipy
import pickle
import sklearn
import scipy.io
import tifffile
import torchvision
import statistics
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
# we do this for the brain data averaged from x to (x + 49)
# so , the brain data in here means the data from x to (x + 49) averaged
# for brain :
# we train a classifier with brain BH datas , and test the classifier with brain BI datas
# we train a classifier with brain BH datas , and test the classifier with brain BL datas
# we train a classifier with brain BI datas , and test the classifier with brain BH datas
# we train a classifier with brain BI datas , and test the classifier with brain BL datas
# we train a classifier with brain BL datas , and test the classifier with brain BH datas
# we train a classifier with brain BL datas , and test the classifier with brain BI datas
# we do this for the brain data averaged from 0 to 49 , and add 5ms to interval , until we reach end of 900ms data
# we do this for brain data averaged from 0 to 49 , 5 to 54 , 10 to 59 , ... , 850 to 899
# we plot the accuracy of brain for BX train / BX test through experiment time
# in the end we also average the accuracies for the 6 possible BX/BX combinations and plot with standard deviation

#   ***************************************************************************
#   |  this code uses 6 body images , bodies are 3 human and 3 animal bodies  |
#   |  this code uses 6 face images , faces are 3 human and 3 animal faces    |
#   ***************************************************************************


model_names = 'Brain'
neuron_output = []
mat_directory = 'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked 50_50_50\\'
for file in sorted(os.listdir(mat_directory)):
    print(file)
    if file == 'desktop.ini':
        pass
    else:
        neuron_output.append(scipy.io.loadmat(mat_directory + file))

train_output = ['face', 'face', 'face', 'face', 'face', 'face', 'body', 'body', 'body', 'body',
                'body', 'body', 'natural', 'natural', 'natural', 'natural', 'natural', 'natural',
                'manmade', 'manmade', 'manmade', 'manmade', 'manmade', 'manmade']

test_output = ['face', 'face', 'face', 'face', 'face', 'face', 'body', 'body', 'body', 'body',
               'body', 'body', 'natural', 'natural', 'natural', 'natural', 'natural', 'natural',
               'manmade', 'manmade', 'manmade', 'manmade', 'manmade', 'manmade']

train_test_labels = ["BH / BI ", "BH / BL ", "BI / BH ", "BI / BL ", "BL / BH ",
                     "BL / BI "]

print(len(neuron_output))
print(type(neuron_output[0]))
print(neuron_output[0])
print(neuron_output[0]["neurons_stacked_averaged_for_0_to_49"])
accuracies_BH_BI = []
accuracies_BH_BL = []
accuracies_BL_BH = []
accuracies_BL_BI = []
accuracies_BI_BH = []
accuracies_BI_BL = []

time = []
colors = ['green', 'blue', 'red', 'black', 'purple', 'orange']
F, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

for kkk in range(len(neuron_output)):
    time.append(str(((-300 + 5 * kkk) + (-300 + 49 + 5 * kkk)) / 2))
    train_input_BH = np.transpose(
        neuron_output[kkk]["neurons_stacked_averaged_for_" + str(0 + 5 * kkk) + "_to_" + str(49 + 5 * kkk)][:, 77:101])
    train_input_BI = np.transpose(
        neuron_output[kkk]["neurons_stacked_averaged_for_" + str(0 + 5 * kkk) + "_to_" + str(49 + 5 * kkk)][:, 104:128])
    train_input_BL = np.transpose(
        neuron_output[kkk]["neurons_stacked_averaged_for_" + str(0 + 5 * kkk) + "_to_" + str(49 + 5 * kkk)][:, 131:155])

    test_input_BH = np.transpose(
        neuron_output[kkk]["neurons_stacked_averaged_for_" + str(0 + 5 * kkk) + "_to_" + str(49 + 5 * kkk)][:, 77:101])
    test_input_BI = np.transpose(
        neuron_output[kkk]["neurons_stacked_averaged_for_" + str(0 + 5 * kkk) + "_to_" + str(49 + 5 * kkk)][:, 104:128])
    test_input_BL = np.transpose(
        neuron_output[kkk]["neurons_stacked_averaged_for_" + str(0 + 5 * kkk) + "_to_" + str(49 + 5 * kkk)][:, 131:155])

    # <editor-fold desc="BH train">
    my_classifier = svm.SVC(kernel='linear')
    my_classifier.fit(train_input_BH, train_output)
    prediction = my_classifier.predict(test_input_BI)
    accuracies_BH_BI.append(metrics.accuracy_score(test_output, prediction))
    my_classifier = svm.SVC(kernel='linear')
    my_classifier.fit(train_input_BH, train_output)
    prediction = my_classifier.predict(test_input_BL)
    accuracies_BH_BL.append(metrics.accuracy_score(test_output, prediction))
    my_classifier = svm.SVC(kernel='linear')
    my_classifier.fit(train_input_BI, train_output)
    prediction = my_classifier.predict(test_input_BH)
    accuracies_BI_BH.append(metrics.accuracy_score(test_output, prediction))
    my_classifier = svm.SVC(kernel='linear')
    my_classifier.fit(train_input_BI, train_output)
    prediction = my_classifier.predict(test_input_BL)
    accuracies_BI_BL.append(metrics.accuracy_score(test_output, prediction))
    my_classifier = svm.SVC(kernel='linear')
    my_classifier.fit(train_input_BL, train_output)
    prediction = my_classifier.predict(test_input_BH)
    accuracies_BL_BH.append(metrics.accuracy_score(test_output, prediction))
    my_classifier = svm.SVC(kernel='linear')
    my_classifier.fit(train_input_BL, train_output)
    prediction = my_classifier.predict(test_input_BI)
    accuracies_BL_BI.append(metrics.accuracy_score(test_output, prediction))

# <editor-fold desc="Save Accuracy Values As Plot">
train_test_labels = ["BH_BI ", "BH_BL ", "BI_BH ", "BI_BL ", "BL_BH ", "BL_BI "]
all_accuracies = [accuracies_BH_BI,
                  accuracies_BH_BL,
                  accuracies_BI_BH,
                  accuracies_BI_BL,
                  accuracies_BL_BH,
                  accuracies_BL_BI]
labels = ["BH / BI ", "BH / BL ", "BI / BH ", "BI / BL ", "BL / BH ",
          "BL / BI "]

print(all_accuracies)

average = []
all_stds = []
for ii in range(len(time)):
    little_medium = 0

    accuracies_values = []
    for jj in range(len(all_accuracies)):
        little_medium += all_accuracies[jj][ii]
        accuracies_values.append(all_accuracies[jj][ii])
    std = statistics.stdev(accuracies_values)
    all_stds.append(std)
    average.append(little_medium / len(all_accuracies))

for lll in range(len(labels)):
    plt.figure(figsize=(30, 30))
    plt.yticks(fontsize=30)
    plt.ylim(0, 1.0)
    plt.plot(time, all_accuracies[lll], color=colors[lll], linewidth=2,
             label=labels[lll])
    plt.legend(fontsize=30)
    plt.xticks(np.arange(0, len(time), 5), fontsize=30, rotation=90)
    plt.ylabel("Accuracy", fontsize=30)
    plt.title("Train / Test", fontsize=55)
    plt.legend(fontsize=25)
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\BFNM\\Brain BX_BX_Accuracy 0_900\\'
        + train_test_labels[
            lll] + '_' +
        model_names + ' Accuracy.png',
        dpi=300)
    plt.close()

plt.figure(figsize=(30, 30))
plt.yticks(fontsize=30)
plt.ylim(0, 1.0)
for lll in range(len(labels)):
    plt.plot(time, all_accuracies[lll], color=colors[lll], linewidth=2,
             label=labels[lll])
plt.legend(fontsize=30)
plt.xticks(np.arange(0, len(time), 5), fontsize=30, rotation=90)
plt.ylabel("Accuracy", fontsize=30)
plt.title("Train / Test", fontsize=55)
plt.legend(fontsize=25)

plt.draw()
plt.savefig(
    'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\BFNM\\Brain BX_BX_Accuracy 0_900\\' + 'BX_BX_' +
    model_names + ' Accuracy.png',
    dpi=300)
plt.close()
# </editor-fold>

plt.figure(figsize=(30, 30))
plt.yticks(fontsize=30)
plt.ylim(0, 1.0)
plt.errorbar(time, average, yerr=all_stds, ecolor='blue', color='red', linewidth=8,
             label='average')
plt.legend(fontsize=30)
plt.xticks(np.arange(0, len(time), 5), fontsize=30, rotation=90)
plt.ylabel("Accuracy", fontsize=30)
plt.title("Train / Test", fontsize=55)
plt.legend(fontsize=25)
plt.draw()
plt.savefig(
    'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\BFNM\\'
    'Brain BX_BX_Accuracy 0_900\\\\' + 'BX_BX_average_' +
    model_names + ' Accuracy.png',
    dpi=300)
plt.close()
# </editor-fold>


average_pos = []
average_neg = []

for lll in range(len(time)):
    average_pos.append(average[lll] + all_stds[lll])
    average_neg.append(average[lll] - all_stds[lll])

plt.figure(figsize=(30, 30))
plt.yticks(fontsize=30)
plt.ylim(0, 1.0)
plt.fill_between(time, average, average_pos, color='blue', alpha=.3, edgecolor="blue", linewidth=5)
plt.fill_between(time, average, average_neg, color='blue', alpha=.3, edgecolor="blue", linewidth=5)
plt.legend(fontsize=30)
plt.xticks(np.arange(0, len(time), 5), fontsize=30, rotation=90)
plt.ylabel("Accuracy", fontsize=30)
plt.title("Train / Test", fontsize=55)
plt.draw()
plt.savefig(
    'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\BFNM\\'
    'Brain BX_BX_Accuracy 0_900\\' + 'BX_BX_average_2_' +
    model_names + ' Accuracy.png',
    dpi=300)
plt.close()
# </editor-fold>


for t in range(len(all_accuracies[0])):
    interval_accuracies = []
    for q in range(len(all_accuracies)):
        interval_accuracies.append(all_accuracies[q][t])
    scipy.io.savemat(
        'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\BFNM\\'
        'Brain BX_BX_Accuracy 0_900 mats\\Brain_for_'
        + str((5 * t)) + "_to_" + str(49 + 5 * t) + '.mat', {'accuracies': interval_accuracies})
