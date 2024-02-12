import os
import sys
import json
import torch
import pickle
import sklearn
import scipy
import scipy.io
import tifffile
import statistics
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
from scipy.stats import chisquare
import scipy.spatial.distance

times0 = []
for kkk in range(0, 171):
    times0.append(5 * kkk)
print(times0)

all_jensenshannon_distances = []
all_pearson_correlation = []
all_euclidean = []

for yy in range(len(times0)):
    brain = scipy.io.loadmat(
        "C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\brain accuracies mats\\Brain_for_" + str(
            times0[yy]) + "_to_" + str(times0[yy] + 49) + ".mat")
    brain = brain['accuracies'][0]
    print(brain)

    jensenshannon_distances = []
    pearson_correlation = []
    canberra = []
    hamming = []
    euclidean = []
    model_names = ['AlexNet', 'CorNet-R', 'CorNet-S', 'CorNet-Z', 'GoogleNet', 'MobileNet-V2', 'ResNet18', 'ResNet50',
                   'ResNet101',
                   'SqueezeNet1_0',
                   'SqueezeNet1_1', 'VGG-16', 'VGG-19']
    for i in range(len(model_names)):
        model = scipy.io.loadmat(
            "C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\model accuracies mats\\" + model_names[
                i] + ".mat")
        model = model['accuracies'][0]
        print(model_names[i])
        print(model)
        # print(chisquare(f_obs=model, f_exp=brain))
        print(scipy.spatial.distance.jensenshannon(brain, model))
        jensenshannon_distances.append(scipy.spatial.distance.jensenshannon(brain, model))
        pearson_correlation.append(scipy.spatial.distance.correlation(brain, model))
        euclidean.append(scipy.spatial.distance.euclidean(brain, model))

    print(jensenshannon_distances)
    print("Best model according to jensen shannon :\n",
          model_names[jensenshannon_distances.index(min(jensenshannon_distances))])

    print(pearson_correlation)
    print("Best model according to pearson correlation :\n",
          model_names[pearson_correlation.index(min(pearson_correlation))])

    print(euclidean)
    print("Best model according to euclidean :\n",
          model_names[euclidean.index(min(euclidean))])

    # print(scipy.stats.rankdata(jensenshannon_distances))
    # print(statistics.stdev(jensenshannon_distances))
    # print(scipy.stats.rankdata(pearson_correlation))
    # print(statistics.stdev(pearson_correlation))
    # print(scipy.stats.rankdata(euclidean))
    # print(statistics.stdev(euclidean))

    all_jensenshannon_distances.append(jensenshannon_distances)
    all_pearson_correlation.append(pearson_correlation)
    all_euclidean.append(euclidean)

model_names = ['AlexNet', 'CorNet-R', 'CorNet-S', 'CorNet-Z', 'GoogleNet', 'MobileNet-V2', 'ResNet18', 'ResNet50',
               'ResNet101',
               'SqueezeNet1_0',
               'SqueezeNet1_1', 'VGG-16', 'VGG-19']
thirteen_counter = 0
time_time = []
for ttt in range(len(times0)):
    time_time.append(str(-300 + times0[ttt]) + '_' + str(-300 + 49 + times0[ttt]))
for model_name in model_names:
    a = []
    b = []
    c = []
    for iii in range(len(times0)):
        a.append(all_jensenshannon_distances[iii][thirteen_counter])
        b.append(all_pearson_correlation[iii][thirteen_counter])
        c.append(all_euclidean[iii][thirteen_counter])
    plt.figure(figsize=(30, 30))
    plt.yticks(fontsize=30)
    plt.ylim(0, 2)
    plt.plot(time_time, a, color='green',
             linewidth=3,
             label='jensen shannon')
    plt.plot(time_time, b, color='red',
             linewidth=3,
             label='pearson correlation')
    plt.plot(time_time, c, color='blue',
             linewidth=3,
             label='euclidean')
    plt.legend(fontsize=30)
    plt.xticks(np.arange(0, len(time_time), 5), fontsize=35, rotation=90)
    plt.ylabel("distances", fontsize=30)
    plt.title(model_name + ' in 50_50_50 times', fontsize=55)
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\comparing 50_50_50\\' +
        model_name + ' distances 50_50_50.png',
        dpi=100)
    thirteen_counter += 1
    plt.close()

print("hi")
