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

# this code compares the results of BX/BX accuracies (6 bars) for brain and each NN
# we use 3 metrics : jensen shannon , euclidean and correlation
# the result is a list of models , drom best to worst , and their values listed , as a txt file


brain = scipy.io.loadmat(
    "C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\BFNM double human face\\"
    "Brain BX_BX_Accuracy 0_900 mats\\Brain_for_370_to_499.mat")
brain = brain['accuracies'][0]
print(brain)

jensenshannon_distances = []
pearson_correlation = []
canberra = []
hamming = []
euclidean = []
protocols = []
model_names = ['AlexNet', 'CorNet-R', 'CorNet-S', 'CorNet-Z', 'GoogleNet', 'MobileNet-V2', 'ResNet18', 'ResNet50',
               'ResNet101',
               'SqueezeNet1_0',
               'SqueezeNet1_1', 'VGG-16', 'VGG-19']
for i in range(len(model_names)):
    model = scipy.io.loadmat(
        "C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\BFNM double human face\\"
        "model BX_BX accuracies mats\\" + model_names[
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


protocols = [jensenshannon_distances, pearson_correlation, euclidean]
protocol_names = ['jensenshannon_distances', 'pearson_correlation', 'euclidean']
rr = 0

for protocol in protocols:
    FILE = open('C:/Users/Asus/Desktop/Project/Comparing Models Last Layer/BX_BX/BFNM double human face/Rankings/'
                + protocol_names[rr] + '.txt', "w")
    model_value_sorted = []
    models_sorted = [x for _, x in sorted(zip(protocol, model_names))]
    values_sorted = sorted(protocol)
    for kj in range(len(model_names)):
        model_value_sorted.append(models_sorted[kj] + "          " + str(values_sorted[kj]))
    with open('C:\\Users\\Asus\\Desktop\\Project\\Comparing Models Last Layer\\BX_BX\\'
              'BFNM double human face\\Rankings\\' + str(
        protocol_names[rr]) + '.txt',
              'w') as f:
        for kkj in range(len(model_value_sorted)):
            print(model_value_sorted[kkj])
            FILE.write(model_value_sorted[kkj] + "\n")
    print(model_value_sorted)
    rr += 1
