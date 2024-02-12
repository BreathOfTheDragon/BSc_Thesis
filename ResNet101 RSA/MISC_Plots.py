import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import pickle
import sys
import os
import scipy.spatial
import scipy.stats
import pandas as pd
import scipy
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

type_of_image = ['Face_HF', 'Face_IF', 'Face_LF', 'Body_HF', 'Body_IF', 'Body_LF', 'Animate_HF', 'Animate_IF',
                 'Animate_LF', 'Inanimate_HF',
                 'Inanimate_IF', 'Inanimate_LF']
color = ['red', 'blue']
numbers = [7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10]

res101 = models.resnet101(pretrained=True)
slice_names = np.array([])  # a list of the names of all slices of the network
slice_names = np.append(slice_names, np.array(models.feature_extraction.get_graph_node_names(res101)[0]))

comparison_mats_directory = "C:\\Users\\Asus\\Desktop\\Project\\ResNet101\\RDM Comparisons B mats\\"
comparison_mats = []
comparison_mat_no_mat = []
for u in range(len(type_of_image)):
    comparison_mats.append(
        scipy.io.loadmat(comparison_mats_directory + str(numbers[u]) + '_RDM_comparison_' + type_of_image[u]))
big_values = []

for l in range(len(comparison_mats)):
    comparisons = comparison_mats[l]['RDM_comparison']
    comparisons = comparisons.squeeze()
    data = dict.fromkeys(slice_names, None)
    for i in range(len(slice_names)):
        data[slice_names[i]] = comparisons[i]
    print(comparisons.shape)
    slices = list(data.keys())
    values = list(data.values())
    print(slices)
    print(values)
    big_values.append(values)

glist = [0, 1, 2, 6, 7, 8]
kkk = 0
gg = 0
type_of_image2 = ['Face_Body_HF', 'Face_Body_IF', 'Face_Body_LF', 'Animate_Inanimate_HF', 'Animate_Inanimate_IF',
                  'Animate_Inanimate_LF']
for g in glist:
    plt.figure(figsize=(20, 20))
    plt.yticks(fontsize=20)
    plt.ylim(-1.0, 1.0)
    plt.legend(fontsize=15)
    plt.xticks(np.arange(0, len(slices[1:]), 5), fontsize=10, rotation=90)
    plt.title("ResNet101", fontsize=25)
    plt.ylabel("Correlation", fontsize=25)
    plt.plot(slices[1:], big_values[g][1:], color=color[(gg % 2)], linewidth=3, label=type_of_image[g])
    plt.plot(slices[1:], big_values[g + 3][1:], color=color[(gg + 1) % 2], linewidth=3, label=type_of_image[g + 3])
    plt.legend(fontsize=30)
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\ResNet101\\RDM Comparisons MISC Plots\\' + str(kkk + 1) + '_' +
        type_of_image2[
            kkk] + '.png',
        dpi=100)
    plt.close()
    kkk += 1
    gg += 2

type_of_image3 = ['Face_HIL', 'Body_HIL', 'Animate_HIL', 'Inanimate_HIL']
color = ['red', 'blue', 'green']
glist = [0, 3, 6, 9]
kkk = 0
for g in glist:
    plt.figure(figsize=(20, 20))
    plt.yticks(fontsize=20)
    plt.ylim(-1.0, 1.0)
    plt.legend(fontsize=15)
    plt.xticks(np.arange(0, len(slices[1:]), 5), fontsize=10, rotation=90)
    plt.title("ResNet101", fontsize=25)
    plt.ylabel("Correlation", fontsize=25)
    plt.plot(slices[1:], big_values[g][1:], color=color[g % 3], linewidth=3, label=type_of_image[g])
    plt.plot(slices[1:], big_values[g + 1][1:], color=color[(g + 1) % 3], linewidth=3, label=type_of_image[g + 1])
    plt.plot(slices[1:], big_values[g + 2][1:], color=color[(g + 2) % 3], linewidth=3, label=type_of_image[g + 2])
    plt.legend(fontsize=30)
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\ResNet101\\RDM Comparisons MISC Plots\\' + str(kkk + 7) + '_' +
        type_of_image3[
            kkk] + '.png',
        dpi=100)
    plt.close()
    kkk += 1
