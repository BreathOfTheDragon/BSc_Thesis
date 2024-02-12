import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import pickle
import os
import pandas as pd
import scipy
from torchvision import models

type_of_image = ['Face_HF', 'Face_IF', 'Face_LF', 'Body_HF', 'Body_IF', 'Body_LF', 'Animate_HF', 'Animate_IF',
                 'Animate_LF', 'Inanimate_HF',
                 'Inanimate_IF', 'Inanimate_LF']
color = ['red', 'blue', 'green']

alexnet = models.alexnet(pretrained=True)
slice_names = np.array([])  # a list of the names of all slices of the network
slice_names = np.append(slice_names, np.array(models.feature_extraction.get_graph_node_names(alexnet)[0]))

comparison_mats_directory = "E:\\Project\\AlexNet\\RDM Comparisons B mats\\"
comparison_mats = []
comparison_mat_no_mat = []
for u in range(len(type_of_image)):
    comparison_mats.append(
        scipy.io.loadmat(comparison_mats_directory + 'RDM_comparison_' + type_of_image[u]))
big_values = []
for l in range(len(comparison_mats)):
    comparisons = comparison_mats[l]['RDM_comparison' + str(int(l / 3) + 7)]
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
glist = [0, 3, 6, 9]
kkk = 7
for g in glist:
    plt.figure(figsize=(20, 20))
    plt.yticks(fontsize=20)
    plt.ylim(-1.0, 1.0)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=10, rotation=90)
    plt.title("AlexNet", fontsize=25)
    plt.ylabel("Correlation", fontsize=25)
    plt.plot(slices, big_values[g], color=color[g % 3], linewidth=3, label=type_of_image[g])
    plt.plot(slices, big_values[g + 1], color=color[(g + 1) % 3], linewidth=3, label=type_of_image[g + 1])
    plt.plot(slices, big_values[g + 2], color=color[(g + 2) % 3], linewidth=3, label=type_of_image[g + 2])
    plt.legend(fontsize=30)
    plt.draw()
    plt.savefig(
        'E:\\Project\\AlexNet\\RDM Comparisons MISC Plots\\' + str(kkk) + '.png',
        dpi=100)
    plt.close()
    kkk += 1
