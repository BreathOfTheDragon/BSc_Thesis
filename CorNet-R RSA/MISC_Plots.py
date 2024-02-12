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
color = ['red', 'blue']

slice_names = ['V1.conv_input',
               'V1.norm_input',
               'V1.nonlin_input',
               'V1.conv1',
               'V1.norm1',
               'V1.nonlin1',
               'V1.output',
               'V2.conv_input',
               'V2.norm_input',
               'V2.nonlin_input',
               'V2.conv1',
               'V2.norm1',
               'V2.nonlin1',
               'V2.output',
               'V4.conv_input',
               'V4.norm_input',
               'V4.nonlin_input',
               'V4.conv1',
               'V4.norm1',
               'V4.nonlin1',
               'V4.output',
               'IT.conv_input',
               'IT.norm_input',
               'IT.nonlin_input',
               'IT.conv1',
               'IT.norm1',
               'IT.nonlin1',
               'IT.output',
               'decoder.avgpool',
               'decoder.flatten',
               'decoder.linear'
               ]

comparison_mats_directory = "C:\\Users\\Asus\\Desktop\\Project\\CorNet-R\\RDM Comparisons B mats\\"
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
glist = [0, 1, 2, 6, 7, 8]
kkk = 1
gg = 0
for g in glist:
    plt.figure(figsize=(20, 20))
    plt.yticks(fontsize=20)
    plt.ylim(-1.0, 1.0)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=10, rotation=90)
    plt.title("CorNet-R", fontsize=25)
    plt.ylabel("Correlation", fontsize=25)
    plt.plot(slices, big_values[g], color=color[(gg % 2)], linewidth=3, label=type_of_image[g])
    plt.plot(slices, big_values[g + 3], color=color[(gg + 1) % 2], linewidth=3, label=type_of_image[g + 3])
    plt.legend(fontsize=30)
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\CorNet-R\\RDM Comparisons MISC Plots\\' + str(kkk) + '.png',
        dpi=100)
    plt.close()
    kkk += 1
    gg += 2
