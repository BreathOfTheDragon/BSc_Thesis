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

slice_names = ['V1.conv1',
               'V1.norm1',
               'V1.nonlin1',
               'V1.pool',
               'V1.conv2',
               'V1.norm2',
               'V1.nonlin2',
               'V1.output',
               'V2.conv_input',
               'V2.skip',
               'V2.norm_skip',
               'V2.conv1',
               'V2.nonlin1',
               'V2.conv2',
               'V2.nonlin2',
               'V2.conv3',
               'V2.nonlin3',
               'V2.output',
               'V2.norm1_0',
               'V2.norm2_0',
               'V2.norm3_0',
               'V2.norm1_1',
               'V2.norm2_1',
               'V2.norm3_1',
               'V4.conv_input',
               'V4.skip',
               'V4.norm_skip',
               'V4.conv1',
               'V4.nonlin1',
               'V4.conv2',
               'V4.nonlin2',
               'V4.conv3',
               'V4.nonlin3',
               'V4.output',
               'V4.norm1_0',
               'V4.norm2_0',
               'V4.norm3_0',
               'V4.norm1_1',
               'V4.norm2_1',
               'V4.norm3_1',
               'V4.norm1_2',
               'V4.norm2_2',
               'V4.norm3_2',
               'V4.norm1_3',
               'V4.norm2_3',
               'V4.norm3_3',
               'IT.conv_input',
               'IT.skip',
               'IT.norm_skip',
               'IT.conv1',
               'IT.nonlin1',
               'IT.conv2',
               'IT.nonlin2',
               'IT.conv3',
               'IT.nonlin3',
               'IT.output',
               'IT.norm1_0',
               'IT.norm2_0',
               'IT.norm3_0',
               'IT.norm1_1',
               'IT.norm2_1',
               'IT.norm3_1',
               'decoder.avgpool',
               'decoder.flatten',
               'decoder.linear',
               'decoder.output'
               ]

comparison_mats_directory = "C:\\Users\\Asus\\Desktop\\Project\\CorNet-S\\RDM Comparisons B mats\\"
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
    plt.title("CorNet-S", fontsize=25)
    plt.ylabel("Correlation", fontsize=25)
    plt.plot(slices, big_values[g], color=color[g % 3], linewidth=3, label=type_of_image[g])
    plt.plot(slices, big_values[g + 1], color=color[(g + 1) % 3], linewidth=3, label=type_of_image[g + 1])
    plt.plot(slices, big_values[g + 2], color=color[(g + 2) % 3], linewidth=3, label=type_of_image[g + 2])
    plt.legend(fontsize=30)
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\CorNet-S\\RDM Comparisons MISC Plots\\' + str(kkk) + '.png',
        dpi=100)
    plt.close()
    kkk += 1
