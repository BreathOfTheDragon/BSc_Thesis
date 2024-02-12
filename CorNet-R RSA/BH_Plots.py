import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import pickle
import os
import pandas as pd
import scipy
from torchvision import models

type_of_image = ['Human_Face_HF', 'Animal_Face_HF', 'Human_Body_HF', 'Animal_Body_HF', 'Natural_HF', 'Man_Made_HF',
                 'Face_HF', 'Body_HF',
                 'Animate_HF', 'Inanimate_HF', 'All_HF']
color = ['black', 'lime', 'orange', 'blue', 'fuchsia', 'red', 'purple', 'grey', 'green', 'cyan', 'saddlebrown']

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
        scipy.io.loadmat(comparison_mats_directory + str(u + 1) + '_RDM_comparison_' + type_of_image[u]))
big_values = []
for l in range(len(comparison_mats)):
    comparisons = comparison_mats[l]['RDM_comparison' + str(l + 1)]
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
    plt.figure(figsize=(20, 20))
    plt.yticks(fontsize=20)
    plt.ylim(-1.0, 1.0)
    plt.plot(slices, values, color=color[l], linewidth=3, label=type_of_image[l])
    plt.legend(fontsize=15)
    plt.xticks(fontsize=10, rotation=90)
    plt.ylabel("Correlation", fontsize=25)
    plt.title("Correlation between brain region and CorNet-R Slices", fontsize=25)
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\CorNet-R\\RDM Comparisons B Plots\\' + str(l + 1) + '_' + type_of_image[l] + '.png',
        dpi=100)
    plt.close()

plt.figure(figsize=(20, 20))
plt.yticks(fontsize=20)
plt.ylim(-1.0, 1.0)
plt.xticks(fontsize=10, rotation=90)
plt.ylabel("Correlation", fontsize=25)
plt.title("Correlation between brain region and CorNet-R Slices", fontsize=25)
for g in range(11):
    plt.plot(slices, big_values[g], color=color[g], linewidth=3, label=type_of_image[g])
plt.legend(fontsize=12)
plt.draw()
plt.savefig(
    'C:\\Users\\Asus\\Desktop\\Project\\CorNet-R\\RDM Comparisons B Plots\\Combined_HF.png',
    dpi=100)
plt.close()

max_comparisons = []
max_indices = []
for w in range(len(comparison_mats)):
    comparisons = comparison_mats[w]['RDM_comparison' + str(w + 1)][1:]

    max_comparisons.append(np.amax(comparisons))
    max_indices.append(np.argmax(comparisons))

for p in range(len(max_indices)):
    max_indices[p] = max_indices[p] + 1
print(max_comparisons)
print(max_indices)

plt.figure(figsize=(25, 25))
# creating the step plot
plt.ylim(-1.0, 1.0)
plt.bar(type_of_image, max_comparisons, color='turquoise')
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.ylabel("Correlation", fontsize=25)
plt.title("Max correlation between brain region and image types", fontsize=25)
plt.draw()
plt.savefig('C:\\Users\\Asus\\Desktop\\Project\\CorNet-R\\RDM Comparisons MAX Plots\\MAX_BH_Types.png',
            dpi=100)
plt.close()
plt.figure(figsize=(25, 25))
plt.ylim(0, len(slice_names))
plt.bar(type_of_image, max_indices, color='red')
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.ylabel("Best Layer", fontsize=25)
plt.title("Best CNN layer for each image type , excluding X ", fontsize=25)
plt.draw()
plt.savefig('C:\\Users\\Asus\\Desktop\\Project\\CorNet-R\\RDM Comparisons Best Layer Plots\\Best_Layer_BH_Types.png',
            dpi=100)
