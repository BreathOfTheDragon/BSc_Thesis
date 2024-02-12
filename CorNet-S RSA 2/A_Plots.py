import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import pickle
import os
import pandas as pd
import scipy
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

type_of_image = ['Human_face', 'Animal_Face', 'Human_Body', 'Animal_body', 'Natural', 'Man_Made', 'Face', 'Body',
                 'Animate', 'Inanimate', 'All']
color = ['black', 'lime', 'orange', 'blue', 'fuchsia', 'red', 'purple', 'grey', 'green', 'cyan', 'saddlebrown']

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

comparison_mats_directory = "C:\\Users\\Asus\\Desktop\\Project\\CorNet-S 2\\RDM Comparisons A mats\\"
comparison_mats = []
comparison_mat_no_mat = []
for u in range(len(os.listdir(comparison_mats_directory))):
    comparison_mats.append(
        scipy.io.loadmat(comparison_mats_directory + str(u + 1) + '_RDM_comparison_' + type_of_image[u]))
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
    plt.figure(figsize=(20, 20))
    plt.yticks(fontsize=20)
    plt.ylim(-1.0, 1.0)
    plt.plot(slices[1:], values[1:], color=color[l], linewidth=3, label=type_of_image[l])
    plt.legend(fontsize=15)
    plt.xticks(fontsize=10, rotation=90)
    plt.ylabel("Correlation", fontsize=25)
    plt.title("Correlation between brain region and CorNet-S Slices", fontsize=25)
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\CorNet-S 2\\RDM Comparisons A Plots\\' + str(l + 1) + '_' + type_of_image[l] + '.png',
        dpi=100)
    plt.close()

plt.figure(figsize=(20, 20))
plt.yticks(fontsize=20)
plt.ylim(-1.0, 1.0)
plt.xticks(fontsize=10, rotation=90)
plt.ylabel("Correlation", fontsize=25)
plt.title("Correlation between brain region and CorNet-S Slices", fontsize=25)
for g in range(len(type_of_image)):
    plt.plot(slices[1:], big_values[g][1:], color=color[g], linewidth=3, label=type_of_image[g])
plt.legend(fontsize=12)
plt.draw()
plt.savefig(
    'C:\\Users\\Asus\\Desktop\\Project\\CorNet-S 2\\RDM Comparisons A Plots\\Combined.png',
    dpi=100)
plt.close()

max_comparisons = []
max_indices = []
for w in range(len(comparison_mats)):
    comparisons = comparison_mats[w]['RDM_comparison'][1:]

    max_comparisons.append(np.amax(comparisons))
    max_indices.append(np.argmax(comparisons))

#becasue we want layer numbers to start with 1 , excluding X , and only considering NN layers
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
plt.savefig('C:\\Users\\Asus\\Desktop\\Project\\CorNet-S 2\\RDM Comparisons MAX Plots\\MAX_A_Types.png',
            dpi=100)
plt.close()
plt.figure(figsize=(25, 25))
plt.ylim(0, len(slice_names))
plt.bar(type_of_image, max_indices, color='red' )
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.ylabel("Best Layer", fontsize=25)
plt.title("Best CNN layer for each image type , excluding X ", fontsize=25)
plt.draw()
plt.savefig('C:\\Users\\Asus\\Desktop\\Project\\CorNet-S 2\\RDM Comparisons Best Layer Plots\\Best_Layer_A_Types.png',
            dpi=100)
