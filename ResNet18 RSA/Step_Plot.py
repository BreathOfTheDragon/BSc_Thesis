import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import matplotlib as mpl
import io
import scipy
import scipy.io
import os

res18 = models.resnet18(pretrained=True)
slice_names = np.array([])  # a list of the names of all slices of the network
slice_names = np.append(slice_names, np.array(models.feature_extraction.get_graph_node_names(res18)[0]))

type_of_image = ['1_HF noF', '2_AF noF', '3_HBP noF', '4_ABP noF', '5_N noF', '6_MM noF', '7_HF hF', '8_AF hF',
                 '9_HBP hF', '10_ABP hF', '11_N hF', '12_MM hF', '13_HF iF', '14_AF iF', '15_HBP iF', '16_ABP iF',
                 '17_N iF', '18_MM iF', '19_HF LF', '20_AF LF', '21_HBP LF', '22_ABP LF', '23_N LF', '24_MM LF',
                 '25_All Images']

types = ['HF noF', 'AF noF', 'HBP noF', 'ABP noF', 'N noF', 'MM noF', 'HF hF', 'AF hF',
         'HBP hF',
         'ABP hF', 'N hF', 'MM hF', 'HF iF', 'AF iF', 'HBP iF', 'ABP iF', 'N iF',
         'MM iF',
         'HF LF', 'AF LF', 'HBP LF', 'ABP LF', 'N LF', 'MM LF', 'All Images']

color = ['red', 'green']

res18 = models.resnet18(pretrained=True)
slice_names = np.array([])  # a list of the names of all slices of the network
slice_names = np.append(slice_names, np.array(models.feature_extraction.get_graph_node_names(res18)[0]))

comparison_mats_directory = "E:\\Project\\ResNet18\\RDM Comparison mats\\"
comparison_mats = []
comparison_mat_no_mat = []
u = 0
for u in range(len(os.listdir(comparison_mats_directory))):
    print(u)
    print(type_of_image[u])
    print(comparison_mats_directory + str(u + 1) + '_RDM_comparison_' + type_of_image[u])
    comparison_mats.append(
        scipy.io.loadmat(comparison_mats_directory + str(u + 1) + '_RDM_comparison_' + type_of_image[u]))

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
    plt.figure(figsize=(20, 20))
    plt.yticks(fontsize=20)
    # creating the step plot
    plt.ylim(-1.0, 1.0)
    plt.plot(slices, values, color='navy', linewidth=7)
    plt.xticks(fontsize=10, rotation=90)
    # plt.xlabel("Slices")
    plt.ylabel("Correlation", fontsize=25)
    plt.title("Correlation between brain region and ResNet18 Slices", fontsize=25)
    plt.draw()
    plt.savefig(
        'E:\\Project\\ResNet18\\RDM Comparison Plots\\' + type_of_image[l] + '_StepPlot.png',
        dpi=200)
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

plt.figure(figsize=(20, 20))
# creating the step plot
plt.ylim(-1.0, 1.0)
plt.bar(types, max_comparisons, color='turquoise')
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.ylabel("Correlation", fontsize=25)
plt.title("Max correlation between brain region and image types", fontsize=25)
plt.draw()
plt.savefig('E:\\Project\\ResNet18\\RDM Comparison MAX Plot\\MAX_All_Types.png',
            dpi=200)
plt.close()
plt.figure(figsize=(20, 20))
plt.ylim(0, 69)
plt.bar(types, max_indices, color='red')
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.ylabel("Best Layer", fontsize=25)
plt.title("Best CNN layer for each image type , excluding X ", fontsize=25)
plt.draw()
plt.savefig('E:\\Project\\ResNet18\\RDM Comparison Best Layer Plot\\Best_Layer_All_Types.png',
            dpi=200)
