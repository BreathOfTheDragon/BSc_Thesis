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

# This code imports the data of brain neurons (352 neurons) for type BI categories (human_face , inanimate & ...)
# also imports data of all neurons (each layer has many neurons) of all layers of a NN for the same type BI categories
# then it calculates the RDM for both brain and the NN layer
# it calculates euclidean distance of pairwise vectors of brain
# a brain vector is the output of 1 image for 352 neurons
# then calculates euclidean distance of pairwise vectors of NN layer
# a model vector is the output of 1 image for many neurons
# then it calculates the Spearman rank correlation between the two lists of euclidean distances (RDMs)
# it does the correlation calculation between brain and each layer of NN
# in the end , plots the correlation of brain for type BI categories and each of the NN layers for type BI categories
# finally , we have a plot that shows correlation of brain data and each layer of NN
# we can decide if a certain layer has a high or low correlation with brain data
# this shows which layer is very similar to brain , or which layer is different from brain
# in the end we will have #(NN_layers) numbers , for example 22 for alex-net (excluding X)
# for example :
# we have 9 images of human face , brain has 352 outputs for each , NN layer has many outputs for each
# we calculate pairwise euclidean distances for both brain and model layer
# it gives us 9C2 = 36 distances
# then we correlate the 36 distances of brain with 36 distances of model layer
# each NN layer and brain correlation is a single number , and we have #(NN_layers) numbers

type_of_image = ['Human_Face_IF', 'Animal_Face_IF', 'Human_Body_IF', 'Animal_Body_IF', 'Natural_IF', 'Man_Made_IF',
                 'Face_IF', 'Body_IF', 'Animate_IF', 'Inanimate_IF', 'All_IF']
color = ['black', 'lime', 'orange', 'blue', 'fuchsia', 'red', 'purple', 'grey', 'green', 'cyan', 'saddlebrown']
model_name = 'MobileNet-V2'
k = 0
mobilenetv2 = models.mobilenet_v2(pretrained=True)
slice_names = np.array([])  # a list of the names of all slices of the network
slice_names = np.append(slice_names, np.array(models.feature_extraction.get_graph_node_names(mobilenetv2)[0]))

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monkey_matrix = [monke_brain_matrix[101:107], monke_brain_matrix[107:110], monke_brain_matrix[109:113],
                 monke_brain_matrix[113:116],
                 monke_brain_matrix[116: 122], monke_brain_matrix[122:128], monke_brain_matrix[101:110],
                 monke_brain_matrix[110:116], monke_brain_matrix[101:116], monke_brain_matrix[116: 128],
                 monke_brain_matrix[101: 128]]

# <editor-fold desc="calculating pairwise euclidean distances of brain and model and correlating them and saving the correlation of each layer and brain">
for mk in range(len(type_of_image)):
    monke_brain_matrix = monkey_matrix[mk]
    comparisons = np.array([])
    monke_rdm = scipy.spatial.distance.pdist(monke_brain_matrix, metric='euclidean')
    for i in range(len(slice_names)):
        sliceF_x_image = pd.read_pickle(
            r'C:\Users\Asus\Desktop\Project\MobileNet-V2\CNN Slices Flattened by Image as pickle\\' + str(i).zfill(3) + '_' +
            slice_names[
                i] + '_for_all_images' + '.p')
        sliceF_x_image = sliceF_x_image.transpose()
        sliceF_matrix = [sliceF_x_image[101:107], sliceF_x_image[107:110], sliceF_x_image[109:113],
                         sliceF_x_image[113:116],
                         sliceF_x_image[116: 122], sliceF_x_image[122:128], sliceF_x_image[101:110],
                         sliceF_x_image[110:116], sliceF_x_image[101:116],
                         sliceF_x_image[116: 128],
                         sliceF_x_image[101: 128]]
        sliceF_x_image = sliceF_matrix[k]
        slice_rdm = scipy.spatial.distance.pdist(sliceF_x_image, metric='euclidean')
        comparisons = np.append(comparisons, scipy.stats.pearsonr(monke_rdm, slice_rdm)[0])
        # comparisons.append(scipy.stats.pearsonr(monke_rdm, slice_rdm)[0])
    comparisons = np.reshape(comparisons, (-1, 1))
    scipy.io.savemat(
        'C:/Users/Asus/Desktop/Project/MobileNet-V2/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' +
        type_of_image[
            k] + '.mat',
        {'RDM_comparison': comparisons})
    i = 0
    k += 1
# </editor-fold>

# <editor-fold desc="plotting type BI correlations between brain and each NN layer , for each type BI category , each in 1 plot">
comparison_mats_directory = "C:\\Users\\Asus\\Desktop\\Project\\MobileNet-V2\\RDM Comparisons B mats\\"
comparison_mats = []
comparison_mat_no_mat = []
for u in range(len(type_of_image)):
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
    plt.xticks(np.arange(0, len(slices[1:]), 5), fontsize=10, rotation=90)
    plt.ylabel("Correlation", fontsize=25)
    plt.title("Correlation between brain region and MobileNet-V2 Slices", fontsize=25)
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\MobileNet-V2\\RDM Comparisons B Plots\\' + str(l + 1) + '_' + type_of_image[
            l] + '.png',
        dpi=100)
    plt.close()
# </editor-fold>

# <editor-fold desc="plotting type BI correlations between brain and each NN layer , for each type BI category , all in 1 plot">
plt.figure(figsize=(20, 20))
plt.yticks(fontsize=20)
plt.ylim(-1.0, 1.0)
plt.xticks(np.arange(0, len(slices[1:]), 5), fontsize=10, rotation=90)
plt.ylabel("Correlation", fontsize=25)
plt.title("Correlation between brain region and MobileNet-V2 Slices", fontsize=25)
for g in range(len(type_of_image)):
    plt.plot(slices[1:], big_values[g][1:], color=color[g], linewidth=3, label=type_of_image[g])
plt.legend(fontsize=12)
plt.draw()
plt.savefig(
    'C:\\Users\\Asus\\Desktop\\Project\\MobileNet-V2\\RDM Comparisons B Plots\\Combined_IF.png',
    dpi=100)
plt.close()
# </editor-fold>

# <editor-fold desc="plotting which layer has the highest correlation for each type BI category , aka best layer">
max_comparisons = []
max_indices = []
for w in range(len(comparison_mats)):
    comparisons = comparison_mats[w]['RDM_comparison'][1:]

    max_comparisons.append(np.amax(comparisons))
    max_indices.append(np.argmax(comparisons))

# becasue we want layer numbers to start with 1 , excluding X , and only considering NN layers
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
plt.title("Max correlation between brain region and image types for " + model_name, fontsize=25)
plt.draw()
plt.savefig('C:\\Users\\Asus\\Desktop\\Project\\MobileNet-V2\\RDM Comparisons MAX Plots\\MAX_BI_Types.png',
            dpi=100)
plt.close()
# </editor-fold>

# <editor-fold desc="plotting what is the correlation value of the best layer for each type BI category , aka max correlation">
plt.figure(figsize=(25, 25))
plt.ylim(0, len(slice_names))
plt.bar(type_of_image, max_indices, color='red')
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.ylabel("Best Layer", fontsize=25)
plt.title("Best CNN layer for each image type , excluding X for " + model_name, fontsize=25)
plt.draw()
plt.savefig('C:\\Users\\Asus\\Desktop\\Project\\MobileNet-V2\\RDM Comparisons Best Layer Plots\\Best_Layer_BI_Types.png',
            dpi=100)
plt.close()
# </editor-fold>
