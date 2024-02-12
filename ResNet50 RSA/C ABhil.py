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

k = 0
type_of_image = ['1_HF noF', '2_AF noF', '3_HBP noF', '4_ABP noF', '5_N noF', '6_MM noF', '7_HF hF', '8_AF hF',
                 '9_HBP hF',
                 '10_ABP hF', '11_N hF', '12_MM hF', '13_HF iF', '14_AF iF', '15_HBP iF', '16_ABP iF', '17_N iF',
                 '18_MM iF',
                 '19_HF LF', '20_AF LF', '21_HBP LF', '22_ABP LF', '23_N LF', '24_MM LF', '25_All Images']
model_name = 'ResNet50'
res50 = models.resnet50(pretrained=True)
slice_names = np.array([])  # a list of the names of all slices of the network
slice_names = np.append(slice_names, np.array(models.feature_extraction.get_graph_node_names(res50)[0]))

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monkey_matrix = [monke_brain_matrix[0:9], monke_brain_matrix[9:18], monke_brain_matrix[18:28],
                 monke_brain_matrix[28:37], monke_brain_matrix[np.r_[37:54, 71:74]], monke_brain_matrix[54:71],
                 monke_brain_matrix[74:80], monke_brain_matrix[80:83], monke_brain_matrix[83:86],
                 monke_brain_matrix[86:89], monke_brain_matrix[89:95], monke_brain_matrix[95:101],
                 monke_brain_matrix[101:107], monke_brain_matrix[107:110], monke_brain_matrix[110:113],
                 monke_brain_matrix[113:116], monke_brain_matrix[116:122], monke_brain_matrix[122:128],
                 monke_brain_matrix[128:134], monke_brain_matrix[134:137], monke_brain_matrix[137:140],
                 monke_brain_matrix[140:143], monke_brain_matrix[143:149], monke_brain_matrix[149:155],
                 monke_brain_matrix[0:155]]

# <editor-fold desc="calculating pairwise euclidean distances of brain and model and correlating them and saving the correlation of each layer and brain">
for mk in range(len(type_of_image)):
    monke_brain_matrix = monkey_matrix[mk]
    comparisons = np.array([])
    monke_rdm = scipy.spatial.distance.pdist(monke_brain_matrix, metric='euclidean')
    for i in range(len(slice_names)):
        sliceF_x_image = pd.read_pickle(
            r'C:\Users\Asus\Desktop\Project\ResNet50\CNN Slices Flattened by Image as pickle\\' + str(i).zfill(3) + '_' +
            slice_names[
                i] + '_for_all_images' + '.p')
        sliceF_x_image = sliceF_x_image.transpose()
        sliceF_matrix = [sliceF_x_image[0:9], sliceF_x_image[9:18], sliceF_x_image[18:28],
                         sliceF_x_image[28:37], sliceF_x_image[np.r_[37:54, 71:74]], sliceF_x_image[54:71],
                         sliceF_x_image[74:80], sliceF_x_image[80:83], sliceF_x_image[83:86],
                         sliceF_x_image[86:89], sliceF_x_image[89:95], sliceF_x_image[95:101],
                         sliceF_x_image[101:107], sliceF_x_image[107:110], sliceF_x_image[110:113],
                         sliceF_x_image[113:116], sliceF_x_image[116:122], sliceF_x_image[122:128],
                         sliceF_x_image[128:134], sliceF_x_image[134:137], sliceF_x_image[137:140],
                         sliceF_x_image[140:143], sliceF_x_image[143:149], sliceF_x_image[149:155],
                         sliceF_x_image[0:155]]

        sliceF_x_image = sliceF_matrix[k]
        slice_rdm = scipy.spatial.distance.pdist(sliceF_x_image, metric='euclidean')
        comparisons = np.append(comparisons, scipy.stats.pearsonr(monke_rdm, slice_rdm)[0])
        # comparisons.append(scipy.stats.pearsonr(monke_rdm, slice_rdm)[0])
    comparisons = np.reshape(comparisons, (-1, 1))
    scipy.io.savemat(
        'C:/Users/Asus/Desktop/Project/ResNet50/RDM Comparisons BAHIL mats/' + str(k + 1) + '_RDM_comparison_' +
        type_of_image[
            k] + '.mat',
        {'RDM_comparison': comparisons})
    i = 0
    k += 1
