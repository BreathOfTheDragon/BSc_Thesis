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
type_of_image = ['Human_Face', 'Animal_Face', 'Human_Body', 'Animal_Body', 'Natural', 'Man_Made', 'Face', 'Body',
                 'Animate', 'Inanimate', 'All']
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
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monkey_matrix = [monke_brain_matrix[0:9], monke_brain_matrix[9:18], monke_brain_matrix[18:28],
                 monke_brain_matrix[28:37],
                 monke_brain_matrix[np.r_[37:54, 71:74]], monke_brain_matrix[54:71], monke_brain_matrix[0:18],
                 monke_brain_matrix[18:37], monke_brain_matrix[0:37], monke_brain_matrix[np.r_[37:54, 71:74, 54:71]],
                 monke_brain_matrix[np.r_[0:37, 37:54, 71:74, 54:71]]]

for mk in range(len(type_of_image)):
    monke_brain_matrix = monkey_matrix[mk]
    comparisons = np.array([])
    monke_rdm = scipy.spatial.distance.pdist(monke_brain_matrix, metric='euclidean')
    for i in range(len(slice_names)):
        sliceF_x_image = pd.read_pickle(
            r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' +
            slice_names[
                i] + '_for_all_images' + '.p')
        sliceF_x_image = sliceF_x_image.transpose()
        sliceF_matrix = [sliceF_x_image[0:9], sliceF_x_image[9:18], sliceF_x_image[18:28],
                         sliceF_x_image[28:37],
                         sliceF_x_image[np.r_[37:54, 71:74]], sliceF_x_image[54:71], sliceF_x_image[0:18],
                         sliceF_x_image[18:37], sliceF_x_image[0:37],
                         sliceF_x_image[np.r_[37:54, 71:74, 54:71]],
                         sliceF_x_image[np.r_[0:37, 37:54, 71:74, 54:71]]]
        sliceF_x_image = sliceF_matrix[k]
        slice_rdm = scipy.spatial.distance.pdist(sliceF_x_image, metric='euclidean')
        comparisons = np.append(comparisons, scipy.stats.pearsonr(monke_rdm, slice_rdm)[0])
        # comparisons.append(scipy.stats.pearsonr(monke_rdm, slice_rdm)[0])
    comparisons = np.reshape(comparisons, (-1, 1))
    scipy.io.savemat(
        'C:/Users/Asus/Desktop/Project/CorNet-S 2/RDM Comparisons A mats/' + str(k + 1) + '_RDM_comparison_' +
        type_of_image[
            k] + '.mat',
        {'RDM_comparison': comparisons})
    i = 0
    k += 1
